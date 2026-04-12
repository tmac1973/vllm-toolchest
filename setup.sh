#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# vllm-toolchest setup — distro-agnostic, runtime-agnostic setup and launcher
# ─────────────────────────────────────────────────────────────────────────────

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly CDI_SYSTEM_DIR="/etc/cdi"
readonly CDI_USER_DIR="${HOME}/.config/containers/cdi"

# ─── Global state (populated by detect_* functions) ──────────────────────────

GPU_VENDOR=""           # cuda, rocm
GPU_INFO=""             # human-readable GPU description
AMD_GFX_VERSION=""      # HSA_OVERRIDE_GFX_VERSION value (empty = not needed)
HOST_VIDEO_GID=""       # host video group GID
HOST_RENDER_GID=""      # host render group GID

VLLMCTL_PORT="3000"
VLLMCTL_INFERENCE_PORT="8000"
VLLMCTL_MODELS_DIR=""

CONTAINER_CMD=""        # docker or podman
COMPOSE_CMD=""          # "docker compose" or "podman-compose" or "podman compose"
CONTAINER_VERSION=""
COMPOSE_VERSION=""

DISTRO_ID=""
DISTRO_NAME=""
DISTRO_FAMILY=""
PKG_MANAGER=""

ACTIONS=()
PREREQS=()

# ─── Utility ─────────────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log()   { echo -e "${BLUE}==>${NC} $*"; }
ok()    { echo -e "${GREEN}  ✓${NC} $*"; }
warn()  { echo -e "${YELLOW}  ⚠${NC} $*" >&2; }
err()   { echo -e "${RED}  ✗${NC} $*" >&2; }
fatal() { err "$@"; exit 1; }

need_cmd() {
    command -v "$1" &>/dev/null
}

run_sudo() {
    if [[ $EUID -eq 0 ]]; then
        "$@"
    else
        sudo "$@"
    fi
}

prompt_confirm() {
    local prompt="$1"
    local answer
    read -rp "$(echo -e "${BOLD}${prompt}${NC} [Y/n] ")" answer
    case "${answer:-Y}" in
        [Yy]*|"") return 0 ;;
        *)        return 1 ;;
    esac
}

# ─── Detection: GPU ──────────────────────────────────────────────────────────

detect_host_gpu_gids() {
    if need_cmd getent; then
        HOST_VIDEO_GID="$(getent group video 2>/dev/null | cut -d: -f3)" || true
        HOST_RENDER_GID="$(getent group render 2>/dev/null | cut -d: -f3)" || true
    fi
    if [[ -z "$HOST_VIDEO_GID" ]]; then
        HOST_VIDEO_GID="$(grep '^video:' /etc/group 2>/dev/null | cut -d: -f3)" || true
    fi
    if [[ -z "$HOST_RENDER_GID" ]]; then
        HOST_RENDER_GID="$(grep '^render:' /etc/group 2>/dev/null | cut -d: -f3)" || true
    fi
}

detect_amd_gfx_version() {
    local gfx_target=""
    if need_cmd rocminfo; then
        gfx_target="$(rocminfo 2>/dev/null | grep -oP 'gfx\d+' | head -1)" || true
    fi
    [[ -z "$gfx_target" ]] && return
    case "$gfx_target" in
        gfx1200|gfx1201|gfx1100|gfx1101|gfx1102|gfx1103)
            AMD_GFX_VERSION="" ;;
        gfx1030|gfx1031|gfx1032|gfx1033|gfx1034|gfx1035|gfx1036)
            AMD_GFX_VERSION="" ;;
        gfx1010|gfx1011|gfx1012|gfx1013)
            AMD_GFX_VERSION="10.1.0" ;;
        gfx900|gfx902|gfx904|gfx906|gfx908|gfx909)
            AMD_GFX_VERSION="9.0.0" ;;
        *) AMD_GFX_VERSION="" ;;
    esac
}

detect_gpu() {
    if need_cmd nvidia-smi; then
        if nvidia-smi --query-gpu=name --format=csv,noheader &>/dev/null; then
            GPU_VENDOR="cuda"
            GPU_INFO="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || true)"
            GPU_INFO="${GPU_INFO%%$'\n'*}"
            return
        fi
    fi
    if [[ -e /dev/nvidia0 ]]; then
        GPU_VENDOR="cuda"
        GPU_INFO="NVIDIA GPU detected (nvidia-smi unavailable)"
        return
    fi
    if [[ -e /dev/kfd ]]; then
        GPU_VENDOR="rocm"
        GPU_INFO="AMD GPU"
        if need_cmd rocminfo; then
            local name
            name="$(rocminfo 2>/dev/null | grep 'Marketing Name' | sed 's/.*: *//' \
                | grep -iE 'Radeon|Instinct|FirePro' | head -1)" || true
            [[ -n "$name" ]] && GPU_INFO="$name"
        fi
        detect_amd_gfx_version
        detect_host_gpu_gids
        return
    fi
    GPU_VENDOR="cuda"
    GPU_INFO="No GPU detected (defaulting to CUDA)"
}

# ─── Detection: Container runtime ────────────────────────────────────────────

detect_container_runtime() {
    local user_override="${RUNTIME:-}"

    if [[ -n "$user_override" ]]; then
        case "$user_override" in
            docker) need_cmd docker || fatal "RUNTIME=docker but docker not found"; CONTAINER_CMD="docker" ;;
            podman) need_cmd podman || fatal "RUNTIME=podman but podman not found"; CONTAINER_CMD="podman" ;;
            *) fatal "Unknown RUNTIME=$user_override" ;;
        esac
    else
        if need_cmd docker && docker info &>/dev/null 2>&1; then
            if docker --version 2>/dev/null | grep -qi podman; then
                CONTAINER_CMD="podman"
            else
                CONTAINER_CMD="docker"
            fi
        elif need_cmd podman; then
            CONTAINER_CMD="podman"
        else
            fatal "No container runtime found. Install Docker or Podman."
        fi
    fi

    CONTAINER_VERSION="$($CONTAINER_CMD --version 2>/dev/null || true)"
    CONTAINER_VERSION="${CONTAINER_VERSION%%$'\n'*}"

    if [[ "$CONTAINER_CMD" == "docker" ]]; then
        if docker compose version &>/dev/null 2>&1; then
            COMPOSE_CMD="docker compose"
        elif need_cmd docker-compose; then
            COMPOSE_CMD="docker-compose"
        else
            fatal "Docker found but no compose plugin"
        fi
    else
        if podman compose version &>/dev/null 2>&1; then
            COMPOSE_CMD="podman compose"
        elif need_cmd podman-compose; then
            COMPOSE_CMD="podman-compose"
        else
            fatal "Podman found but no compose command"
        fi
    fi
    COMPOSE_VERSION="$($COMPOSE_CMD version 2>/dev/null || $COMPOSE_CMD --version 2>/dev/null || true)"
    COMPOSE_VERSION="${COMPOSE_VERSION%%$'\n'*}"
}

# ─── Detection: Linux distribution ───────────────────────────────────────────

detect_distro() {
    [[ ! -f /etc/os-release ]] && fatal "Cannot detect distribution"
    # shellcheck disable=SC1091
    source /etc/os-release
    DISTRO_ID="${ID:-unknown}"
    DISTRO_NAME="${PRETTY_NAME:-$DISTRO_ID}"
    local id_like="${ID_LIKE:-}"

    case "$DISTRO_ID" in
        debian|ubuntu|pop|linuxmint) DISTRO_FAMILY="debian"; PKG_MANAGER="apt" ;;
        fedora|rhel|centos|rocky|alma|nobara) DISTRO_FAMILY="fedora"; PKG_MANAGER="dnf" ;;
        arch|cachyos|endeavouros|manjaro) DISTRO_FAMILY="arch"; PKG_MANAGER="pacman" ;;
        opensuse-leap|opensuse-tumbleweed) DISTRO_FAMILY="suse"; PKG_MANAGER="zypper" ;;
        *)
            if [[ "$id_like" == *"debian"* || "$id_like" == *"ubuntu"* ]]; then
                DISTRO_FAMILY="debian"; PKG_MANAGER="apt"
            elif [[ "$id_like" == *"fedora"* || "$id_like" == *"rhel"* ]]; then
                DISTRO_FAMILY="fedora"; PKG_MANAGER="dnf"
            elif [[ "$id_like" == *"arch"* ]]; then
                DISTRO_FAMILY="arch"; PKG_MANAGER="pacman"
            else
                DISTRO_FAMILY="unknown"; PKG_MANAGER=""
            fi
            ;;
    esac
}

# ─── Prerequisites ────────────────────────────────────────────────────────────

has_nvidia_toolkit() { need_cmd nvidia-ctk; }
has_cdi_spec() { [[ -f "$CDI_SYSTEM_DIR/nvidia.yaml" ]] || [[ -f "$CDI_USER_DIR/nvidia.yaml" ]]; }
docker_has_nvidia_runtime() { docker info 2>/dev/null | grep -qi "nvidia"; }

check_prerequisites() {
    PREREQS=()
    ACTIONS=()

    if [[ "$GPU_VENDOR" == "cuda" ]]; then
        if ! has_nvidia_toolkit; then
            PREREQS+=("install_nvidia_toolkit")
            ACTIONS+=("Install NVIDIA Container Toolkit")
        fi
        if [[ "$CONTAINER_CMD" == "docker" ]]; then
            if ! docker_has_nvidia_runtime; then
                PREREQS+=("configure_docker_nvidia")
                ACTIONS+=("Configure Docker NVIDIA runtime")
            fi
        elif [[ "$CONTAINER_CMD" == "podman" ]]; then
            if ! has_cdi_spec; then
                PREREQS+=("generate_cdi_spec")
                ACTIONS+=("Generate NVIDIA CDI spec for Podman")
            fi
        fi
    fi

    ACTIONS+=("Build container image (Dockerfile.${GPU_VENDOR})")
    ACTIONS+=("Start vllm-toolchest")
}

install_nvidia_toolkit() {
    case "$PKG_MANAGER" in
        apt)
            log "Adding NVIDIA Container Toolkit apt repo..."
            run_sudo apt-get update -qq
            run_sudo apt-get install -y -qq curl gpg
            curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
                | run_sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
            curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
                | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
                | run_sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
            run_sudo apt-get update -qq
            run_sudo apt-get install -y nvidia-container-toolkit ;;
        dnf)
            curl -fsSL https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
                | run_sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo > /dev/null
            run_sudo dnf install -y nvidia-container-toolkit ;;
        pacman)
            run_sudo pacman -Sy --noconfirm nvidia-container-toolkit ;;
        zypper)
            run_sudo zypper ar -f https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo nvidia-container-toolkit 2>/dev/null || true
            run_sudo zypper --gpg-auto-import-keys install -y nvidia-container-toolkit ;;
        *) fatal "Cannot install NVIDIA Container Toolkit: unsupported package manager" ;;
    esac
    ok "NVIDIA Container Toolkit installed"
}

configure_docker_nvidia() {
    log "Configuring Docker NVIDIA runtime..."
    run_sudo nvidia-ctk runtime configure --runtime=docker
    run_sudo systemctl restart docker
    ok "Docker NVIDIA runtime configured"
}

generate_cdi_spec() {
    log "Generating NVIDIA CDI spec..."
    run_sudo mkdir -p "$CDI_SYSTEM_DIR"
    run_sudo nvidia-ctk cdi generate --output="$CDI_SYSTEM_DIR/nvidia.yaml"
    ok "CDI spec written to $CDI_SYSTEM_DIR/nvidia.yaml"
}

install_prerequisites() {
    for prereq in "${PREREQS[@]}"; do
        case "$prereq" in
            install_nvidia_toolkit)  install_nvidia_toolkit ;;
            configure_docker_nvidia) configure_docker_nvidia ;;
            generate_cdi_spec)       generate_cdi_spec ;;
        esac
    done
}

# ─── Port configuration ──────────────────────────────────────────────────────

is_port_available() {
    local port="$1"
    if need_cmd ss; then
        ! ss -tlnH "sport = :${port}" 2>/dev/null | grep -q .
    else
        return 0
    fi
}

load_env_ports() {
    local env_file="${SCRIPT_DIR}/.env"
    if [[ -f "$env_file" ]]; then
        local val
        val="$(grep '^VLLMCTL_PORT=' "$env_file" 2>/dev/null | cut -d= -f2)" || true
        [[ -n "$val" ]] && VLLMCTL_PORT="$val" || true
        val="$(grep '^VLLMCTL_INFERENCE_PORT=' "$env_file" 2>/dev/null | cut -d= -f2)" || true
        [[ -n "$val" ]] && VLLMCTL_INFERENCE_PORT="$val" || true
        val="$(grep '^VLLMCTL_MODELS_DIR=' "$env_file" 2>/dev/null | cut -d= -f2)" || true
        [[ -n "$val" ]] && VLLMCTL_MODELS_DIR="$val" || true
    fi
}

prompt_ports() {
    echo ""
    echo -e "${BOLD}Port configuration${NC}"
    echo "  Management UI:  ${VLLMCTL_PORT}"
    echo "  Inference API:  ${VLLMCTL_INFERENCE_PORT}"

    local ports_ok=true
    if ! is_port_available "$VLLMCTL_PORT"; then
        warn "Port ${VLLMCTL_PORT} is already in use"
        ports_ok=false
    fi
    if ! is_port_available "$VLLMCTL_INFERENCE_PORT"; then
        warn "Port ${VLLMCTL_INFERENCE_PORT} is already in use"
        ports_ok=false
    fi

    if [[ "$ports_ok" == true ]] && prompt_confirm "Use these ports?"; then
        return
    fi

    local port
    while true; do
        read -rp "  Management UI port [${VLLMCTL_PORT}]: " port
        port="${port:-$VLLMCTL_PORT}"
        if [[ "$port" =~ ^[0-9]+$ ]] && (( port >= 1 && port <= 65535 )); then
            VLLMCTL_PORT="$port"; break
        fi
        err "Invalid port"
    done
    while true; do
        read -rp "  Inference API port [${VLLMCTL_INFERENCE_PORT}]: " port
        port="${port:-$VLLMCTL_INFERENCE_PORT}"
        if [[ "$port" =~ ^[0-9]+$ ]] && (( port >= 1 && port <= 65535 )); then
            VLLMCTL_INFERENCE_PORT="$port"; break
        fi
        err "Invalid port"
    done
}

prompt_models_dir() {
    echo ""
    echo -e "${BOLD}Model storage${NC}"
    if [[ -n "$VLLMCTL_MODELS_DIR" ]]; then
        echo "  Current: ${VLLMCTL_MODELS_DIR}"
    else
        echo "  Current: Docker volume (default)"
    fi
    echo "  Mount a host directory so models persist across container rebuilds."

    local path
    read -rp "  Host path [${VLLMCTL_MODELS_DIR:-none}]: " path
    if [[ -z "$path" ]]; then return; fi
    if [[ "$path" == "none" || "$path" == "-" ]]; then
        VLLMCTL_MODELS_DIR=""; return
    fi
    path="${path/#\~/$HOME}"
    [[ "$path" != /* ]] && path="$(cd "$SCRIPT_DIR" && realpath -m "$path")"
    [[ ! -d "$path" ]] && mkdir -p "$path"
    VLLMCTL_MODELS_DIR="$path"
    export VLLMCTL_MODELS_DIR
}

# ─── Container operations ────────────────────────────────────────────────────

compose_file() {
    if [[ "$GPU_VENDOR" == "rocm" ]]; then
        echo "docker-compose.yml"
    else
        echo "docker-compose.cuda.yml"
    fi
}

compose_cmd() {
    local cmd="$COMPOSE_CMD -f $(compose_file)"
    if [[ -n "${VLLMCTL_MODELS_DIR:-}" ]]; then
        cmd+=" -f docker-compose.models.yml"
    fi
    echo "$cmd"
}

write_env_file() {
    local env_file="${SCRIPT_DIR}/.env"
    : > "$env_file"
    echo "VLLMCTL_PORT=${VLLMCTL_PORT}" >> "$env_file"
    echo "VLLMCTL_INFERENCE_PORT=${VLLMCTL_INFERENCE_PORT}" >> "$env_file"
    if [[ -n "$VLLMCTL_MODELS_DIR" ]]; then
        echo "VLLMCTL_MODELS_DIR=${VLLMCTL_MODELS_DIR}" >> "$env_file"
    fi
    if [[ -n "$AMD_GFX_VERSION" ]]; then
        echo "HSA_OVERRIDE_GFX_VERSION=${AMD_GFX_VERSION}" >> "$env_file"
    fi
    if [[ -n "$HOST_VIDEO_GID" ]]; then
        echo "HOST_VIDEO_GID=${HOST_VIDEO_GID}" >> "$env_file"
    fi
    if [[ -n "$HOST_RENDER_GID" ]]; then
        echo "HOST_RENDER_GID=${HOST_RENDER_GID}" >> "$env_file"
    fi
}

container_install() {
    write_env_file
    $(compose_cmd) up -d --build
}

container_up() {
    $(compose_cmd) up -d
}

container_down() {
    $(compose_cmd) down
}

container_rebuild() {
    container_down
    $CONTAINER_CMD rm vllm-toolchest 2>/dev/null || true
    write_env_file
    $(compose_cmd) build --no-cache
    $(compose_cmd) up -d
}

container_quick_rebuild() {
    container_down
    write_env_file
    $(compose_cmd) up -d --build
}

container_logs() {
    $(compose_cmd) logs -f
}

# ─── Summary ─────────────────────────────────────────────────────────────────

print_summary() {
    echo ""
    echo -e "${BOLD}════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}  vllm-toolchest setup${NC}"
    echo -e "${BOLD}════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "  ${CYAN}GPU${NC}           ${GPU_INFO}"
    echo -e "  ${CYAN}Backend${NC}       ${GPU_VENDOR}"
    echo -e "  ${CYAN}Runtime${NC}       ${CONTAINER_VERSION}"
    echo -e "  ${CYAN}Compose${NC}       ${COMPOSE_VERSION}"
    echo -e "  ${CYAN}Distro${NC}        ${DISTRO_NAME}"
    echo -e "  ${CYAN}Compose file${NC}  $(compose_file)"
    echo -e "  ${CYAN}UI port${NC}       ${VLLMCTL_PORT}"
    echo -e "  ${CYAN}Inference port${NC} ${VLLMCTL_INFERENCE_PORT}"
    if [[ -n "$VLLMCTL_MODELS_DIR" ]]; then
        echo -e "  ${CYAN}Models dir${NC}    ${VLLMCTL_MODELS_DIR}"
    fi
    if [[ -n "$AMD_GFX_VERSION" ]]; then
        echo -e "  ${CYAN}HSA Override${NC}  ${AMD_GFX_VERSION}"
    fi
    echo ""
    if [[ ${#ACTIONS[@]} -gt 0 ]]; then
        echo -e "  ${BOLD}Actions:${NC}"
        local i=1
        for action in "${ACTIONS[@]}"; do
            echo "    ${i}. ${action}"
            ((i++))
        done
        echo ""
    fi
    if [[ ${#PREREQS[@]} -gt 0 ]]; then
        echo -e "  ${YELLOW}Note:${NC} Prerequisite steps require sudo"
        echo ""
    fi
}

# ─── Main ─────────────────────────────────────────────────────────────────────

usage() {
    cat <<'USAGE'
vllm-toolchest setup — auto-detect GPU + container runtime, build & run

Usage: ./setup.sh <command>

Lifecycle:
  install     Detect everything, install prerequisites, build image, start
  quick       Fast rebuild — reuse cached base layers, rebuild Go code only
  rebuild     Full rebuild with no cache
  uninstall   Stop and remove container + image

Runtime:
  up          Start a stopped container
  down        Stop the container
  logs        Follow container logs

Info:
  status      Show detected environment, then exit
  detect      Print detected GPU backend (cuda/rocm) and exit
  help        Show this help

Environment:
  GPU=cuda|rocm          Override GPU detection
  RUNTIME=docker|podman  Override runtime detection
USAGE
}

main() {
    local command="${1:-help}"
    cd "$SCRIPT_DIR"

    case "$command" in
        install|quick|rebuild|uninstall|up|down|logs|detect|status) ;;
        -h|--help|help) usage; exit 0 ;;
        *) err "Unknown command: $command"; usage; exit 1 ;;
    esac

    if [[ -n "${GPU:-}" ]]; then
        GPU_VENDOR="$GPU"
        GPU_INFO="(manually set: $GPU)"
    else
        detect_gpu
    fi

    if [[ "$command" == "detect" ]]; then
        echo "$GPU_VENDOR"
        exit 0
    fi

    detect_container_runtime
    detect_distro
    load_env_ports

    case "$command" in
        up)     container_up;   ok "vllm-toolchest started"; exit 0 ;;
        down)   container_down; ok "vllm-toolchest stopped"; exit 0 ;;
        logs)   container_logs; exit 0 ;;
        quick)
            log "Quick rebuild (cached)..."
            container_quick_rebuild
            ok "vllm-toolchest is running"
            echo "  Web UI: http://localhost:${VLLMCTL_PORT}"
            exit 0
            ;;
        uninstall)
            container_down
            $CONTAINER_CMD rm vllm-toolchest 2>/dev/null || true
            ok "vllm-toolchest removed"
            exit 0
            ;;
    esac

    # install, rebuild, status
    check_prerequisites
    print_summary

    [[ "$command" == "status" ]] && exit 0

    prompt_ports
    prompt_models_dir

    if [[ ${#PREREQS[@]} -gt 0 ]]; then
        if prompt_confirm "Install prerequisites?"; then
            echo ""
            install_prerequisites
            echo ""
        fi
    fi

    if ! prompt_confirm "Build and start vllm-toolchest?"; then
        echo "Aborted."
        exit 0
    fi

    echo ""
    case "$command" in
        install) container_install ;;
        rebuild) container_rebuild ;;
    esac

    echo ""
    ok "vllm-toolchest is running"
    echo ""
    echo "  Web UI:     http://localhost:${VLLMCTL_PORT}"
    echo "  Inference:  http://localhost:${VLLMCTL_INFERENCE_PORT}"
    echo ""
    echo "  Logs:       ./setup.sh logs"
    echo "  Stop:       ./setup.sh down"
    echo "  Quick rebuild: ./setup.sh quick"
    echo ""
}

main "$@"

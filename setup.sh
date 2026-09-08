#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# vllm-toolchest setup — distro-agnostic, runtime-agnostic setup and launcher
# ─────────────────────────────────────────────────────────────────────────────

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly CDI_SYSTEM_DIR="/etc/cdi"
readonly CDI_USER_DIR="${HOME}/.config/containers/cdi"

readonly QUADLET_USER_DIR="${HOME}/.config/containers/systemd"
readonly QUADLET_SYSTEM_DIR="/etc/containers/systemd"
readonly PODMAN_SERVICE_NAME="vllm-toolchest"

# Base image for the radiance variant. KEEP IN SYNC with the defaults in
# Dockerfile.radiance and docker-compose.radiance.yml.
readonly RADIANCE_DEFAULT_IMAGE="docker.io/stilldeadcode/vllm-radiance:0.9.3"

# ─── Global state (populated by detect_* functions) ──────────────────────────

GPU_VENDOR=""           # cuda, rocm
GPU_INFO=""             # human-readable GPU description
BUILD_VARIANT=""        # generic, radiance — which image to build (rocm only)
GPU_DEVICES=""          # HIP_VISIBLE_DEVICES value; empty = use every GPU
AMD_GFX_TARGET=""       # detected gfx target (e.g. gfx1201) — passed as GPU_ARCH build arg
AMD_GFX_VERSION=""      # HSA_OVERRIDE_GFX_VERSION value (empty = not needed)
HOST_VIDEO_GID=""       # host video group GID
HOST_RENDER_GID=""      # host render group GID

VLLMCTL_PORT="3000"            # host port for management UI
VLLMCTL_INFERENCE_PORT="8000"  # host port for inference API
VLLMCTL_MODELS_DIR=""          # host path for model storage (empty = use docker volume)

CONTAINER_CMD=""        # docker or podman
COMPOSE_CMD=""          # "docker compose" or "podman-compose" or "podman compose"
CONTAINER_VERSION=""
COMPOSE_VERSION=""

DISTRO_ID=""            # debian, ubuntu, fedora, arch, cachyos, opensuse-leap, etc.
DISTRO_NAME=""          # Pretty name from os-release
DISTRO_FAMILY=""        # debian, fedora, arch, suse
PKG_MANAGER=""          # apt, dnf, pacman, zypper

ACTIONS=()              # list of human-readable actions to perform
PREREQS=()              # list of prerequisite action keys

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

# Portable existence checks. `image exists` / `container exists` are podman
# subcommands that Docker does not have -- on Docker they fail as "unknown
# command", which reads as "not present" and silently sends the caller down
# the wrong branch. `inspect` exists on both.
image_exists() {
    $CONTAINER_CMD image inspect "$1" >/dev/null 2>&1
}

container_exists() {
    $CONTAINER_CMD container inspect "$1" >/dev/null 2>&1
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

# Detect host video/render group GIDs for container device access.
# Container group_add needs the host's actual GIDs, not names,
# because the container's /etc/group may have different GID mappings.
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

# Detect AMD gfx target from rocminfo, set AMD_GFX_TARGET and whether
# HSA_OVERRIDE_GFX_VERSION is needed for ROCm compatibility.
detect_amd_gfx_version() {
    local gfx_target=""

    if need_cmd rocminfo; then
        gfx_target="$(rocminfo 2>/dev/null | grep -oP 'gfx\d+' | head -1)" || true
    fi

    [[ -z "$gfx_target" ]] && return
    AMD_GFX_TARGET="$gfx_target"

    # HSA override is only needed for GPUs not natively supported by ROCm 7.x.
    case "$gfx_target" in
        # RDNA 4 — native in ROCm 7.2+
        gfx1200|gfx1201)            AMD_GFX_VERSION="" ;;
        # RDNA 3 — native
        gfx1100|gfx1101|gfx1102|gfx1103)    AMD_GFX_VERSION="" ;;
        # RDNA 2 — native
        gfx1030|gfx1031|gfx1032|gfx1033|gfx1034|gfx1035|gfx1036)
                                    AMD_GFX_VERSION="" ;;
        # RDNA 1 — needs override
        gfx1010|gfx1011|gfx1012|gfx1013)    AMD_GFX_VERSION="10.1.0" ;;
        # Vega — needs override
        gfx900|gfx902|gfx904|gfx906|gfx908|gfx909)
                                    AMD_GFX_VERSION="9.0.0" ;;
        *)                          AMD_GFX_VERSION="" ;;
    esac
}

detect_gpu() {
    # NVIDIA: check for nvidia-smi AND that it can talk to a GPU
    if need_cmd nvidia-smi; then
        if nvidia-smi --query-gpu=name --format=csv,noheader &>/dev/null; then
            GPU_VENDOR="cuda"
            GPU_INFO="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || true)"
            GPU_INFO="${GPU_INFO%%$'\n'*}"
            return
        fi
    fi
    # NVIDIA: fallback — device node exists but nvidia-smi missing/broken
    if [[ -e /dev/nvidia0 ]]; then
        GPU_VENDOR="cuda"
        GPU_INFO="NVIDIA GPU detected (nvidia-smi unavailable)"
        return
    fi

    # AMD: check for ROCm kernel driver
    if [[ -e /dev/kfd ]]; then
        GPU_VENDOR="rocm"
        GPU_INFO="AMD GPU"
        if need_cmd rocminfo; then
            local name
            name="$(rocminfo 2>/dev/null | grep 'Marketing Name' | sed 's/.*: *//' \
                | grep -iE 'Radeon|Instinct|FirePro' | head -1)" || true
            [[ -n "$name" ]] && GPU_INFO="$name"
        elif [[ -d /sys/class/drm ]]; then
            # Sysfs fallback — works without rocminfo installed on host
            for card_dir in /sys/class/drm/card[0-9]*/device; do
                if [[ -f "$card_dir/vendor" && "$(cat "$card_dir/vendor")" == "0x1002" ]]; then
                    GPU_INFO="AMD GPU ($(cat "$card_dir/device" 2>/dev/null || echo "unknown"))"
                    break
                fi
            done
        fi
        detect_amd_gfx_version
        detect_host_gpu_gids
        return
    fi

    # No GPU — vLLM has no meaningful CPU inference path, so fall back to CUDA
    # (the user probably just doesn't have drivers installed yet).
    GPU_VENDOR="cuda"
    GPU_INFO="No GPU detected (defaulting to CUDA)"
}

# vllm-radiance is compiled for a single GPU architecture and its prune step
# asserts it, so the variant is only offered on RDNA4.
radiance_supported() {
    [[ "$GPU_VENDOR" == "rocm" ]] && [[ "$AMD_GFX_TARGET" == "gfx1201" ]]
}

# Choose the image variant. Explicit VARIANT= always wins; otherwise default to
# the portable build and let install offer the RDNA4 one interactively.
detect_variant() {
    # VLLMCTL_VARIANT is the documented override, matching the key stored in
    # .env. Bare VARIANT is accepted as a convenience, but only when it names a
    # real variant: VARIANT is also an /etc/os-release field, so a value we do
    # not recognise belongs to somebody else and must not be a fatal error.
    local want="${VLLMCTL_VARIANT:-}"
    if [[ -z "$want" && "${VARIANT:-}" =~ ^(generic|radiance)$ ]]; then
        want="$VARIANT"
    fi

    if [[ -n "$want" ]]; then
        case "$want" in
            generic|radiance) BUILD_VARIANT="$want" ;;
            *) fatal "Unknown VLLMCTL_VARIANT=$want (expected: generic or radiance)" ;;
        esac
        if [[ "$BUILD_VARIANT" == "radiance" && "$GPU_VENDOR" != "rocm" ]]; then
            fatal "The radiance variant needs an AMD ROCm GPU (detected backend: $GPU_VENDOR)"
        fi
        return
    fi

    # Already chosen in a previous run — .env is the record of that decision.
    [[ -n "$BUILD_VARIANT" ]] && return

    BUILD_VARIANT="generic"
}

# ─── Detection: Container runtime ────────────────────────────────────────────

detect_container_runtime() {
    local user_override="${RUNTIME:-}"

    if [[ -n "$user_override" ]]; then
        case "$user_override" in
            docker)
                need_cmd docker || fatal "RUNTIME=docker specified but docker is not installed"
                CONTAINER_CMD="docker"
                ;;
            podman)
                need_cmd podman || fatal "RUNTIME=podman specified but podman is not installed"
                CONTAINER_CMD="podman"
                ;;
            *)
                fatal "Unknown RUNTIME=$user_override (expected: docker or podman)"
                ;;
        esac
    else
        # Auto-detect: prefer docker if available, fall back to podman
        if need_cmd docker && docker info &>/dev/null 2>&1; then
            # Make sure it's real Docker, not podman emulating docker
            if docker --version 2>/dev/null | grep -qi podman; then
                CONTAINER_CMD="podman"
            else
                CONTAINER_CMD="docker"
            fi
        elif need_cmd podman; then
            CONTAINER_CMD="podman"
        else
            fatal "No container runtime found. Install Docker or Podman first."
        fi
    fi

    CONTAINER_VERSION="$($CONTAINER_CMD --version 2>/dev/null || true)"
    CONTAINER_VERSION="${CONTAINER_VERSION%%$'\n'*}"

    if [[ "$CONTAINER_CMD" == "docker" ]]; then
        if docker compose version &>/dev/null 2>&1; then
            COMPOSE_CMD="docker compose"
            COMPOSE_VERSION="$(docker compose version 2>/dev/null || true)"
        elif need_cmd docker-compose; then
            COMPOSE_CMD="docker-compose"
            COMPOSE_VERSION="$(docker-compose --version 2>/dev/null || true)"
        else
            fatal "Docker is installed but neither 'docker compose' plugin nor 'docker-compose' found"
        fi
    else
        if podman compose version &>/dev/null 2>&1; then
            COMPOSE_CMD="podman compose"
            COMPOSE_VERSION="$(podman compose version 2>/dev/null || true)"
        elif need_cmd podman-compose; then
            COMPOSE_CMD="podman-compose"
            COMPOSE_VERSION="$(podman-compose --version 2>/dev/null || true)"
        else
            fatal "Podman is installed but neither 'podman compose' nor 'podman-compose' found"
        fi
    fi
    COMPOSE_VERSION="${COMPOSE_VERSION%%$'\n'*}"
}

# ─── Detection: Linux distribution ───────────────────────────────────────────

detect_distro() {
    if [[ ! -f /etc/os-release ]]; then
        fatal "Cannot detect distribution: /etc/os-release not found"
    fi

    # Read os-release in a SUBSHELL and hand back only the three fields we
    # want. It is a shell fragment, so sourcing it directly dumps every field
    # it defines into this script's namespace -- and Ubuntu Server defines
    # VARIANT="Server Edition", which clobbered our own VARIANT override and
    # made `./setup.sh install` fail on exactly that distro. printf %q keeps
    # values with spaces intact through the eval.
    local id pretty id_like
    eval "$(
        # shellcheck disable=SC1091
        . /etc/os-release
        printf 'id=%q\npretty=%q\nid_like=%q\n' \
            "${ID:-unknown}" "${PRETTY_NAME:-}" "${ID_LIKE:-}"
    )"

    DISTRO_ID="$id"
    DISTRO_NAME="${pretty:-$DISTRO_ID}"

    case "$DISTRO_ID" in
        debian|ubuntu|pop|linuxmint|elementary|zorin|kali)
            DISTRO_FAMILY="debian"; PKG_MANAGER="apt" ;;
        fedora|rhel|centos|rocky|alma|nobara)
            DISTRO_FAMILY="fedora"; PKG_MANAGER="dnf" ;;
        arch|cachyos|endeavouros|manjaro|garuda|artix)
            DISTRO_FAMILY="arch"; PKG_MANAGER="pacman" ;;
        opensuse-leap|opensuse-tumbleweed|sles)
            DISTRO_FAMILY="suse"; PKG_MANAGER="zypper" ;;
        *)
            if [[ "$id_like" == *"debian"* || "$id_like" == *"ubuntu"* ]]; then
                DISTRO_FAMILY="debian"; PKG_MANAGER="apt"
            elif [[ "$id_like" == *"fedora"* || "$id_like" == *"rhel"* ]]; then
                DISTRO_FAMILY="fedora"; PKG_MANAGER="dnf"
            elif [[ "$id_like" == *"arch"* ]]; then
                DISTRO_FAMILY="arch"; PKG_MANAGER="pacman"
            elif [[ "$id_like" == *"suse"* ]]; then
                DISTRO_FAMILY="suse"; PKG_MANAGER="zypper"
            else
                warn "Unknown distro: $DISTRO_ID (ID_LIKE=$id_like)"
                warn "Will skip automatic prerequisite installation"
                DISTRO_FAMILY="unknown"; PKG_MANAGER=""
            fi
            ;;
    esac
}

# ─── Prerequisite checks ─────────────────────────────────────────────────────

has_nvidia_toolkit() { need_cmd nvidia-ctk; }
has_cdi_spec() { [[ -f "$CDI_SYSTEM_DIR/nvidia.yaml" ]] || [[ -f "$CDI_USER_DIR/nvidia.yaml" ]]; }
docker_has_nvidia_runtime() { docker info 2>/dev/null | grep -qi "nvidia"; }

selinux_enforcing() {
    need_cmd getenforce && [[ "$(getenforce 2>/dev/null)" == "Enforcing" ]]
}

selinux_device_bool_set() {
    need_cmd getsebool && getsebool container_use_devices 2>/dev/null | grep -q "on"
}

check_prerequisites() {
    PREREQS=()
    ACTIONS=()

    if [[ "$GPU_VENDOR" == "cuda" ]]; then
        if ! has_nvidia_toolkit; then
            PREREQS+=("install_nvidia_toolkit")
            ACTIONS+=("Install NVIDIA Container Toolkit")
        fi

        if [[ "$CONTAINER_CMD" == "docker" ]]; then
            if has_nvidia_toolkit && ! docker_has_nvidia_runtime; then
                PREREQS+=("configure_docker_nvidia")
                ACTIONS+=("Configure Docker NVIDIA runtime + restart Docker daemon")
            elif ! has_nvidia_toolkit; then
                PREREQS+=("configure_docker_nvidia")
                ACTIONS+=("Configure Docker NVIDIA runtime + restart Docker daemon")
            fi
        fi

        if [[ "$CONTAINER_CMD" == "podman" ]]; then
            if ! has_cdi_spec; then
                PREREQS+=("generate_cdi_spec")
                ACTIONS+=("Generate NVIDIA CDI spec for Podman")
            fi
        fi
    fi

    # SELinux: needed for ROCm device access on Fedora/RHEL
    if [[ "$GPU_VENDOR" == "rocm" ]] && selinux_enforcing && ! selinux_device_bool_set; then
        PREREQS+=("selinux_device_bool")
        ACTIONS+=("Enable SELinux container_use_devices boolean")
    fi

    # The radiance image is compiled for gfx1201 only; on anything else it
    # will not run, so fail here rather than after a long pull.
    if [[ "$BUILD_VARIANT" == "radiance" ]] && ! radiance_supported; then
        if [[ -z "$AMD_GFX_TARGET" ]]; then
            warn "Could not detect a gfx target (rocminfo missing?); the radiance"
            warn "image only runs on gfx1201. Use VARIANT=generic if this is not RDNA4."
        else
            fatal "VARIANT=radiance requires gfx1201 (RDNA4); detected ${AMD_GFX_TARGET}. Use VARIANT=generic."
        fi
    fi

    ACTIONS+=("Build container image ($(dockerfile))")
    ACTIONS+=("Start vllm-toolchest service")
}

# ─── Prerequisite installation ───────────────────────────────────────────────

install_nvidia_toolkit_apt() {
    log "Adding NVIDIA Container Toolkit apt repository..."
    run_sudo apt-get update -qq
    run_sudo apt-get install -y -qq curl gpg

    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | run_sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        | run_sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null

    run_sudo apt-get update -qq
    run_sudo apt-get install -y nvidia-container-toolkit
    ok "NVIDIA Container Toolkit installed"
}

install_nvidia_toolkit_dnf() {
    log "Adding NVIDIA Container Toolkit dnf repository..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
        | run_sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo > /dev/null
    run_sudo dnf install -y nvidia-container-toolkit
    ok "NVIDIA Container Toolkit installed"
}

install_nvidia_toolkit_pacman() {
    log "Installing NVIDIA Container Toolkit via pacman..."
    run_sudo pacman -Sy --noconfirm nvidia-container-toolkit
    ok "NVIDIA Container Toolkit installed"
}

install_nvidia_toolkit_zypper() {
    log "Adding NVIDIA Container Toolkit zypper repository..."
    run_sudo zypper ar -f \
        https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
        nvidia-container-toolkit 2>/dev/null || true
    run_sudo zypper --gpg-auto-import-keys install -y nvidia-container-toolkit
    ok "NVIDIA Container Toolkit installed"
}

install_nvidia_toolkit() {
    case "$PKG_MANAGER" in
        apt)    install_nvidia_toolkit_apt ;;
        dnf)    install_nvidia_toolkit_dnf ;;
        pacman) install_nvidia_toolkit_pacman ;;
        zypper) install_nvidia_toolkit_zypper ;;
        *)      fatal "Cannot install NVIDIA Container Toolkit: unsupported package manager" ;;
    esac
}

configure_docker_nvidia() {
    log "Configuring Docker NVIDIA runtime..."
    run_sudo nvidia-ctk runtime configure --runtime=docker
    log "Restarting Docker daemon..."
    run_sudo systemctl restart docker
    ok "Docker NVIDIA runtime configured"
}

generate_cdi_spec() {
    log "Generating NVIDIA CDI spec..."
    run_sudo mkdir -p "$CDI_SYSTEM_DIR"
    run_sudo nvidia-ctk cdi generate --output="$CDI_SYSTEM_DIR/nvidia.yaml"
    ok "CDI spec written to $CDI_SYSTEM_DIR/nvidia.yaml"

    if need_cmd nvidia-ctk; then
        log "Verifying CDI devices..."
        nvidia-ctk cdi list 2>/dev/null | head -5 || true
    fi
}

selinux_device_bool() {
    log "Enabling SELinux container_use_devices..."
    run_sudo setsebool -P container_use_devices 1
    ok "SELinux boolean set"
}

install_prerequisites() {
    for prereq in "${PREREQS[@]}"; do
        case "$prereq" in
            install_nvidia_toolkit)  install_nvidia_toolkit ;;
            configure_docker_nvidia) configure_docker_nvidia ;;
            generate_cdi_spec)       generate_cdi_spec ;;
            selinux_device_bool)     selinux_device_bool ;;
            *)                       warn "Unknown prerequisite: $prereq" ;;
        esac
    done
}

# ─── Port configuration ──────────────────────────────────────────────────────

is_port_available() {
    local port="$1"
    if need_cmd ss; then
        ! ss -tlnH "sport = :${port}" 2>/dev/null | grep -q .
    elif need_cmd netstat; then
        ! netstat -tln 2>/dev/null | grep -q ":${port} "
    else
        return 0
    fi
}

prompt_ports() {
    echo ""
    echo -e "${BOLD}Port configuration${NC}"
    echo ""
    echo "  Current ports:"
    echo "    Management UI:  ${VLLMCTL_PORT}"
    echo "    Inference API:  ${VLLMCTL_INFERENCE_PORT}"
    echo ""

    local ports_ok=true
    if ! is_port_available "$VLLMCTL_PORT"; then
        warn "Port ${VLLMCTL_PORT} is already in use"
        ports_ok=false
    fi
    if ! is_port_available "$VLLMCTL_INFERENCE_PORT"; then
        warn "Port ${VLLMCTL_INFERENCE_PORT} is already in use"
        ports_ok=false
    fi

    if [[ "$ports_ok" == true ]]; then
        if prompt_confirm "Use these ports?"; then
            return
        fi
    else
        echo ""
        echo "  One or more ports are in use. Please choose alternative ports."
    fi

    echo ""
    local port
    while true; do
        read -rp "$(echo -e "  ${BOLD}Management UI port${NC} [${VLLMCTL_PORT}]: ")" port
        port="${port:-$VLLMCTL_PORT}"
        if [[ "$port" =~ ^[0-9]+$ ]] && (( port >= 1 && port <= 65535 )); then
            VLLMCTL_PORT="$port"
            break
        fi
        err "Invalid port number: $port"
    done

    while true; do
        read -rp "$(echo -e "  ${BOLD}Inference API port${NC} [${VLLMCTL_INFERENCE_PORT}]: ")" port
        port="${port:-$VLLMCTL_INFERENCE_PORT}"
        if [[ "$port" =~ ^[0-9]+$ ]] && (( port >= 1 && port <= 65535 )); then
            if [[ "$port" == "$VLLMCTL_PORT" ]]; then
                err "Cannot use the same port as management UI ($VLLMCTL_PORT)"
                continue
            fi
            VLLMCTL_INFERENCE_PORT="$port"
            break
        fi
        err "Invalid port number: $port"
    done
}

prompt_models_dir() {
    echo ""
    echo -e "${BOLD}Model storage${NC}"
    echo ""
    if [[ -n "$VLLMCTL_MODELS_DIR" ]]; then
        echo "  Current: ${VLLMCTL_MODELS_DIR} (host directory)"
    else
        echo "  Current: Docker volume (default)"
    fi
    echo ""
    echo "  Mount a host directory so models persist even if the"
    echo "  container volume is removed."
    echo ""

    local path
    read -rp "$(echo -e "  ${BOLD}Host path${NC} [${VLLMCTL_MODELS_DIR:-none}]: ")" path

    if [[ -z "$path" ]]; then
        return
    fi

    if [[ "$path" == "none" || "$path" == "-" ]]; then
        VLLMCTL_MODELS_DIR=""
        echo "  → Models will use Docker volume"
        return
    fi

    path="${path/#\~/$HOME}"

    if [[ "$path" != /* ]]; then
        path="$(cd "$SCRIPT_DIR" && realpath -m "$path" 2>/dev/null || echo "$SCRIPT_DIR/$path")"
    fi

    if [[ ! -d "$path" ]]; then
        log "Creating directory: $path"
        mkdir -p "$path" || { err "Cannot create $path"; return; }
    fi

    VLLMCTL_MODELS_DIR="$path"
    export VLLMCTL_MODELS_DIR
    echo "  → Models will be stored at: $path"
}

prompt_variant() {
    # Forced on the command line -- nothing to ask. Mirrors detect_variant's
    # handling, including ignoring an unrelated os-release VARIANT.
    if [[ -n "${VLLMCTL_VARIANT:-}" ]] || [[ "${VARIANT:-}" =~ ^(generic|radiance)$ ]]; then
        return
    fi
    if ! radiance_supported; then
        return  # not RDNA4 — only the generic image is buildable here
    fi

    echo ""
    echo -e "${BOLD}RDNA4 detected (${AMD_GFX_TARGET})${NC}"
    echo ""
    echo    "  There is a second image for this card: vllm-radiance, a from-source"
    echo    "  vLLM stack hand-tuned for gfx1201 — custom attention, GEMM and"
    echo    "  all-reduce kernels, tuned FP8 and MoE configs, and MTP drafting."
    echo    "  It installs much faster too, since it pulls a prebuilt base instead"
    echo    "  of compiling ROCm and vLLM from source."
    echo ""
    echo    "  The trade-off: it pins vLLM and transformers, so support for models"
    echo    "  newer than that release is frozen. The generic image tracks vLLM main."
    echo ""
    echo    "  Third-party project, credited in README.md:"
    echo    "  https://codeberg.org/StillDeadcode/vllm-radiance"
    echo ""

    if prompt_confirm "  Build the radiance variant?"; then
        BUILD_VARIANT="radiance"
    else
        BUILD_VARIANT="generic"
    fi
    echo "  → Building the ${BUILD_VARIANT} image"
}

# List AMD GPUs in HIP enumeration order, one per line as:
#   <hip-index>\t<pci-address>\t<vram-GiB>
#
# HIP enumerates by PCI bus order, so sorting the render nodes by their resolved
# PCI address is what makes the index we print here the same index
# HIP_VISIBLE_DEVICES expects. Reading sysfs rather than rocminfo keeps this
# working on hosts with no ROCm userspace installed.
detect_amd_gpus() {
    local i=0 d pci vram
    while IFS= read -r d; do
        pci="$(basename "$(readlink -f "$d/device")")"
        vram="$(cat "$d/device/mem_info_vram_total" 2>/dev/null || echo 0)"
        printf '%d\t%s\t%d\n' "$i" "$pci" "$(( vram / 1073741824 ))"
        i=$(( i + 1 ))
    done < <(
        for d in /sys/class/drm/renderD*; do
            [[ -e "$d/device/vendor" ]] || continue
            [[ "$(cat "$d/device/vendor" 2>/dev/null)" == "0x1002" ]] || continue
            printf '%s\t%s\n' "$(basename "$(readlink -f "$d/device")")" "$d"
        done | sort | cut -f2-
    )
}

prompt_gpus() {
    [[ "$GPU_VENDOR" == "rocm" ]] || return 0

    local gpus
    gpus="$(detect_amd_gpus)"
    [[ -n "$gpus" ]] || return 0

    local count
    count="$(printf '%s\n' "$gpus" | wc -l)"
    # One GPU and nothing to choose between.
    [[ "$count" -gt 1 ]] || return 0

    echo ""
    echo -e "${BOLD}GPU selection${NC}"
    echo ""
    echo "  Found ${count} AMD GPUs:"
    echo ""
    local idx pci vram note
    while IFS=$'\t' read -r idx pci vram; do
        note=""
        # An integrated GPU shows up here too and must not be handed to vLLM;
        # its tiny VRAM is the giveaway.
        [[ "$vram" -lt 4 ]] && note="  ← integrated? exclude this one"
        printf "    [%s] %-14s %3s GiB%s\n" "$idx" "$pci" "$vram" "$note"
    done <<< "$gpus"
    echo ""
    echo "  Enter the indices to use, comma-separated (e.g. 0,1), or 'all'."
    echo "  Tensor-parallel size is set per model later, in the web UI."
    echo ""

    local answer
    read -rp "$(echo -e "  ${BOLD}GPUs${NC} [${GPU_DEVICES:-all}]: ")" answer
    answer="${answer:-${GPU_DEVICES:-all}}"

    case "$answer" in
        all|ALL|"")
            GPU_DEVICES=""
            echo "  → Using all ${count} GPUs"
            return 0
            ;;
    esac

    if [[ ! "$answer" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
        warn "Not a valid index list: '$answer' — using all GPUs"
        GPU_DEVICES=""
        return 0
    fi

    # Reject an out-of-range index rather than letting HIP silently see fewer
    # GPUs than the user asked for.
    local n
    for n in ${answer//,/ }; do
        if [[ "$n" -ge "$count" ]]; then
            warn "No GPU with index ${n} (found ${count}) — using all GPUs"
            GPU_DEVICES=""
            return 0
        fi
    done

    GPU_DEVICES="$answer"
    echo "  → Using GPU(s): ${GPU_DEVICES}"
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
        # The variant decides which compose/Dockerfile every later command
        # uses, so up/down/logs/rebuild must read it back, not re-ask.
        val="$(grep '^VLLMCTL_VARIANT=' "$env_file" 2>/dev/null | cut -d= -f2)" || true
        [[ -n "$val" ]] && BUILD_VARIANT="$val" || true
        val="$(grep '^HIP_VISIBLE_DEVICES=' "$env_file" 2>/dev/null | cut -d= -f2)" || true
        [[ -n "$val" ]] && GPU_DEVICES="$val" || true
    fi
}

# ─── Radiance base image ─────────────────────────────────────────────────────
#
# vllm-radiance is published as an OCI manifest whose layers carry *Docker*
# media types. Docker/BuildKit tolerates the mix; containers/image -- the
# library behind podman, buildah AND skopeo alike -- refuses to rewrite such a
# manifest, so `FROM <that image>` dies before the first instruction:
#
#   unsupported MIME type for compression:
#   "application/vnd.docker.image.rootfs.diff.tar.gzip"
#
# Every manifest-level repair hits the same wall (`push --format v2s2`, `save
# --format docker-archive`, skopeo). But podman can RUN the image perfectly
# well, so the way through is to flatten it: export the container filesystem
# and re-import it as a fresh single-layer image, carrying the env and
# entrypoint across by hand.
#
# This is an upstream packaging bug, so it is behind a cheap probe: the day a
# conformant image is published, the probe passes and none of this runs.

radiance_base_ref() {
    if [[ -n "${RADIANCE_IMAGE:-}" ]]; then
        echo "$RADIANCE_IMAGE"; return
    fi
    local val
    val="$(grep '^RADIANCE_IMAGE=' "${SCRIPT_DIR}/.env" 2>/dev/null | cut -d= -f2-)" || true
    echo "${val:-$RADIANCE_DEFAULT_IMAGE}"
}

# Can the build actually use this image as a base? A LABEL-only build is enough
# to find out: it fails while creating the build container, before running
# anything, so the probe costs nothing on a good image.
base_is_buildable() {
    local img="$1" tmpdir rc=1
    tmpdir="$(mktemp -d)"
    printf 'FROM %s\nLABEL vllmctl.probe=1\n' "$img" > "$tmpdir/Dockerfile"
    if $CONTAINER_CMD build -t localhost/vllmctl-baseprobe:tmp "$tmpdir" >/dev/null 2>&1; then
        rc=0
    fi
    $CONTAINER_CMD rmi -f localhost/vllmctl-baseprobe:tmp >/dev/null 2>&1 || true
    rm -rf "$tmpdir"
    return $rc
}

# export | import, reconstructing the metadata import would otherwise drop.
flatten_image() {
    local src="$1" dst="$2"
    local cid rc e ep cm wd
    local -a args=()

    while IFS= read -r e; do
        [[ -n "$e" ]] && args+=(--change "ENV $e")
    done < <($CONTAINER_CMD image inspect "$src" --format '{{range .Config.Env}}{{println .}}{{end}}')

    ep="$($CONTAINER_CMD image inspect "$src" --format '{{json .Config.Entrypoint}}' 2>/dev/null)"
    [[ -n "$ep" && "$ep" != "null" ]] && args+=(--change "ENTRYPOINT $ep")
    cm="$($CONTAINER_CMD image inspect "$src" --format '{{json .Config.Cmd}}' 2>/dev/null)"
    [[ -n "$cm" && "$cm" != "null" ]] && args+=(--change "CMD $cm")
    wd="$($CONTAINER_CMD image inspect "$src" --format '{{.Config.WorkingDir}}' 2>/dev/null)"
    [[ -n "$wd" ]] && args+=(--change "WORKDIR $wd")

    [[ "${#args[@]}" -gt 0 ]] || { err "could not read image config from $src"; return 1; }

    cid="$($CONTAINER_CMD create "$src")" || return 1
    $CONTAINER_CMD export "$cid" | $CONTAINER_CMD import "${args[@]}" - "$dst"
    rc=$?
    $CONTAINER_CMD rm -f "$cid" >/dev/null 2>&1 || true
    return $rc
}

# Make sure the radiance base is present and usable, flattening it if the
# runtime cannot build on top of it. Exports RADIANCE_IMAGE for the build.
ensure_radiance_base() {
    [[ "$BUILD_VARIANT" == "radiance" ]] || return 0

    local src flat tag
    src="$(radiance_base_ref)"

    if ! image_exists "$src"; then
        log "Pulling ${src} (about 4 GB)..."
        $CONTAINER_CMD pull "$src" || fatal "Could not pull ${src}"
    fi

    if base_is_buildable "$src"; then
        export RADIANCE_IMAGE="$src"
        return 0
    fi

    tag="${src##*:}"
    [[ "$tag" == "$src" ]] && tag="latest"
    flat="localhost/vllm-radiance-flat:${tag}"

    if image_exists "$flat"; then
        log "Using previously normalized base image ${flat}"
        export RADIANCE_IMAGE="$flat"
        return 0
    fi

    warn "${src} cannot be used as a build base by ${CONTAINER_CMD}:"
    warn "it is an OCI manifest carrying Docker-typed layers, which"
    warn "containers/image refuses to rewrite. Normalizing it locally."
    echo ""
    echo "  This flattens the image into a single layer, once per radiance"
    echo "  version. It needs roughly 10 GB of free space and a few minutes."
    echo ""

    local avail
    avail="$(df -BG --output=avail "$HOME" 2>/dev/null | tail -1 | tr -dc '0-9')" || avail=""
    if [[ -n "$avail" && "$avail" -lt 15 ]]; then
        warn "Only ${avail} GB free — normalization may run out of space."
    fi

    log "Normalizing ${src} -> ${flat} ..."
    if ! flatten_image "$src" "$flat"; then
        $CONTAINER_CMD rmi -f "$flat" >/dev/null 2>&1 || true
        fatal "Could not normalize ${src}. Building the radiance variant needs
       either a container runtime that accepts this image (Docker does) or
       a conformant image published upstream."
    fi

    if ! base_is_buildable "$flat"; then
        fatal "Normalized image ${flat} is still not usable as a build base."
    fi

    ok "Normalized base image ready: ${flat}"
    export RADIANCE_IMAGE="$flat"
}

# ─── Container operations ────────────────────────────────────────────────────

# The radiance variant has its own image; every other combination is keyed by
# GPU vendor. Keeping GPU_VENDOR as the hardware family (rather than folding
# radiance into it) is what lets the ROCm prerequisite checks, GID detection
# and SELinux handling apply unchanged to both ROCm images.
image_key() {
    if [[ "$BUILD_VARIANT" == "radiance" ]]; then
        echo "radiance"
    else
        echo "$GPU_VENDOR"
    fi
}

compose_file() {
    echo "docker-compose.$(image_key).yml"
}

dockerfile() {
    echo "Dockerfile.$(image_key)"
}

# compose_cmd builds the full compose command with all required -f flags.
compose_cmd() {
    local cmd="$COMPOSE_CMD -f $(compose_file)"
    if [[ -n "${VLLMCTL_MODELS_DIR:-}" ]]; then
        cmd+=" -f docker-compose.models.yml"
    fi
    echo "$cmd"
}

has_quadlet() {
    [[ "$CONTAINER_CMD" == "podman" && $EUID -ne 0 ]] \
        && [[ -f "${QUADLET_USER_DIR}/${PODMAN_SERVICE_NAME}.container" ]]
}

# Write .env file for docker-compose variable substitution
write_env_file() {
    local env_file="${SCRIPT_DIR}/.env"

    # Keys this script owns. Everything else in .env belongs to the user --
    # HF_TOKEN, VLLMCTL_API_KEY, the RADIANCE_* switches -- and truncating the
    # file would silently discard it on every install/rebuild.
    local managed=(
        VLLMCTL_PORT VLLMCTL_INFERENCE_PORT VLLMCTL_VARIANT VLLMCTL_MODELS_DIR
        HSA_OVERRIDE_GFX_VERSION GPU_ARCH HOST_VIDEO_GID HOST_RENDER_GID
        HIP_VISIBLE_DEVICES
    )

    local preserved=""
    if [[ -f "$env_file" ]]; then
        local pattern
        pattern="^($(IFS='|'; echo "${managed[*]}"))="
        preserved="$(grep -Ev "$pattern" "$env_file" 2>/dev/null || true)"
    fi

    {
        echo "VLLMCTL_PORT=${VLLMCTL_PORT}"
        echo "VLLMCTL_INFERENCE_PORT=${VLLMCTL_INFERENCE_PORT}"
        echo "VLLMCTL_VARIANT=${BUILD_VARIANT}"

        [[ -n "$VLLMCTL_MODELS_DIR" ]] && echo "VLLMCTL_MODELS_DIR=${VLLMCTL_MODELS_DIR}"
        [[ -n "$AMD_GFX_VERSION" ]]    && echo "HSA_OVERRIDE_GFX_VERSION=${AMD_GFX_VERSION}"
        [[ -n "$AMD_GFX_TARGET" ]]     && echo "GPU_ARCH=${AMD_GFX_TARGET}"
        [[ -n "$HOST_VIDEO_GID" ]]     && echo "HOST_VIDEO_GID=${HOST_VIDEO_GID}"
        [[ -n "$HOST_RENDER_GID" ]]    && echo "HOST_RENDER_GID=${HOST_RENDER_GID}"
        # Only written when a subset was chosen. An empty HIP_VISIBLE_DEVICES
        # is not "all GPUs" -- HIP reads it as "no GPUs" -- so the variable
        # must be absent rather than blank.
        [[ -n "$GPU_DEVICES" ]]        && echo "HIP_VISIBLE_DEVICES=${GPU_DEVICES}"

        # User-owned lines last, so they are visibly theirs to edit.
        [[ -n "$preserved" ]] && printf '%s\n' "$preserved"
    } > "$env_file"

    [[ -n "$VLLMCTL_MODELS_DIR" ]] && export VLLMCTL_MODELS_DIR
    return 0
}

container_up() {
    if has_quadlet; then
        log "Starting vllm-toolchest via systemd (Quadlet)..."
        systemctl_cmd start "${PODMAN_SERVICE_NAME}.service"
    else
        $(compose_cmd) up -d
    fi
}

container_down() {
    if has_quadlet; then
        log "Stopping vllm-toolchest via systemd (Quadlet)..."
        systemctl_cmd stop "${PODMAN_SERVICE_NAME}.service"
    else
        $(compose_cmd) down
    fi
}

container_install() {
    ensure_radiance_base
    write_env_file

    # Remove any existing container before bringing one up. `up -d` alone does
    # NOT apply a changed environment under podman-compose -- it sees the
    # container already exists and simply starts it -- so re-running install
    # after changing the GPU selection, the ports or the models directory
    # silently kept the old values, and the only clue was the running service
    # still reporting the previous configuration.
    local quadlet_active=false
    has_quadlet && quadlet_active=true
    if container_exists vllm-toolchest; then
        log "Removing the existing container so the new configuration applies..."
        container_down
        $CONTAINER_CMD rm -f vllm-toolchest 2>/dev/null || true
    fi

    BUILDKIT_PROGRESS=plain $(compose_cmd) up -d --build

    if [[ "$quadlet_active" == true ]]; then
        log "Restarting via systemd (Quadlet)..."
        $(compose_cmd) down >/dev/null 2>&1 || true
        systemctl_cmd start "${PODMAN_SERVICE_NAME}.service"
    fi
}

container_rebuild() {
    ensure_radiance_base
    local quadlet_active=false
    has_quadlet && quadlet_active=true

    container_down
    $CONTAINER_CMD rm vllm-toolchest 2>/dev/null || true
    write_env_file
    BUILDKIT_PROGRESS=plain $(compose_cmd) build --no-cache --progress=plain

    if [[ "$quadlet_active" == true ]]; then
        log "Starting via systemd (Quadlet)..."
        systemctl_cmd start "${PODMAN_SERVICE_NAME}.service"
    else
        $(compose_cmd) up -d
    fi
}

# Quick rebuild: only rebuild layers that changed (Go code), reuse cached base layers.
container_quick_rebuild() {
    ensure_radiance_base
    container_down
    write_env_file
    BUILDKIT_PROGRESS=plain $(compose_cmd) up -d --build
}

container_logs() {
    if has_quadlet; then
        journalctl --user -u "${PODMAN_SERVICE_NAME}.service" -n 100 -f
    else
        $(compose_cmd) logs -f
    fi
}

# ─── Auto-start (enable/disable) ─────────────────────────────────────────────

quadlet_dir() {
    if [[ $EUID -eq 0 ]]; then
        echo "$QUADLET_SYSTEM_DIR"
    else
        echo "$QUADLET_USER_DIR"
    fi
}

systemctl_cmd() {
    if [[ $EUID -eq 0 ]]; then
        systemctl "$@"
    else
        systemctl --user "$@"
    fi
}

get_restart_policy() {
    $CONTAINER_CMD inspect --format '{{.HostConfig.RestartPolicy.Name}}' vllm-toolchest 2>/dev/null || echo ""
}

is_autostart_enabled() {
    if [[ "$CONTAINER_CMD" == "docker" ]]; then
        local policy
        policy="$(get_restart_policy)"
        [[ "$policy" == "always" || "$policy" == "unless-stopped" ]]
    else
        if [[ $EUID -eq 0 ]]; then
            local policy
            policy="$(get_restart_policy)"
            [[ "$policy" == "always" || "$policy" == "unless-stopped" ]]
        else
            [[ -f "$(quadlet_dir)/${PODMAN_SERVICE_NAME}.container" ]]
        fi
    fi
}

get_volume_name() {
    $CONTAINER_CMD inspect --format '{{range .Mounts}}{{.Name}}{{end}}' vllm-toolchest 2>/dev/null \
        || echo "vllmctl-data"
}

generate_quadlet() {
    local image_name="localhost/vllm-toolchest:latest"
    local volume_name
    volume_name="$(get_volume_name)"
    local gpu_args=""

    if [[ "$GPU_VENDOR" == "cuda" ]]; then
        gpu_args="AddDevice=nvidia.com/gpu=all"
    elif [[ "$GPU_VENDOR" == "rocm" ]]; then
        local hsa_env=""
        if [[ -n "$AMD_GFX_VERSION" ]]; then
            hsa_env="Environment=HSA_OVERRIDE_GFX_VERSION=${AMD_GFX_VERSION}"
        fi
        local extra_caps=""
        if [[ "$BUILD_VARIANT" == "radiance" ]]; then
            # py-spy profiling and the optional RADIANCE_NUMA_BIND mempolicy
            # syscalls; both no-ops unless used.
            extra_caps="AddCapability=SYS_PTRACE
AddCapability=SYS_NICE"
        fi
        gpu_args="AddDevice=/dev/kfd
AddDevice=/dev/dri
SecurityLabelDisable=true
PodmanArgs=--ipc=host
ShmSize=${VLLMCTL_SHM_SIZE:-8gb}
GroupAdd=${HOST_VIDEO_GID:-video}
GroupAdd=${HOST_RENDER_GID:-render}
${extra_caps}
${hsa_env}"
    fi

    cat <<EOF
# Auto-generated by vllm-toolchest setup.sh
# GPU backend: ${GPU_VENDOR}
# Image variant: ${BUILD_VARIANT}
# Runtime: ${CONTAINER_CMD}

[Unit]
Description=vllm-toolchest - local vLLM management
After=network-online.target

[Container]
Image=${image_name}
ContainerName=vllm-toolchest
PublishPort=${VLLMCTL_PORT}:3000
PublishPort=${VLLMCTL_INFERENCE_PORT}:8000
Volume=${volume_name}:/data:z
${gpu_args}

[Service]
Restart=on-failure
TimeoutStartSec=900

[Install]
WantedBy=default.target
EOF
}

autostart_enable() {
    if is_autostart_enabled; then
        ok "Auto-start is already enabled"
        return
    fi

    if [[ "$CONTAINER_CMD" == "docker" ]]; then
        if ! docker inspect vllm-toolchest &>/dev/null; then
            fatal "Container 'vllm-toolchest' not found. Run './setup.sh install' first."
        fi
        log "Setting restart policy to 'unless-stopped'..."
        docker update --restart unless-stopped vllm-toolchest
        ok "Auto-start enabled"
    elif [[ $EUID -eq 0 ]]; then
        if ! podman container exists vllm-toolchest 2>/dev/null; then
            fatal "Container 'vllm-toolchest' not found. Run './setup.sh install' first."
        fi
        log "Setting restart policy to 'unless-stopped'..."
        podman update --restart unless-stopped vllm-toolchest
        ok "Auto-start enabled"
    else
        # Podman rootless: need Quadlet + linger to survive reboot
        local qdir
        qdir="$(quadlet_dir)"

        # If the container is already running outside of systemd, stop it first
        # so the Quadlet service can take over without a name conflict.
        local was_running=false
        if podman container exists vllm-toolchest 2>/dev/null; then
            was_running=true
            log "Stopping existing container so systemd can take over..."
            podman stop vllm-toolchest 2>/dev/null || true
            podman rm vllm-toolchest 2>/dev/null || true
        fi

        log "Installing Quadlet unit: ${qdir}/${PODMAN_SERVICE_NAME}.container"

        mkdir -p "$qdir"
        generate_quadlet > "${qdir}/${PODMAN_SERVICE_NAME}.container"

        systemctl_cmd daemon-reload

        # Enable lingering so user services run without an active login session
        local linger_status
        linger_status="$(loginctl show-user "$USER" --property=Linger 2>/dev/null || true)"
        if [[ "$linger_status" != *"yes"* ]]; then
            log "Enabling loginctl linger for user $USER..."
            if ! loginctl enable-linger "$USER" 2>/dev/null; then
                run_sudo loginctl enable-linger "$USER"
            fi
        fi

        if [[ "$was_running" == true ]]; then
            log "Starting vllm-toolchest via systemd..."
            systemctl_cmd start "${PODMAN_SERVICE_NAME}.service"
        fi

        # Quadlet units are auto-activated by systemd's generator via
        # WantedBy= in the [Install] section — no explicit enable needed.
        ok "Auto-start enabled via Podman Quadlet"

        echo ""
        echo "  vllm-toolchest will auto-start on boot."
        echo "  Manage with: systemctl --user {start,stop,status} ${PODMAN_SERVICE_NAME}"
    fi
}

autostart_disable() {
    if ! is_autostart_enabled; then
        ok "Auto-start is already disabled"
        return
    fi

    if [[ "$CONTAINER_CMD" == "docker" ]]; then
        log "Setting restart policy to 'no'..."
        docker update --restart no vllm-toolchest
        ok "Auto-start disabled"
    elif [[ $EUID -eq 0 ]]; then
        log "Setting restart policy to 'no'..."
        podman update --restart no vllm-toolchest
        ok "Auto-start disabled"
    else
        local qdir
        qdir="$(quadlet_dir)"

        log "Removing Quadlet unit: ${qdir}/${PODMAN_SERVICE_NAME}.container"

        systemctl_cmd stop "${PODMAN_SERVICE_NAME}.service" 2>/dev/null || true
        rm -f "${qdir}/${PODMAN_SERVICE_NAME}.container"
        systemctl_cmd daemon-reload
        ok "Auto-start disabled"
    fi
}

# ─── Uninstall ───────────────────────────────────────────────────────────────

container_uninstall() {
    local actions=()
    local has_autostart=false
    local has_container=false
    local has_image=false

    # Compose tags the build "vllm-toolchest:latest"; podman stores that under
    # an implicit localhost/ prefix, Docker does not. Check both.
    local image_name="" candidate
    for candidate in "localhost/vllm-toolchest:latest" "vllm-toolchest:latest"; do
        if image_exists "$candidate"; then
            image_name="$candidate"
            break
        fi
    done

    if is_autostart_enabled; then
        has_autostart=true
        actions+=("Disable auto-start on boot")
    fi

    if container_exists vllm-toolchest; then
        has_container=true
        actions+=("Stop and remove container 'vllm-toolchest'")
    fi

    if [[ -n "$image_name" ]]; then
        has_image=true
        actions+=("Remove image '${image_name}'")
    fi

    if [[ ${#actions[@]} -eq 0 ]]; then
        ok "Nothing to uninstall — vllm-toolchest is not installed"
        return
    fi

    echo ""
    echo -e "${BOLD}The following will be removed:${NC}"
    echo ""
    local i=1
    for action in "${actions[@]}"; do
        echo -e "  ${i}. ${action}"
        ((i++))
    done
    echo ""

    local volume_name
    volume_name="$($CONTAINER_CMD inspect --format '{{range .Mounts}}{{.Name}}{{end}}' vllm-toolchest 2>/dev/null || echo "vllmctl-data")"
    echo -e "  ${YELLOW}Note:${NC} The data volume (models, config) will be kept."
    echo -e "        To remove it: ${CONTAINER_CMD} volume rm ${volume_name}"
    echo ""

    if ! prompt_confirm "Proceed with uninstall?"; then
        echo "Aborted."
        exit 0
    fi

    echo ""

    if [[ "$has_autostart" == true ]]; then
        autostart_disable
    fi

    if [[ "$has_container" == true ]]; then
        log "Stopping and removing container..."
        $CONTAINER_CMD stop vllm-toolchest 2>/dev/null || true
        $CONTAINER_CMD rm vllm-toolchest 2>/dev/null || true
        ok "Container removed"
    fi

    if [[ "$has_image" == true ]]; then
        log "Removing image..."
        $CONTAINER_CMD rmi "$image_name" 2>/dev/null || true
        ok "Image removed"
    fi

    echo ""
    ok "vllm-toolchest uninstalled"
}

# ─── Summary ─────────────────────────────────────────────────────────────────

print_summary() {
    local cf df
    cf="$(compose_file)"
    df="$(dockerfile)"

    echo ""
    echo -e "${BOLD}════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}  vllm-toolchest setup${NC}"
    echo -e "${BOLD}════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "  ${CYAN}GPU${NC}           ${GPU_INFO}"
    echo -e "  ${CYAN}Backend${NC}       ${GPU_VENDOR}"
    if [[ "$BUILD_VARIANT" == "radiance" ]]; then
        echo -e "  ${CYAN}Variant${NC}       ${BUILD_VARIANT} (vllm-radiance, gfx1201-tuned)"
    else
        echo -e "  ${CYAN}Variant${NC}       ${BUILD_VARIANT}"
    fi
    echo -e "  ${CYAN}Runtime${NC}       ${CONTAINER_VERSION}"
    echo -e "  ${CYAN}Compose${NC}       ${COMPOSE_VERSION}"
    echo -e "  ${CYAN}Distro${NC}        ${DISTRO_NAME}"
    echo -e "  ${CYAN}Dockerfile${NC}    ${df}"
    echo -e "  ${CYAN}Compose file${NC}  ${cf}"
    echo -e "  ${CYAN}UI port${NC}       ${VLLMCTL_PORT}"
    echo -e "  ${CYAN}Inference port${NC} ${VLLMCTL_INFERENCE_PORT}"
    if [[ -n "$VLLMCTL_MODELS_DIR" ]]; then
        echo -e "  ${CYAN}Models dir${NC}    ${VLLMCTL_MODELS_DIR}"
    fi
    if [[ -n "$AMD_GFX_TARGET" ]]; then
        echo -e "  ${CYAN}GPU arch${NC}      ${AMD_GFX_TARGET}"
    fi
    if [[ "$GPU_VENDOR" == "rocm" ]]; then
        local gpu_count
        gpu_count="$(detect_amd_gpus | wc -l)"
        if [[ -n "$GPU_DEVICES" ]]; then
            echo -e "  ${CYAN}GPUs in use${NC}   ${GPU_DEVICES} (of ${gpu_count} detected)"
        elif [[ "$gpu_count" -gt 0 ]]; then
            echo -e "  ${CYAN}GPUs in use${NC}   all ${gpu_count}"
        fi
    fi
    if [[ -n "$AMD_GFX_VERSION" ]]; then
        echo -e "  ${CYAN}HSA Override${NC}  ${AMD_GFX_VERSION}"
    fi

    local autostart_status="disabled"
    if is_autostart_enabled; then
        autostart_status="enabled"
    fi
    echo -e "  ${CYAN}Auto-start${NC}    ${autostart_status}"
    echo ""

    if [[ ! -f "${SCRIPT_DIR}/${cf}" ]]; then
        err "Compose file ${cf} not found!"
        echo "  Available compose files:"
        ls -1 "${SCRIPT_DIR}"/docker-compose.*.yml 2>/dev/null | sed 's|.*/|    |' || echo "    (none)"
        echo ""
        fatal "Cannot proceed without compose file"
    fi

    if [[ ${#ACTIONS[@]} -gt 0 ]]; then
        echo -e "  ${BOLD}Actions:${NC}"
        local i=1
        for action in "${ACTIONS[@]}"; do
            echo -e "    ${i}. ${action}"
            ((i++))
        done
        echo ""
    fi

    if [[ ${#PREREQS[@]} -gt 0 ]]; then
        echo -e "  ${YELLOW}Note:${NC} Prerequisite steps require sudo"
        echo ""
    fi
}

# ─── Main ────────────────────────────────────────────────────────────────────

usage() {
    cat <<'USAGE'
vllm-toolchest setup — auto-detect GPU + container runtime, build & run

Usage: ./setup.sh <command>

Lifecycle:
  install     Detect GPU/runtime/distro, install prerequisites, build image,
              and start the container
  uninstall   Stop container, disable auto-start, and remove container + image
  quick       Fast rebuild — only recompile Go code, reuse cached base layers
  rebuild     Full rebuild with no cache, then start

Runtime:
  up          Start a stopped container. Does NOT re-read .env -- a container
              keeps the environment it was created with, so after changing GPU
              selection or ports run `install` (or `down` then `up`) to have
              the container recreated
  down        Stop the container
  logs        Follow container logs (Ctrl-C to stop)

Auto-start:
  enable      Enable auto-start on boot
                Docker:       sets restart policy to 'unless-stopped'
                Podman root:  sets restart policy to 'unless-stopped'
                Podman user:  installs a Quadlet systemd unit and enables
                              loginctl linger so the service survives logout
  disable     Disable auto-start on boot
                Docker:       sets restart policy to 'no'
                Podman root:  sets restart policy to 'no'
                Podman user:  removes the Quadlet systemd unit

Info:
  status      Show detected environment and planned actions, then exit
  detect      Print detected GPU backend (cuda/rocm) and image variant, exit
  help        Show this help message

Image variants (AMD only):
  generic     Builds ROCm + vLLM from source, tracking vLLM main. Portable
              across GPU generations, newest model support, long build.
  radiance    Layers this UI on the third-party vllm-radiance image, a stack
              hand-tuned for RDNA4 / gfx1201 (custom attention, GEMM and
              all-reduce kernels, tuned FP8 + MoE configs, MTP drafting).
              Fast to install, but vLLM and transformers are pinned, so model
              support is frozen at that release.
              https://codeberg.org/StillDeadcode/vllm-radiance

  `install` offers the choice on an RDNA4 card; the answer is stored in .env
  and reused by every later command. Force it with VLLMCTL_VARIANT= any time.

Environment variables:
  GPU=cuda|rocm                        Override GPU auto-detection
  VLLMCTL_VARIANT=generic|radiance     Override image variant (skips the prompt)
  RUNTIME=docker|podman                Override container runtime auto-detection
  RADIANCE_IMAGE=<ref>                 Base image for the radiance variant

  VARIANT= is accepted as a short form, but VLLMCTL_VARIANT is preferred:
  VARIANT is also an /etc/os-release field (Ubuntu Server sets it to
  "Server Edition"), so the short form can collide on some distros.

Port configuration is stored in .env (see .env.example for details).
You can edit .env directly instead of using the interactive setup.

Examples:
  ./setup.sh install              # detect everything, install prereqs, build & run
  ./setup.sh status               # dry run — show what would happen
  ./setup.sh enable               # start on boot
  ./setup.sh disable              # stop starting on boot
  ./setup.sh uninstall            # remove everything
  ./setup.sh quick                # fast rebuild (code changes only)
  ./setup.sh rebuild              # full clean rebuild (no cache)
  RUNTIME=podman ./setup.sh install  # force Podman runtime
  VLLMCTL_VARIANT=radiance ./setup.sh install   # build the RDNA4-tuned image
  VLLMCTL_VARIANT=generic ./setup.sh rebuild    # switch back to the portable image
USAGE
}

main() {
    local command="${1:-help}"
    cd "$SCRIPT_DIR"

    case "$command" in
        install|uninstall|up|down|rebuild|quick|logs|detect|status|enable|disable) ;;
        -h|--help|help) usage; exit 0 ;;
        *)
            err "Unknown command: $command"
            echo ""
            usage
            exit 1
            ;;
    esac

    if [[ -n "${GPU:-}" ]]; then
        GPU_VENDOR="$GPU"
        GPU_INFO="(manually set: $GPU)"
        # Forcing the vendor should skip vendor *detection*, not the AMD probes
        # that feed the build: without these, GPU_ARCH, the HSA override and the
        # video/render GIDs are all silently empty.
        if [[ "$GPU_VENDOR" == "rocm" ]]; then
            detect_amd_gfx_version
            detect_host_gpu_gids
        fi
    else
        detect_gpu
    fi

    if [[ "$command" == "detect" ]]; then
        load_env_ports
        detect_variant
        echo "$GPU_VENDOR $BUILD_VARIANT"
        exit 0
    fi

    detect_container_runtime
    detect_distro
    # load_env_ports also restores BUILD_VARIANT from a previous install, so it
    # has to run before detect_variant picks a default.
    load_env_ports
    detect_variant

    case "$command" in
        up)        container_up;   ok "vllm-toolchest started"; exit 0 ;;
        down)      container_down; ok "vllm-toolchest stopped"; exit 0 ;;
        logs)      container_logs; exit 0 ;;
        enable)    autostart_enable;  exit 0 ;;
        disable)   autostart_disable; exit 0 ;;
        uninstall) container_uninstall; exit 0 ;;
        quick)
            log "Quick rebuild (cached)..."
            container_quick_rebuild
            ok "vllm-toolchest is running"
            echo ""
            echo "  Web UI:     http://localhost:${VLLMCTL_PORT}"
            echo ""
            exit 0
            ;;
    esac

    # install, rebuild, status
    # Ask which image to build first: the variant decides which Dockerfile the
    # summary reports and which prerequisites are checked, so choosing after
    # printing the summary would show the user the wrong plan. `status` is a
    # dry run and never prompts.
    if [[ "$command" != "status" ]]; then
        prompt_variant
    fi

    check_prerequisites
    print_summary

    if [[ "$command" == "status" ]]; then
        exit 0
    fi

    prompt_gpus
    prompt_ports
    prompt_models_dir

    if [[ ${#PREREQS[@]} -gt 0 ]]; then
        if ! prompt_confirm "Install prerequisites?"; then
            echo "Aborted."
            exit 0
        fi
        echo ""
        install_prerequisites
        echo ""
    fi

    if ! prompt_confirm "Build and start vllm-toolchest?"; then
        echo "Aborted."
        exit 0
    fi

    echo ""
    case "$command" in
        install)  container_install ;;
        rebuild)  container_rebuild ;;
    esac

    echo ""
    ok "vllm-toolchest is running"
    echo ""
    echo "  Web UI:     http://localhost:${VLLMCTL_PORT}"
    echo "  Inference:  http://localhost:${VLLMCTL_INFERENCE_PORT}"
    echo ""
    echo "  Logs:       ./setup.sh logs"
    echo "  Stop:       ./setup.sh down"
    echo "  Auto-start: ./setup.sh enable"
    echo ""
}

main "$@"

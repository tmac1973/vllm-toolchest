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

# ─── Global state (populated by detect_* functions) ──────────────────────────

GPU_VENDOR=""           # cuda, rocm
GPU_INFO=""             # human-readable GPU description
BUILD_VARIANT=""        # variant id from variants/*.conf — which image to build
GPU_MODEL=""            # PCI device id or compute capability, for variant matching
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

# ─── Variant manifests ───────────────────────────────────────────────────────
#
# variants/<id>.conf describes one image this tool can be built on: its base
# image, the hardware it runs on, and the feature switches ("knobs") its
# Settings panel offers. Each fact is declared exactly once, there, and both
# this script and the Go binary generate from it -- the binary go:embeds the
# same files and parses the same grammar (see variants/variants.go).
#
# The grammar is a strict subset of bash on purpose, so this script can source
# a manifest with no jq, yq or python on the host: NAME='value', one per line,
# always single-quoted. That literalness is what lets a tooltip carry commas,
# double quotes and em-dashes unescaped; the one character it cannot carry is
# the ASCII apostrophe.
#
# Every key is VARIANT_-, GROUP_- or KNOB_-prefixed because these files land in
# this script's own shell: a bare ID= or VENDOR= would clobber a global.

readonly VARIANTS_DIR="${SCRIPT_DIR}/variants"

# list_variants prints every declared variant id, one per line.
#
# Sorted byte-wise, not by locale. A glob expands in collation order, which in
# most locales ignores punctuation and puts "rocm-cdna" before "rocm" -- while
# Go sorts by byte value and puts "rocm" first. The two readers disagreeing
# about order is the kind of thing that goes unnoticed until a menu numbers its
# entries differently from the list somebody read.
list_variants() {
    local f base
    for f in "${VARIANTS_DIR}"/*.conf; do
        [[ -e "$f" ]] || continue
        base="${f##*/}"
        printf '%s\n' "${base%.conf}"
    done | LC_ALL=C sort
}

# load_variant_manifest sources a manifest into the current shell.
#
# Sourcing a second manifest leaves the first one's KNOB_* variables behind,
# because a manifest only assigns the keys it declares. Anything that walks
# every variant must therefore do it in a subshell -- see variant_field.
load_variant_manifest() {
    local f="${VARIANTS_DIR}/$1.conf"
    [[ -r "$f" ]] || return 1
    # shellcheck source=/dev/null
    . "$f"
}

# knob_attr SLUG ATTR prints one knob attribute, empty when undeclared.
# The ${!var-} form matters under `set -u`: an absent attribute has to read as
# empty, not abort the script.
knob_attr() {
    local var="KNOB_$1_$2"
    printf '%s' "${!var-}"
}

# variant_field ID NAME prints one VARIANT_* field from a manifest without
# leaving that manifest loaded, so callers can query several in a loop.
variant_field() {
    ( load_variant_manifest "$1" || exit 1
      local var="VARIANT_$2"
      printf '%s' "${!var-}" )
}

# knob_env_names prints the environment variables the loaded manifest's knobs
# own, in declaration order. $KNOBS is deliberately unquoted: word splitting is
# how the list is iterated.
knob_env_names() {
    local slug
    # shellcheck disable=SC2086
    for slug in ${KNOBS:-}; do
        printf '%s\n' "$(knob_attr "$slug" ENV)"
    done
}

# knob_recommended_env prints the KEY=VALUE lines an "apply recommended
# settings" action would write. A recommendation of "-" means the manifest's
# advice is to leave the image's own default alone, so it emits nothing.
knob_recommended_env() {
    local slug rec env
    # shellcheck disable=SC2086
    for slug in ${KNOBS:-}; do
        rec="$(knob_attr "$slug" RECOMMENDED)"
        [[ -n "$rec" && "$rec" != "-" ]] || continue
        env="$(knob_attr "$slug" ENV)"
        printf '%s=%s\n' "$env" "$rec"
    done
}

# ─── Generated knob documentation ────────────────────────────────────────────
#
# The feature switches used to be written out by hand in .env.example, in the
# Settings template, in the Go config layer and in three more places. They are
# declared once in variants/<id>.conf now, so the documentation is generated
# from the same declaration rather than kept in step with it.

# knob_example_value SLUG — a value worth showing in an example line.
# For a picker, the first real option; for free text, the placeholder or the
# recommendation. Never the unset sentinel: an example that does nothing
# teaches nothing.
knob_example_value() {
    local slug="$1" rec vals v
    rec="$(knob_attr "$slug" RECOMMENDED)"
    if [[ -n "$rec" && "$rec" != "-" ]]; then
        printf '%s' "$rec"; return
    fi
    vals="$(knob_attr "$slug" VALUES)"
    if [[ -n "$vals" ]]; then
        # shellcheck disable=SC2086
        for v in $vals; do
            [[ "$v" == "-" ]] && continue
            printf '%s' "$v"; return
        done
    fi
    # A placeholder is grey hint text in the UI, which is sometimes a real
    # value ("1:8,2:7,4:6") and sometimes prose ("off — try: auto"). Only the
    # former belongs after an "=". Whitespace is the tell, and getting it wrong
    # means shipping a .env line that sets a knob to a sentence.
    v="$(knob_attr "$slug" PLACEHOLDER)"
    if [[ -n "$v" && "$v" != *[[:space:]]* ]]; then
        printf '%s' "$v"; return
    fi
    printf ''
}

# print_env_knobs writes the commented KEY=VALUE documentation for every
# variant that declares switches, in .env format.
print_env_knobs() {
    local id first=1 slug label help example group last_group
    for id in $(list_variants); do
        ( load_variant_manifest "$id" || exit 0
          [[ -n "${KNOBS:-}" ]] || exit 0

          [[ $first -eq 1 ]] || echo ""
          echo "# ── ${VARIANT_LABEL} — VLLMCTL_VARIANT=${VARIANT_ID} ──"
          [[ -n "${VARIANT_DOC_URL:-}" ]] && echo "# ${VARIANT_DOC_URL}"

          last_group=""
          # shellcheck disable=SC2086
          for slug in $KNOBS; do
              group="$(knob_attr "$slug" GROUP)"
              if [[ "$group" != "$last_group" ]]; then
                  local gt gn
                  gt="$(eval "printf '%s' \"\${GROUP_${group}_TITLE-}\"")"
                  gn="$(eval "printf '%s' \"\${GROUP_${group}_NOTE-}\"")"
                  if [[ -n "$gt" || -n "$gn" ]]; then
                      echo "#"
                      [[ -n "$gt" ]] && echo "# ${gt}."
                      [[ -n "$gn" ]] && fold -s -w 70 <<<"$gn" | sed 's/^/# /; s/[[:space:]]*$//'
                  fi
                  last_group="$group"
              fi

              label="$(knob_attr "$slug" LABEL)"
              help="$(knob_attr "$slug" HELP)"
              example="$(knob_example_value "$slug")"

              echo "#"
              fold -s -w 70 <<<"${label}. ${help}" | sed 's/^/# /; s/[[:space:]]*$//'
              echo "#$(knob_attr "$slug" ENV)=${example}"
          done )
        first=0
    done
}

# write_env_example rewrites the generated block of a .env template in place,
# leaving everything outside the markers alone. Splitting the file rather than
# generating all of it keeps the hand-written parts -- ports, GPU selection,
# storage -- hand-written, because those are prose about choices rather than a
# list derived from the manifests.
write_env_example() {
    local target="$1" tmp
    [[ -f "$target" ]] || fatal "no such file: $target"
    tmp="$(mktemp)"

    local in_block=0 line
    while IFS= read -r line || [[ -n "$line" ]]; do
        case "$line" in
            "# >>> BEGIN GENERATED KNOBS")
                printf '%s\n' "$line" >> "$tmp"
                print_env_knobs >> "$tmp"
                in_block=1
                continue
                ;;
            "# <<< END GENERATED KNOBS")
                in_block=0
                ;;
        esac
        [[ $in_block -eq 1 ]] && continue
        printf '%s\n' "$line" >> "$tmp"
    done < "$target"

    if ! grep -q '^# <<< END GENERATED KNOBS$' "$tmp"; then
        rm -f "$tmp"
        fatal "$target has no generated-knob markers to fill"
    fi
    mv "$tmp" "$target"
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

# Intel splits by generation and the split matters: Battlemage (B60/B70) has a
# working vLLM path, Alchemist (A770) does not -- it needs a different image
# lineage entirely. Enumerating the device is not enough; the device ID is.
detect_intel_gpu() {
    need_cmd lspci || return 1
    local line id
    while IFS= read -r line; do
        # "VGA compatible controller [0300]: Intel ... [8086:e20b]"
        id="$(sed -n 's/.*\[8086:\([0-9a-f]\{4\}\)\].*/\1/p' <<<"$line")"
        [[ -n "$id" ]] || continue
        GPU_VENDOR="xpu"
        GPU_MODEL="$id"
        GPU_INFO="$(sed 's/.*Intel/Intel/; s/ \[8086:.*//' <<<"$line")"
        return 0
    done < <(lspci -nn 2>/dev/null | grep -iE 'VGA|Display|3D controller' | grep -i '8086:')
    return 1
}

detect_gpu() {
    # NVIDIA: check for nvidia-smi AND that it can talk to a GPU
    if need_cmd nvidia-smi; then
        if nvidia-smi --query-gpu=name --format=csv,noheader &>/dev/null; then
            GPU_VENDOR="cuda"
            GPU_INFO="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || true)"
            GPU_INFO="${GPU_INFO%%$'\n'*}"
            # Compute capability is what tells sm_121a (DGX Spark) apart from
            # every other NVIDIA card. Paired with the machine architecture,
            # because that variant is arm64 and nothing else is.
            GPU_MODEL="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)" || true
            GPU_MODEL="${GPU_MODEL//./}"
            [[ -n "$GPU_MODEL" ]] && GPU_MODEL="sm_${GPU_MODEL}"
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
            name="$(rocminfo 2>/dev/null | grep 'Marketing Name' | sed 's/.*: *//; s/[[:space:]]*$//' \
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

    # Intel comes last: an Intel iGPU alongside a discrete AMD or NVIDIA card
    # is common, and the discrete card is what someone wants to serve on.
    if detect_intel_gpu; then
        return
    fi

    # No GPU — vLLM has no meaningful CPU inference path, so fall back to CUDA
    # (the user probably just doesn't have drivers installed yet). Said out
    # loud in the menu rather than silently picking a vendor.
    GPU_VENDOR="cuda"
    GPU_INFO="No GPU detected (defaulting to CUDA)"
}

# ─── Variant matching ────────────────────────────────────────────────────────
#
# A variant matches this host when its vendor matches, its architecture list is
# empty or contains what was found, and its host architecture matches. Among
# the matches, the most specific wins: a variant naming one gfx target beats
# one that runs on anything, because it was built for the card in the machine.

# The manifests name vendors (amd/nvidia/intel); GPU_VENDOR names the compute
# stack (rocm/cuda/xpu). One mapping, in one place.
detected_vendor() {
    case "$GPU_VENDOR" in
        cuda) echo "nvidia" ;;
        rocm) echo "amd" ;;
        xpu)  echo "intel" ;;
        *)    echo "$GPU_VENDOR" ;;
    esac
}

# variant_matches ID — true when this variant can run on what was detected.
variant_matches() {
    local id="$1"
    ( load_variant_manifest "$id" || exit 1

      [[ "$VARIANT_VENDOR" == "$(detected_vendor)" ]] || exit 1

      # An empty target list means the variant is not tied to one architecture.
      if [[ -n "${VARIANT_GFX_TARGETS:-}" ]]; then
          local want found=1 probe="${AMD_GFX_TARGET:-$GPU_MODEL}"
          for want in $VARIANT_GFX_TARGETS; do
              [[ "$probe" == "$want" ]] && found=0
          done
          [[ $found -eq 0 ]] || exit 1
      fi

      if [[ -n "${VARIANT_HOST_ARCH:-}" ]]; then
          [[ "$VARIANT_HOST_ARCH" == "$(uname -m)" ]] || exit 1
      fi
      exit 0 )
}

# variant_specificity ID — higher sorts first in the menu. A named architecture
# beats a general one; ties break on the manifest's declared priority.
variant_specificity() {
    ( load_variant_manifest "$1" || exit 0
      local score=0
      [[ -n "${VARIANT_GFX_TARGETS:-}" ]] && score=$((score + 1000))
      printf '%s' "$((score + ${VARIANT_PRIORITY:-0}))" )
}

# matching_variants prints every variant that runs here, best first.
matching_variants() {
    local id
    for id in $(list_variants); do
        variant_matches "$id" || continue
        printf '%s %s\n' "$(variant_specificity "$id")" "$id"
    done | sort -rn -k1,1 | awk '{print $2}'
}

# recommended_variant prints the single best match, or nothing.
recommended_variant() {
    matching_variants | head -1
}

# variant_exists ID — does a manifest by this name ship?
variant_exists() {
    [[ -r "${VARIANTS_DIR}/$1.conf" ]]
}

# Choose the image variant. An explicit VLLMCTL_VARIANT always wins; otherwise
# a variant recorded by a previous install stands, and failing that the best
# match for the detected hardware.
detect_variant() {
    # VLLMCTL_VARIANT is the documented override, matching the key stored in
    # .env. Bare VARIANT is accepted as a convenience, but only when it names a
    # real variant: VARIANT is also an /etc/os-release field (Ubuntu Server
    # sets it to "Server Edition"), so a value we do not recognise belongs to
    # somebody else and must not be a fatal error.
    local want="${VLLMCTL_VARIANT:-}"
    if [[ -z "$want" ]] && [[ -n "${VARIANT:-}" ]] && variant_exists "${VARIANT}"; then
        want="$VARIANT"
    fi

    if [[ -n "$want" ]]; then
        variant_exists "$want" || fatal "Unknown VLLMCTL_VARIANT=$want. Known variants: $(list_variants | tr '\n' ' ')"
        BUILD_VARIANT="$want"
        return
    fi

    # Already chosen in a previous run -- .env is the record of that decision.
    [[ -n "$BUILD_VARIANT" ]] && return

    BUILD_VARIANT="$(recommended_variant)"

    # Nothing matched. That means hardware we have no manifest for, or a GPU
    # we failed to detect; either way the install cannot proceed on a guess,
    # because the wrong base image fails minutes into a build.
    if [[ -z "$BUILD_VARIANT" ]]; then
        err "No image variant matches this machine."
        err "  Detected: ${GPU_INFO:-unknown} (${GPU_VENDOR}${AMD_GFX_TARGET:+, $AMD_GFX_TARGET}${GPU_MODEL:+, $GPU_MODEL}) on $(uname -m)"
        err ""
        err "  Known variants: $(list_variants | tr '\n' ' ')"
        err "  Force one with VLLMCTL_VARIANT=<id> if you know it fits."
        exit 1
    fi
}

# ─── Host requirements ───────────────────────────────────────────────────────
#
# Each variant lists the checks its image needs, with a severity per check:
#
#   HOSTREQ_n='kind|block|warn|value|message'
#
# Severity lives in the manifest rather than in this code on purpose. Most of
# these images run on hardware nobody here can test, so a check that turns out
# to be wrong has to be a one-line manifest edit rather than a code change and
# a release. VLLMCTL_SKIP_HOSTCHECK=1 bypasses all of them.

# hostreq_gfx VALUE — the detected gfx target is one of VALUE's space list.
hostreq_gfx() {
    local want
    for want in $1; do
        [[ "$AMD_GFX_TARGET" == "$want" ]] && return 0
    done
    return 1
}

hostreq_arch() {
    [[ "$(uname -m)" == "$1" ]]
}

hostreq_compute_cap() {
    local want
    for want in $1; do
        [[ "$GPU_MODEL" == "$want" ]] && return 0
    done
    return 1
}

# hostreq_pci_id VALUE — the detected PCI device id is one of VALUE's list.
# This is the Battlemage / Alchemist split: both are Intel graphics on
# /dev/dri, and only one has a working vLLM path.
hostreq_pci_id() {
    local want
    for want in $1; do
        [[ "$GPU_MODEL" == "$want" ]] && return 0
    done
    return 1
}

# version_ge A B — is version A at least B? Pure sort -V, no bc or python.
version_ge() {
    [[ "$1" == "$2" ]] && return 0
    [[ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -1)" == "$2" ]]
}

hostreq_rocm_min() {
    local have=""
    if need_cmd hipconfig; then
        have="$(hipconfig --version 2>/dev/null | head -1)" || true
    fi
    if [[ -z "$have" ]] && [[ -r /opt/rocm/.info/version ]]; then
        have="$(cat /opt/rocm/.info/version 2>/dev/null)" || true
    fi
    # Undetectable is not the same as too old. Passing here is deliberate:
    # the alternative blocks every host that installed ROCm somewhere we do
    # not look, which is a worse failure than a version we could not read.
    [[ -n "$have" ]] || return 0
    have="${have%%-*}"
    version_ge "$have" "$1"
}

hostreq_nvidia_driver_min() {
    need_cmd nvidia-smi || return 0
    local have
    have="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)" || true
    [[ -n "$have" ]] || return 0
    version_ge "$have" "$1"
}

# check_memlock_limit warns when the host will not let a container pin much
# memory. Not a manifest HOSTREQ_: it is a property of this machine rather than
# of the variant, and every image that offloads anything wants it.
#
# Rootless podman cannot raise a limit above the invoking user's hard limit, so
# a compose file asking for memlock=-1 gets silently clamped to whatever the
# host allows -- 8 MiB on a default Fedora install. The engine then fails to
# pin and says so in one line most of a screen into its startup log:
#
#   PLE offload: locked 0.0 GiB, FAILED to lock 47.7 GiB
#
# which is a long way from the install that caused it.
check_memlock_limit() {
    local hard
    hard="$(ulimit -H -l 2>/dev/null || echo unlimited)"
    [[ "$hard" == "unlimited" ]] && return 0
    # KiB. A gibibyte is far below what an offloaded model pins and far above
    # anything a default install grants, so it separates the two cleanly.
    [[ "$hard" =~ ^[0-9]+$ ]] || return 0
    (( hard >= 1048576 )) && return 0

    warn "This host limits locked memory to ${hard} KiB for your user."
    warn "Models that offload experts or the n-gram table to system RAM pin it,"
    warn "and a rootless container cannot exceed the host limit however the"
    warn "compose file is written. They will load unpinned or not at all."
    warn "To lift it, add to /etc/security/limits.conf and log out and back in:"
    warn "    *  soft  memlock  unlimited"
    warn "    *  hard  memlock  unlimited"
}

# check_host_requirements walks the selected variant's HOSTREQ_* entries in
# order. Runs after the variant is chosen and before anything is built.
check_host_requirements() {
    check_memlock_limit

    if [[ "${VLLMCTL_SKIP_HOSTCHECK:-}" == "1" ]]; then
        warn "VLLMCTL_SKIP_HOSTCHECK=1 — not checking host requirements for ${BUILD_VARIANT}"
        return 0
    fi

    local reqs blocked=0
    reqs="$( ( load_variant_manifest "$BUILD_VARIANT" || exit 0
               local i=1 var
               while :; do
                   var="HOSTREQ_$i"
                   [[ -n "${!var-}" ]] || break
                   printf '%s\n' "${!var}"
                   i=$((i + 1))
               done ) )"
    [[ -n "$reqs" ]] || return 0

    local line kind severity value message
    while IFS= read -r line; do
        [[ -n "$line" ]] || continue
        IFS='|' read -r kind severity value message <<<"$line"

        # An unknown check kind is an authoring mistake in the manifest, not a
        # property of the host. Say so rather than silently passing.
        if ! declare -F "hostreq_${kind}" >/dev/null; then
            warn "${BUILD_VARIANT}: unknown host check \"${kind}\" — skipping (manifest bug)"
            continue
        fi

        if "hostreq_${kind}" "$value"; then
            continue
        fi

        if [[ "$severity" == "block" ]]; then
            err "${BUILD_VARIANT}: ${message}"
            blocked=1
        else
            warn "${BUILD_VARIANT}: ${message}"
        fi
    done <<<"$reqs"

    if [[ $blocked -eq 1 ]]; then
        err ""
        err "  Detected: ${GPU_INFO:-unknown}${AMD_GFX_TARGET:+ ($AMD_GFX_TARGET)}"
        err "  Pick a different variant, or set VLLMCTL_SKIP_HOSTCHECK=1 if you"
        err "  are sure this check is wrong -- and please report it."
        exit 1
    fi
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

    # What this variant needs of the host, from its manifest. Runs here so a
    # mismatch fails before a long pull or a longer build, not after.
    check_host_requirements

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

# port_published_by_us — is this host port published by the vllm-toolchest
# container that is already running?
#
# prompt_ports runs before container_install stops that container, so on any
# machine that already has one, the install warned that 3000 and 8000 were
# taken -- by the very thing it was about to replace -- and then said "Please
# choose alternative ports". Taking that advice moves a working UI off 3000
# for no reason, and the wording makes it sound obligatory rather than
# optional. Seen on a real install, 2026-09-21.
#
# Matching ":PORT->" reads the host side of a mapping only. podman renders one
# as "0.0.0.0:3001->3000/tcp", so a container forwarding 3000 from a different
# host port genuinely does not hold 3000, and must not silence a real clash.
# The container name is spelled literally here, as it is in container_down,
# container_rebuild and get_restart_policy; PODMAN_SERVICE_NAME happens to
# carry the same string but names the systemd unit, not the container.
port_published_by_us() {
    local port="$1" ports
    [[ -n "${CONTAINER_CMD:-}" ]] || return 1
    ports="$($CONTAINER_CMD ps --filter "name=vllm-toolchest" --format '{{.Ports}}' 2>/dev/null)" || return 1
    [[ -n "$ports" ]] || return 1
    grep -q -- ":${port}->" <<<"$ports"
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
    local -a ours=()
    local p
    for p in "$VLLMCTL_PORT" "$VLLMCTL_INFERENCE_PORT"; do
        is_port_available "$p" && continue
        if port_published_by_us "$p"; then
            ours+=("$p")
        else
            warn "Port ${p} is already in use"
            ports_ok=false
        fi
    done

    if [[ "${#ours[@]}" -gt 0 ]]; then
        local held="${ours[*]}"
        ok "Port ${held// /, } held by the running vllm-toolchest, which this install replaces"
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

# tier_note explains a support tier in one clause, so the menu can be read
# without going and finding the README.
tier_note() {
    case "$1" in
        tested)       echo "run on this hardware by this project" ;;
        community)    echo "published upstream, not validated here" ;;
        experimental) echo "little upstream support, expect rough edges" ;;
        *)            echo "$1" ;;
    esac
}

# prompt_variant offers every variant that runs on the detected hardware, with
# the best match preselected.
#
# A menu rather than the yes/no question this used to be: there is more than
# one reasonable image for some cards now -- on gfx1201 the choice between
# dense-FP8 and MoE-MXFP4 tuning is a real one, and it depends on which models
# someone intends to serve, which this script cannot know.
prompt_variant() {
    # Forced on the command line -- nothing to ask. Mirrors detect_variant's
    # handling, including ignoring an unrelated os-release VARIANT.
    if [[ -n "${VLLMCTL_VARIANT:-}" ]]; then
        return
    fi
    if [[ -n "${VARIANT:-}" ]] && variant_exists "${VARIANT}"; then
        return
    fi

    local -a ids=()
    local id
    while IFS= read -r id; do
        [[ -n "$id" ]] && ids+=("$id")
    done < <(matching_variants)

    # One option is not a choice. Say what was picked and move on.
    if [[ "${#ids[@]}" -le 1 ]]; then
        [[ "${#ids[@]}" -eq 1 ]] && BUILD_VARIANT="${ids[0]}"
        return
    fi

    echo ""
    echo -e "  ${BOLD}Detected:${NC} ${GPU_INFO}${AMD_GFX_TARGET:+ (${AMD_GFX_TARGET})}${GPU_MODEL:+ (${GPU_MODEL})}"
    echo ""
    echo -e "  ${BOLD}Which image should this build on?${NC}"
    echo ""

    local -a tiers=()
    local i=1 label summary tier
    for id in "${ids[@]}"; do
        label="$(variant_field "$id" LABEL)"
        summary="$(variant_field "$id" SUMMARY)"
        tier="$(variant_field "$id" TIER)"
        tiers+=("$tier")

        local marker=""
        [[ $i -eq 1 ]] && marker=" ${GREEN}← recommended${NC}"
        printf "    %d) %-14s %-15s %s%b\n" "$i" "$id" "[${tier}]" "$summary" "$marker"
        i=$((i + 1))
    done

    # Explain only the tiers actually on screen.
    echo ""
    local seen="" t
    for t in "${tiers[@]}"; do
        [[ "$seen" == *"|$t|"* ]] && continue
        seen="${seen}|$t|"
        printf "    %-15s %s\n" "[${t}]" "$(tier_note "$t")"
    done
    echo ""

    local choice
    while :; do
        read -rp "$(echo -e "  ${BOLD}Choice${NC} [1]: ")" choice
        choice="${choice:-1}"
        if [[ "$choice" =~ ^[0-9]+$ ]] && [[ "$choice" -ge 1 ]] && [[ "$choice" -le "${#ids[@]}" ]]; then
            break
        fi
        echo "  Enter a number between 1 and ${#ids[@]}."
    done

    BUILD_VARIANT="${ids[$((choice - 1))]}"
    echo ""
    echo -e "  → Building ${BOLD}${BUILD_VARIANT}${NC}"

    local pin
    pin="$(variant_field "$BUILD_VARIANT" VLLM_PIN)"
    if [[ -n "$pin" && "$pin" != "main" ]]; then
        echo "    Pins vLLM ${pin}: models needing a newer vLLM will not load."
    fi
    local url
    url="$(variant_field "$BUILD_VARIANT" DOC_URL)"
    [[ -n "$url" ]] && echo "    $url"
    echo ""
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

# migrate_variant_name maps a variant recorded by an older install onto the one
# that replaced it.
#
# "generic" was two variants wearing one name -- a ROCm source build and a CUDA
# one -- with the GPU found at install time deciding which you got. Now they
# are separate manifests, so the detected vendor resolves which of the two an
# existing install has been running all along. Anything else passes through.
#
# This runs on every command, not just install, so `./setup.sh up` on an
# untouched install keeps working rather than failing on a name nothing ships.
migrate_variant_name() {
    local name="$1"
    if [[ "$name" != "generic" ]]; then
        printf '%s' "$name"; return
    fi
    case "$(detected_vendor)" in
        nvidia) printf 'cuda-source' ;;
        amd)    printf 'rocm-source' ;;
        # Undetectable vendor: leave the old name so the error names it,
        # rather than picking one and building the wrong thing.
        *)      printf '%s' "$name" ;;
    esac
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
        [[ -n "$val" ]] && BUILD_VARIANT="$(migrate_variant_name "$val")" || true
        val="$(grep '^HIP_VISIBLE_DEVICES=' "$env_file" 2>/dev/null | cut -d= -f2)" || true
        [[ -n "$val" ]] && GPU_DEVICES="$val" || true
    fi
}

# ─── Prebuilt base images ────────────────────────────────────────────────────
#
# Variants that layer onto a published image name it in their manifest. Some of
# those images need repairing before they can be built on, which is what the
# rest of this section is about.
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
# conformant image is published, the probe passes and none of this runs. It is
# applied to every prebuilt variant, not just radiance -- several of these
# bases publish to Docker Hub the same way, so the same bug is expected.

# variant_base_ref prints the base image for the selected variant. An explicit
# VLLMCTL_BASE_IMAGE in the environment wins, so an operator can pin a
# different tag -- an alternate build of the same stack, say -- for one build
# without editing the manifest.
#
# .env is deliberately NOT consulted, and used to be. That was a trap, because
# .env is written by this script: the first install recorded the manifest's
# image there, and every later install read it back and preferred it, so a
# manifest bump was resolved, ignored, and then overwritten in .env with the
# new value it had just declined to use. The tree said one thing and the
# running image was another.
#
# It failed quietly in the worst case. ensure_base_image exports what it
# resolves, and an exported value beats .env for compose, so the build ran on
# the stale base while .env claimed the new one. Dockerfile.prebuilt's version
# assertion cannot catch it either whenever two tags carry the same vLLM build
# -- tcclaviger/vllm 28.02.2 and 28.04.9 both report 0.27.0.dev0+g55c98e370a,
# four releases apart, which is exactly when a bump matters and exactly when
# the assertion is blind.
variant_base_ref() {
    if [[ -n "${VLLMCTL_BASE_IMAGE:-}" ]]; then
        echo "$VLLMCTL_BASE_IMAGE"; return
    fi
    variant_field "$BUILD_VARIANT" BASE_IMAGE
}

# warn_stale_env_base says so when .env pins a base image the manifest no
# longer names. Nothing reads that value any more, but an operator who put it
# there by hand deserves to hear that it is being ignored rather than wonder
# why their pin stopped working.
warn_stale_env_base() {
    local pinned manifest
    pinned="$(grep '^VLLMCTL_BASE_IMAGE=' "${SCRIPT_DIR}/.env" 2>/dev/null | cut -d= -f2-)" || true
    [[ -n "$pinned" ]] || return 0
    manifest="$(variant_field "$BUILD_VARIANT" BASE_IMAGE)"
    [[ -n "$manifest" && "$pinned" != "$manifest" ]] || return 0
    warn ".env pins VLLMCTL_BASE_IMAGE=${pinned}"
    warn "but ${BUILD_VARIANT} now declares ${manifest}. Using the manifest."
    warn "Set VLLMCTL_BASE_IMAGE in the environment to override for one build."
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

# Make sure a prebuilt base is present and usable, flattening it if the runtime
# cannot build on top of it. Exports VLLMCTL_BASE_IMAGE for the build.
#
# A variant that builds vLLM from source declares no base image and returns
# here immediately.
ensure_base_image() {
    local src flat tag
    warn_stale_env_base
    src="$(variant_base_ref)"
    [[ -n "$src" ]] || return 0

    if ! image_exists "$src"; then
        log "Pulling ${src} (about 4 GB)..."
        $CONTAINER_CMD pull "$src" || fatal "Could not pull ${src}"
    fi

    if base_is_buildable "$src"; then
        export VLLMCTL_BASE_IMAGE="$src"
        return 0
    fi

    tag="${src##*:}"
    [[ "$tag" == "$src" ]] && tag="latest"
    flat="localhost/vllmctl-${BUILD_VARIANT}-flat:${tag}"

    if image_exists "$flat"; then
        log "Using previously normalized base image ${flat}"
        export VLLMCTL_BASE_IMAGE="$flat"
        return 0
    fi

    warn "${src} cannot be used as a build base by ${CONTAINER_CMD}:"
    warn "it is an OCI manifest carrying Docker-typed layers, which"
    warn "containers/image refuses to rewrite. Normalizing it locally."
    echo ""
    echo "  This flattens the image into a single layer, once per base image"
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
        fatal "Could not normalize ${src}. Building the ${BUILD_VARIANT} variant
       needs either a container runtime that accepts this image (Docker does)
       or a conformant image published upstream."
    fi

    if ! base_is_buildable "$flat"; then
        fatal "Normalized image ${flat} is still not usable as a build base."
    fi

    ok "Normalized base image ready: ${flat}"
    export VLLMCTL_BASE_IMAGE="$flat"
}

# ─── Container operations ────────────────────────────────────────────────────

# Device wiring is a property of the GPU vendor, not of the vLLM stack on top
# of it, so there is one compose file per vendor and every variant of that
# vendor shares it. Falls back to what was detected only when no variant has
# been chosen yet, which is what `./setup.sh detect` does before selection.
variant_vendor() {
    local v
    v="$(variant_field "$BUILD_VARIANT" VENDOR)"
    echo "${v:-$(detected_vendor)}"
}

compose_file() {
    echo "docker-compose.$(variant_vendor).yml"
}

# Which Dockerfile this variant builds with. Every manifest names one.
dockerfile() {
    local f
    f="$(variant_field "$BUILD_VARIANT" DOCKERFILE)"
    echo "${f:-Dockerfile.${GPU_VENDOR}}"
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
    # HF_TOKEN, VLLMCTL_API_KEY, the feature-knob switches -- and truncating
    # the file would silently discard it on every install/rebuild.
    #
    # The VLLMCTL_BASE_IMAGE / DOCKERFILE / VENV_ROOT / STAMP_FILE / VLLM_PIN
    # group is derived from the chosen variant's manifest and rewritten on
    # every install, so hand-editing them does not stick. Change the manifest,
    # or override VLLMCTL_BASE_IMAGE in the environment for a one-off build.
    local managed=(
        VLLMCTL_PORT VLLMCTL_INFERENCE_PORT VLLMCTL_VARIANT VLLMCTL_MODELS_DIR
        VLLMCTL_VENDOR VLLMCTL_BASE_IMAGE VLLMCTL_DOCKERFILE
        VLLMCTL_VENV_ROOT VLLMCTL_STAMP_FILE VLLMCTL_VLLM_PIN VLLMCTL_TUNER_REF
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

        # The variant's build inputs, so compose can substitute them without
        # knowing which variant is selected.
        echo "VLLMCTL_VENDOR=$(variant_vendor)"
        echo "VLLMCTL_DOCKERFILE=$(dockerfile)"
        local _base _venv _stamp _pin
        _base="$(variant_base_ref)"
        _venv="$(variant_field "$BUILD_VARIANT" VENV_ROOT)"
        _stamp="$(variant_field "$BUILD_VARIANT" STAMP_FILE)"
        _pin="$(variant_field "$BUILD_VARIANT" VLLM_PIN)"
        [[ -n "$_base" ]]  && echo "VLLMCTL_BASE_IMAGE=${_base}"
        [[ -n "$_venv" ]]  && echo "VLLMCTL_VENV_ROOT=${_venv}"
        [[ -n "$_stamp" ]] && echo "VLLMCTL_STAMP_FILE=${_stamp}"
        # "main" is the from-source marker, not a tag to pin a prebuilt base
        # against, so it is not written through.
        [[ -n "$_pin" && "$_pin" != "main" ]] && echo "VLLMCTL_VLLM_PIN=${_pin}"
        local _tref
        _tref="$(variant_field "$BUILD_VARIANT" TUNER_REF)"
        [[ -n "$_tref" ]] && echo "VLLMCTL_TUNER_REF=${_tref}"

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
    ensure_base_image
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
    ensure_base_image
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
    ensure_base_image
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
        # Extra capabilities come from the variant's manifest. Unlike compose,
        # a Quadlet unit is generated per install, so this can be per-variant
        # rather than the union across a vendor.
        local extra_caps="" cap
        for cap in $(variant_field "$BUILD_VARIANT" CAPS); do
            extra_caps+="AddCapability=${cap}"$'\n'
        done
        extra_caps="${extra_caps%$'\n'}"
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
# Match what the compose files grant, or auto-start silently produces a weaker
# container than `setup.sh install` does. memlock is the one that matters: an
# offload path that pins host memory -- the PLE n-gram table, pinned expert
# buffers -- fails against the 8 MiB default and falls back to unpinned, or
# does not load at all.
Ulimit=memlock=-1:-1
Ulimit=nofile=65536:65536
${gpu_args}

[Service]
Restart=on-failure
TimeoutStartSec=900
# Ulimit= in [Container] asks podman for a limit; this is what lets it have
# one. A unit cannot raise a limit above its own service limit, and the user
# manager's default is typically 8 MiB -- the same ceiling that makes a
# rootless container fail to pin the PLE table however generous the compose
# file is. Both are needed: this one, and a host that permits it.
LimitMEMLOCK=infinity

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
    local _tier _pin
    _tier="$(variant_field "$BUILD_VARIANT" TIER)"
    _pin="$(variant_field "$BUILD_VARIANT" VLLM_PIN)"
    echo -e "  ${CYAN}Variant${NC}       ${BUILD_VARIANT}${_tier:+ [${_tier}]}"
    if [[ -n "$_pin" && "$_pin" != "main" ]]; then
        echo -e "  ${CYAN}vLLM${NC}          pinned ${_pin} — newer models will not load"
    else
        echo -e "  ${CYAN}vLLM${NC}          tracks main"
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

# print_variants lists every variant and whether it fits this machine. Useful
# before an install, and the only way to see the ones that do not match without
# reading the manifests.
print_variants() {
    local -a fits=() others=()
    local id
    for id in $(list_variants); do
        if variant_matches "$id"; then fits+=("$id"); else others+=("$id"); fi
    done

    echo ""
    echo -e "  ${BOLD}Detected:${NC} ${GPU_INFO}${AMD_GFX_TARGET:+ (${AMD_GFX_TARGET})}${GPU_MODEL:+ (${GPU_MODEL})} on $(uname -m)"

    local rec
    rec="$(recommended_variant)"

    echo ""
    if [[ "${#fits[@]}" -gt 0 ]]; then
        echo -e "  ${BOLD}Runs on this machine${NC}"
        # Re-order to match what the install menu shows, best first.
        local ordered
        ordered="$(matching_variants)"
        while IFS= read -r id; do
            [[ -n "$id" ]] || continue
            printf "    %-14s %-15s %s%b\n" "$id" "[$(variant_field "$id" TIER)]" \
                "$(variant_field "$id" SUMMARY)" \
                "$([[ "$id" == "$rec" ]] && echo " ${GREEN}← recommended${NC}")"
        done <<<"$ordered"
    else
        echo -e "  ${YELLOW}No variant matches this machine.${NC}"
    fi

    if [[ "${#others[@]}" -gt 0 ]]; then
        echo ""
        echo -e "  ${BOLD}For other hardware${NC}  (VLLMCTL_VARIANT=<id> to force one anyway)"
        for id in "${others[@]}"; do
            printf "    %-14s %-15s %s\n" "$id" "[$(variant_field "$id" TIER)]" \
                "$(variant_field "$id" SUMMARY)"
        done
    fi
    echo ""
}

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
  detect      Print detected GPU backend, image variant and vendor, exit
  variants    List every image variant and whether it fits this machine, exit
  help        Show this help message

Image variants:
  Each variant is a different vLLM stack, described by one file in variants/.
  `install` detects the GPU, offers the ones that fit and preselects the best
  match; the answer is stored in .env and reused by every later command.
  Run `./setup.sh variants` to see them all.

  Support tiers: [tested] means someone on this project ran it on that
  hardware; [community] means it is published upstream but not validated here;
  [experimental] means little upstream support.

Environment variables:
  GPU=cuda|rocm|xpu                    Override GPU auto-detection
  VLLMCTL_VARIANT=<id>                 Override image variant (skips the prompt)
  RUNTIME=docker|podman                Override container runtime auto-detection
  VLLMCTL_BASE_IMAGE=<ref>             Override a prebuilt variant's base image
  VLLMCTL_SKIP_HOSTCHECK=1             Skip the variant's host requirement checks

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
  VLLMCTL_VARIANT=radiance ./setup.sh install   # build a specific variant
  VLLMCTL_VARIANT=rocm-source ./setup.sh rebuild # switch to the from-source image
USAGE
}

main() {
    local command="${1:-help}"
    cd "$SCRIPT_DIR"

    # Used by `make env-example` and the drift test. Not in usage: it exists
    # to keep generated documentation in step with the manifests, and has no
    # meaning to someone installing.
    if [[ "$command" == "--print-env-knobs" ]]; then
        print_env_knobs
        exit 0
    fi
    if [[ "$command" == "--write-env-example" ]]; then
        write_env_example "${2:-.env.example}"
        exit 0
    fi

    case "$command" in
        install|uninstall|up|down|rebuild|quick|logs|detect|variants|status|enable|disable) ;;
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
        echo "$GPU_VENDOR $BUILD_VARIANT $(variant_vendor)"
        exit 0
    fi

    if [[ "$command" == "variants" ]]; then
        print_variants
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

# Run only when executed, not when sourced. Sourcing is how the manifest-reader
# parity test gets at load_variant_manifest and friends: the bash reader and
# the Go parser have to agree about every variants/*.conf, and the only way to
# prove that is to exercise the same functions this script actually uses.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi

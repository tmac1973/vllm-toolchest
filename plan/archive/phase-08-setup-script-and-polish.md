# Phase 8: Setup Script & Polish

**Goal:** One-command setup for end users (`./setup.sh`), a comprehensive Makefile for developers, UI polish for production readiness, a testing strategy, and documentation.

---

## 1. setup.sh Architecture

Adapt from llama-toolchest's ~1000 line setup script. The script is POSIX-compatible (runs under bash, not requiring bashisms beyond what's universally available). It is the primary onboarding experience -- it must be bulletproof, informative, and idempotent (safe to re-run).

### 1.1 Script Structure

```
setup.sh
├── Banner / version display
├── Pre-flight checks (root vs non-root, required commands)
├── Distro detection
├── GPU detection
│   ├── AMD path (RDNA4 focus)
│   └── NVIDIA path (future)
├── Container runtime detection/installation
├── GPU passthrough configuration
├── .env generation
├── Image pull/build
├── First-run launch
└── Post-launch health check + URL display
```

### 1.2 Distro Detection

Detect the Linux distribution to select the correct package manager and package names.

**Detection method:**
1. Parse `/etc/os-release` for `ID` and `ID_LIKE` fields.
2. Map to package manager:

| Distro | ID | Package Manager | Notes |
|--------|-----|-----------------|-------|
| Fedora | `fedora` | `dnf` | Primary dev target |
| RHEL/CentOS Stream | `rhel`, `centos` | `dnf` | May need EPEL |
| Ubuntu | `ubuntu` | `apt` | Need `apt-get` for non-interactive |
| Debian | `debian` | `apt` | |
| Arch | `arch` | `pacman` | Also Manjaro, EndeavourOS via `ID_LIKE` |
| openSUSE Tumbleweed | `opensuse-tumbleweed` | `zypper` | |
| openSUSE Leap | `opensuse-leap` | `zypper` | |

**Fallback:** If distro is unrecognized, warn the user and attempt to continue with manual instructions.

**Data captured:**
```bash
DISTRO_ID=""        # e.g., "fedora"
DISTRO_VERSION=""   # e.g., "43"
PKG_MANAGER=""      # dnf | apt | pacman | zypper
PKG_INSTALL=""      # "dnf install -y" | "apt-get install -y" | "pacman -S --noconfirm" | "zypper install -y"
```

### 1.3 GPU Detection

#### AMD Detection

**Step 1: PCI device scan**
```bash
lspci -nn | grep -i 'vga\|display\|3d' | grep -i 'amd\|ati\|radeon'
```

Parse output for:
- Device name (e.g., "Radeon RX 9700 XT")
- PCI vendor:device ID (e.g., `1002:7480`)
- Count number of AMD GPUs (for multi-GPU detection)

**Step 2: RDNA4 / gfx1201 identification**

Known RDNA4 PCI device IDs (update as more SKUs launch):
- `1002:7480` -- RX 9070 XT
- `1002:7481` -- RX 9070
- `1002:7490` -- RX 9700 XT (target hardware)
- Fallback: check via `rocm-smi --showproductname` if available

If the PCI ID is not in the known list but the vendor is AMD, prompt the user to confirm GPU generation.

**Step 3: Device node verification**
```bash
# /dev/kfd must exist (ROCm kernel driver loaded)
[ -e /dev/kfd ] || warn "ROCm kernel driver not loaded -- /dev/kfd not found"

# At least one render node must exist
ls /dev/dri/renderD* 2>/dev/null || warn "No DRI render nodes found"

# Check permissions
[ -r /dev/kfd ] && [ -w /dev/kfd ] || warn "/dev/kfd not accessible by current user"
```

**Step 4: rocm-smi check (optional)**

If `rocm-smi` is available (host ROCm installation):
```bash
rocm-smi --showproductname   # GPU names
rocm-smi --showmeminfo vram  # VRAM sizes
rocm-smi --showgfxversion    # gfx version confirmation
```

This is informational only -- rocm-smi is inside the container, not required on the host.

**Step 5: HSA_OVERRIDE_GFX_VERSION determination**

For RDNA4 (gfx1201), this may be needed if the ROCm version in the container doesn't natively support gfx1201:
```bash
HSA_OVERRIDE_GFX_VERSION="12.0.1"
```
Include in `.env` if gfx1201 is detected. The container's entrypoint can also set this, but having it in `.env` makes it visible and overridable.

#### NVIDIA Detection (for future Phase 9)

```bash
lspci -nn | grep -i 'vga\|display\|3d' | grep -i 'nvidia'
```

If NVIDIA is detected:
- Check for `nvidia-smi` on host
- Parse GPU name and driver version
- Detect NVIDIA Container Toolkit (`nvidia-ctk --version`)
- Set `GPU_TYPE=nvidia` in `.env`

#### Multi-GPU Detection

```bash
GPU_COUNT=$(lspci -nn | grep -i 'vga\|display\|3d' | grep -ci 'amd\|radeon')
```

For multi-GPU:
- Enumerate all render nodes: `/dev/dri/renderD128`, `/dev/dri/renderD129`, etc.
- Verify all GPUs are the same model (mixed GPU configs are unsupported by vLLM TP).
- Store `GPU_COUNT` in `.env`.

### 1.4 Container Runtime Detection and Installation

#### Detection Priority

1. Check if Docker is available and functional: `docker info >/dev/null 2>&1`
2. Check if Podman is available: `podman info >/dev/null 2>&1`
3. If neither: offer to install one (prefer Docker for broader compatibility, but respect user choice).

#### Docker Detection

```bash
# Check docker binary
command -v docker >/dev/null 2>&1

# Check docker daemon is running
docker info >/dev/null 2>&1

# Check docker compose plugin (v2)
docker compose version >/dev/null 2>&1

# Fallback: check for standalone docker-compose (v1)
command -v docker-compose >/dev/null 2>&1
```

**Docker installation if missing (by distro):**

| Distro | Method |
|--------|--------|
| Fedora | `dnf install docker-ce docker-ce-cli containerd.io docker-compose-plugin` (via Docker repo) |
| Ubuntu/Debian | Docker's official apt repo |
| Arch | `pacman -S docker docker-compose` |
| openSUSE | `zypper install docker docker-compose` |

Post-install: enable and start docker service, add user to `docker` group.

#### Podman Detection

```bash
command -v podman >/dev/null 2>&1
command -v podman-compose >/dev/null 2>&1
```

**Podman installation if missing:**

| Distro | Method |
|--------|--------|
| Fedora | `dnf install podman podman-compose` (usually pre-installed) |
| Ubuntu/Debian | `apt-get install podman podman-compose` |
| Arch | `pacman -S podman podman-compose` |
| openSUSE | `zypper install podman podman-compose` |

#### Runtime Selection

```bash
if docker_available && podman_available; then
    prompt "Both Docker and Podman detected. Which do you prefer? [docker/podman]"
elif docker_available; then
    CONTAINER_RUNTIME="docker"
    COMPOSE_CMD="docker compose"
elif podman_available; then
    CONTAINER_RUNTIME="podman"
    COMPOSE_CMD="podman-compose"
else
    offer_installation
fi
```

### 1.5 GPU Passthrough Configuration

#### AMD GPU Passthrough

**Group GID detection:**
```bash
# Video group GID (needed for /dev/dri/card*)
VIDEO_GID=$(getent group video | cut -d: -f3)

# Render group GID (needed for /dev/dri/renderD*)
RENDER_GID=$(getent group render | cut -d: -f3)
```

**User group membership check:**
```bash
# Check if current user is in video and render groups
id -nG "$USER" | grep -qw video || warn "User $USER is not in the 'video' group"
id -nG "$USER" | grep -qw render || warn "User $USER is not in the 'render' group"

# Offer to add user to groups
if ! id -nG "$USER" | grep -qw render; then
    prompt "Add $USER to 'render' group? (required for GPU access) [Y/n]"
    sudo usermod -aG render "$USER"
    echo "You may need to log out and back in for group changes to take effect."
fi
```

**CDI config for Podman (Container Device Interface):**

Podman 4.1+ supports CDI for GPU passthrough. Generate a CDI spec if using Podman:

```bash
if [ "$CONTAINER_RUNTIME" = "podman" ]; then
    # Check if ROCm CDI spec exists
    CDI_SPEC="/etc/cdi/rocm.yaml"
    if [ ! -f "$CDI_SPEC" ]; then
        echo "Generating ROCm CDI spec for Podman..."
        # Generate CDI spec pointing to /dev/kfd and all /dev/dri/renderD* nodes
        cat > /tmp/rocm-cdi.yaml << 'CDIEOF'
cdiVersion: "0.5.0"
kind: "rocm.amd.com/gpu"
devices:
  - name: "all"
    containerEdits:
      deviceNodes:
        - path: /dev/kfd
        - path: /dev/dri/renderD128
      env:
        - HSA_OVERRIDE_GFX_VERSION=12.0.1
CDIEOF
        sudo mkdir -p /etc/cdi
        sudo cp /tmp/rocm-cdi.yaml "$CDI_SPEC"
    fi
fi
```

**Permission verification for /dev/kfd:**
```bash
# /dev/kfd must be readable and writable
if [ ! -w /dev/kfd ]; then
    echo "WARNING: /dev/kfd is not writable by current user."
    echo "This is required for GPU compute. Checking udev rules..."

    # Check for ROCm udev rule
    if [ ! -f /etc/udev/rules.d/70-amdgpu.rules ]; then
        echo "Creating udev rule for AMD GPU access..."
        echo 'SUBSYSTEM=="kfd", KERNEL=="kfd", TAG+="uaccess", GROUP="render", MODE="0660"' | \
            sudo tee /etc/udev/rules.d/70-amdgpu.rules
        sudo udevadm control --reload-rules
        sudo udevadm trigger
    fi
fi
```

### 1.6 .env Generation

Generate `.env` from all detected configuration.

```bash
cat > .env << ENVEOF
# vllm-toolchest configuration
# Generated by setup.sh on $(date -Iseconds)

# GPU Configuration
GPU_TYPE=${GPU_TYPE}                    # amd | nvidia
GPU_COUNT=${GPU_COUNT}                  # Number of GPUs detected
RENDER_GID=${RENDER_GID}               # Render group GID for GPU access
VIDEO_GID=${VIDEO_GID}                 # Video group GID

# Container Configuration
COMPOSE_FILE=${COMPOSE_FILE}           # docker-compose.yml (rocm) or docker-compose.cuda.yml
CONTAINER_RUNTIME=${CONTAINER_RUNTIME} # docker | podman

# Paths
MODEL_DIR=${MODEL_DIR}                 # Where to store downloaded models
DATA_DIR=${DATA_DIR}                   # Persistent data directory

# AMD-specific
HSA_OVERRIDE_GFX_VERSION=${HSA_GFX_VER}  # e.g., 12.0.1 for gfx1201
PYTORCH_ROCM_ARCH=${ROCM_ARCH}           # e.g., gfx1201

# API
VLLMCTL_API_KEY=${API_KEY}             # Auto-generated if not set

# Ports
UI_PORT=3000                           # Web UI port
VLLM_PORT=8000                         # vLLM API port (internal, proxied through UI_PORT)
ENVEOF
```

**Model directory handling:**
```bash
# Default: ./models (relative to project root)
# Prompt user for custom path
echo "Where should models be stored?"
echo "  1) ./models (default, ~50-200GB per model)"
echo "  2) Custom path"
read -r choice
case $choice in
    2) read -rp "Enter model directory path: " MODEL_DIR ;;
    *) MODEL_DIR="$(pwd)/models" ;;
esac
mkdir -p "$MODEL_DIR"
```

**API key generation:**
```bash
# Generate a random 32-char API key
API_KEY=$(openssl rand -base64 32 | tr -dc 'a-zA-Z0-9' | head -c 32)
echo "Generated API key: $API_KEY"
echo "Store this securely. It's required for API access."
```

### 1.7 Image Pull or Build

```bash
IMAGE_NAME="ghcr.io/tmac1973/vllm-toolchest"
IMAGE_TAG="latest-rocm"  # or latest-cuda for NVIDIA

echo "Pulling container image..."
if $CONTAINER_RUNTIME pull "${IMAGE_NAME}:${IMAGE_TAG}"; then
    echo "Image pulled successfully."
else
    echo "Pull failed. Building from source..."
    if [ "$GPU_TYPE" = "amd" ]; then
        $COMPOSE_CMD build --no-cache
    else
        $COMPOSE_CMD -f docker-compose.cuda.yml build --no-cache
    fi
fi
```

Build from source is the fallback. For RDNA4, the image likely needs to be built from source due to TheRock nightly dependencies that may not be in a published image.

### 1.8 First-Run Launch

```bash
echo "Starting vllm-toolchest..."

# Select compose file based on GPU type
if [ "$GPU_TYPE" = "amd" ]; then
    COMPOSE_FILE="docker-compose.yml"
else
    COMPOSE_FILE="docker-compose.cuda.yml"
fi

# Add models volume override if using external model directory
COMPOSE_FILES="-f $COMPOSE_FILE"
if [ "$MODEL_DIR" != "$(pwd)/models" ]; then
    COMPOSE_FILES="$COMPOSE_FILES -f docker-compose.models.yml"
fi

$COMPOSE_CMD $COMPOSE_FILES up -d
```

### 1.9 Post-Launch Health Check and URL Display

```bash
echo "Waiting for vllm-toolchest to start..."

MAX_WAIT=60
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -sf http://localhost:${UI_PORT}/api/health >/dev/null 2>&1; then
        break
    fi
    sleep 2
    WAITED=$((WAITED + 2))
    printf "."
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo ""
    echo "WARNING: Service did not become healthy within ${MAX_WAIT}s."
    echo "Check logs: $COMPOSE_CMD logs -f"
else
    echo ""
    echo "============================================"
    echo "  vllm-toolchest is running!"
    echo "============================================"
    echo ""
    echo "  Web UI:  http://localhost:${UI_PORT}"
    echo "  API:     http://localhost:${UI_PORT}/v1"
    echo ""
    echo "  API Key: ${API_KEY}"
    echo ""
    echo "  GPU:     ${GPU_NAME} (${GPU_TYPE})"
    echo "  GPUs:    ${GPU_COUNT}"
    echo ""
    echo "  Next steps:"
    echo "    1. Open the Web UI"
    echo "    2. Go to Settings, add your HuggingFace token"
    echo "    3. Browse models and download one"
    echo "    4. Start the inference server"
    echo ""
    echo "  Logs:    $COMPOSE_CMD logs -f"
    echo "  Stop:    $COMPOSE_CMD down"
    echo "============================================"
fi
```

---

## 2. Makefile Targets

File: `Makefile` at project root.

### Full Makefile Design

```makefile
# Auto-detect GPU type
GPU_TYPE := $(shell lspci -nn 2>/dev/null | grep -qi nvidia && echo "nvidia" || echo "amd")
COMPOSE_FILE := $(if $(filter nvidia,$(GPU_TYPE)),docker-compose.cuda.yml,docker-compose.yml)

# Auto-detect container runtime
RUNTIME := $(shell command -v docker >/dev/null 2>&1 && echo "docker" || echo "podman")
COMPOSE := $(if $(filter docker,$(RUNTIME)),docker compose,podman-compose)

# Go build variables
BINARY := vllmctl
VERSION := $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
BUILD_TIME := $(shell date -u +%Y-%m-%dT%H:%M:%SZ)
LDFLAGS := -X main.version=$(VERSION) -X main.buildTime=$(BUILD_TIME)
```

### Targets

**`make build`** -- Build Go binary locally
```makefile
build:
	go build -ldflags "$(LDFLAGS)" -o bin/$(BINARY) ./cmd/vllmctl
```

**`make dev`** -- Run Go binary with hot reload
```makefile
dev:
	@command -v air >/dev/null 2>&1 || go install github.com/air-verse/air@latest
	air -c .air.toml
```

Requires an `.air.toml` config file:
- Watch: `cmd/`, `internal/`, `web/templates/`, `web/static/`
- Exclude: `vendor/`, `bin/`, `plan/`
- Build command: `go build -ldflags "$(LDFLAGS)" -o ./tmp/vllmctl ./cmd/vllmctl`
- Run command: `./tmp/vllmctl --dev` (dev mode: no vLLM process, mock GPU data)

**`make docker-build`** -- Build container image
```makefile
docker-build:
	$(RUNTIME) build -t vllm-toolchest:local -f Dockerfile .
```

For CUDA: `make docker-build-cuda`
```makefile
docker-build-cuda:
	$(RUNTIME) build -t vllm-toolchest:local-cuda -f Dockerfile.cuda .
```

**`make up`** -- Start containers
```makefile
up:
	$(COMPOSE) -f $(COMPOSE_FILE) up -d
```

**`make down`** -- Stop containers
```makefile
down:
	$(COMPOSE) -f $(COMPOSE_FILE) down
```

**`make logs`** -- Tail container logs
```makefile
logs:
	$(COMPOSE) -f $(COMPOSE_FILE) logs -f --tail=100
```

**`make shell`** -- Exec into running container
```makefile
shell:
	$(RUNTIME) exec -it vllm-toolchest /bin/bash
```

**`make test`** -- Run Go tests
```makefile
test:
	go test -v -race -count=1 ./...
```

**`make clean`** -- Remove build artifacts
```makefile
clean:
	rm -rf bin/ tmp/
	go clean -cache
```

**Additional utility targets:**

```makefile
# Lint Go code
lint:
	golangci-lint run ./...

# Format Go code
fmt:
	gofmt -w .

# Generate embedded assets hash (for cache busting)
assets:
	@echo "Static assets are go:embed'd -- no build step needed"

# Run API test scripts
test-api:
	./scripts/test-api.sh

# Show detected GPU info
gpu-info:
	@echo "GPU Type: $(GPU_TYPE)"
	@echo "Compose File: $(COMPOSE_FILE)"
	@echo "Runtime: $(RUNTIME)"
	@if [ "$(GPU_TYPE)" = "amd" ]; then rocm-smi --showproductname 2>/dev/null || echo "rocm-smi not available on host"; fi
	@if [ "$(GPU_TYPE)" = "nvidia" ]; then nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"; fi

# Rebuild and restart
restart: down docker-build up

# Full dev reset (careful -- removes data)
reset:
	$(COMPOSE) -f $(COMPOSE_FILE) down -v
	rm -rf data/
```

---

## 3. UI Polish

### 3.1 Error Handling

**User-friendly error pages (`web/templates/error.html`):**
- 404: "Page not found. Try navigating from the sidebar." with link back to dashboard.
- 500: "Something went wrong. Check the logs for details." with link to server log page.
- Custom error handler registered on the chi router that returns HTML or JSON based on `HX-Request` header.

**Toast notifications for async operations:**

Implement a lightweight toast system using htmx's `hx-trigger` response headers:

```go
// In Go handler, after successful async operation:
w.Header().Set("HX-Trigger", `{"showToast": {"message": "Model download started", "type": "info"}}`)
```

Client-side JavaScript listener (in `layout.html`):
```javascript
// Listen for custom events from htmx response headers
document.body.addEventListener("showToast", function(evt) {
    // Create toast element, auto-dismiss after 5s
    // Types: success (green), error (red), info (blue), warning (yellow)
});
```

Toast container: fixed position bottom-right, stacks vertically, auto-dismiss with fade animation.

**API error responses:**
- All JSON errors follow a consistent shape: `{"error": "message", "code": "ERROR_CODE", "details": {...}}`
- Error codes are documented constants (e.g., `MODEL_NOT_FOUND`, `VLLM_NOT_RUNNING`, `INVALID_CONFIG`, `DOWNLOAD_FAILED`).
- HTML error responses use the same toast system via HX-Trigger headers.

### 3.2 Loading States

**Skeleton screens for htmx swaps:**

When htmx makes a request and swaps content, show a skeleton placeholder during loading.

Pattern using `hx-indicator`:
```html
<div id="model-list" hx-get="/api/models" hx-trigger="load" hx-indicator="#model-list-skeleton">
  <div id="model-list-skeleton" class="htmx-indicator skeleton">
    <!-- Pico CSS skeleton: gray boxes mimicking the layout -->
    <div class="skeleton-row" aria-hidden="true"></div>
    <div class="skeleton-row" aria-hidden="true"></div>
    <div class="skeleton-row" aria-hidden="true"></div>
  </div>
</div>
```

CSS for skeleton:
```css
.skeleton-row {
  height: 1.2em;
  background: var(--pico-muted-border-color);
  border-radius: 4px;
  margin-bottom: 0.5em;
  animation: skeleton-pulse 1.5s ease-in-out infinite;
}
@keyframes skeleton-pulse {
  0%, 100% { opacity: 0.4; }
  50% { opacity: 0.8; }
}
```

**Spinner for long operations:**
- Model downloads: progress bar (SSE-driven, already in Phase 3)
- vLLM start: spinner with elapsed time counter, log preview
- Benchmark runs: progress bar with step counter
- Settings save: inline spinner next to save button

### 3.3 Mobile Responsiveness

Pico CSS handles most responsive layout. Additional work needed:

**Sidebar collapse on small screens:**
- Below 768px width: sidebar collapses to a hamburger menu.
- Implementation: CSS media query hides sidebar, shows hamburger button in top nav.
- Hamburger button toggles sidebar visibility via a small JS snippet (no framework needed).
- Sidebar overlays content on mobile (position: fixed, z-index above main content).

```css
@media (max-width: 768px) {
  .sidebar { display: none; position: fixed; z-index: 100; }
  .sidebar.open { display: block; }
  .hamburger { display: block; }
  .main-content { margin-left: 0; }
}
@media (min-width: 769px) {
  .hamburger { display: none; }
  .sidebar { display: block; }
}
```

**Table responsiveness:**
- Benchmark comparison tables: horizontal scroll on narrow screens (`overflow-x: auto` on table container).
- Model list: card layout on mobile instead of table rows.

**Touch targets:**
- All buttons and interactive elements: minimum 44x44px touch target (Pico CSS defaults handle this).
- Sliders (GPU memory utilization, context length): ensure thumb is large enough for touch.

### 3.4 Keyboard Shortcuts

Implement global keyboard shortcuts using a lightweight JS handler in `layout.html`.

| Shortcut | Action |
|----------|--------|
| `Ctrl+Shift+S` | Start/Stop vLLM service (toggle) |
| `Ctrl+Shift+1` | Navigate to Dashboard |
| `Ctrl+Shift+2` | Navigate to Models |
| `Ctrl+Shift+3` | Navigate to Server |
| `Ctrl+Shift+4` | Navigate to Benchmarks |
| `Ctrl+Shift+5` | Navigate to Settings |
| `Ctrl+Shift+L` | Toggle log panel |
| `Escape` | Close any open modal |
| `/` | Focus search input (on model browse page) |

Implementation: single `keydown` event listener on `document`, check for modifier keys, prevent default, trigger navigation via `htmx.ajax()` or `window.location`.

Keyboard shortcuts are displayed in a help modal accessible via `?` key or a `(?)` link in the sidebar footer.

### 3.5 Favicon and Page Titles

**Favicon:**
- SVG favicon embedded in `layout.html` (no external file needed, keeps go:embed simple).
- Design: stylized wrench + GPU icon, or simple "V" monogram.
- Also provide `apple-touch-icon` for mobile bookmarks.

**Page titles:**
- Dynamic `<title>` per page: "Dashboard - vllm-toolchest", "Models - vllm-toolchest", etc.
- Include service status in title when on Server page: "Server (Running) - vllm-toolchest".
- Template pattern: `{{ define "title" }}Dashboard{{ end }}` in each page template, rendered in layout.

### 3.6 Empty States

Every list/table view needs an empty state with helpful guidance:

| Page | Empty State Message |
|------|-------------------|
| Models (inventory) | "No models downloaded yet. Browse HuggingFace to find and download a model." with button linking to Browse page. |
| Models (browse search) | "Search HuggingFace for text generation models. Try 'Qwen2.5' or 'Llama 3'." |
| Benchmarks | "No benchmark results yet. Start the inference server and run a benchmark." with link to Server page if vLLM not running. |
| Server logs | "No log output yet. Start the inference server to see logs." |
| Server (no model selected) | "No model selected. Go to Models to configure and enable a model." |

Empty states use Pico CSS's muted text styling with an optional icon (SVG inline).

### 3.7 Log Viewer Improvements

Enhance the SSE-powered log viewer (`log-panel.js`) from llama-toolchest:

**Search:**
- Text input above log panel: filters visible log lines client-side (JavaScript `textContent.includes()`).
- Highlight matching text in yellow.
- Show match count: "5 of 234 lines match".

**Filter by level:**
- Buttons: ALL | DEBUG | INFO | WARN | ERROR
- Each log line has a `data-level` attribute parsed from the log format.
- Active filter hides non-matching lines via CSS class toggle.
- ERROR lines always shown in red, WARN in yellow, regardless of filter.

**Copy button:**
- "Copy All" button: copies entire visible log to clipboard via `navigator.clipboard.writeText()`.
- "Copy Selected" button: copies text selection if any, otherwise copies all.
- Visual feedback: button text changes to "Copied!" for 2 seconds.

**Auto-scroll behavior:**
- Auto-scroll to bottom when new lines arrive (default: on).
- Auto-scroll pauses when user scrolls up (detect via scroll position).
- "Jump to bottom" floating button appears when not at bottom.
- Small badge showing number of new lines since user scrolled away.

**Line wrapping toggle:**
- Button to toggle between `white-space: pre` (horizontal scroll) and `white-space: pre-wrap` (wrap long lines).

### 3.8 Accessibility

**ARIA labels:**
- All icon-only buttons have `aria-label` attributes (e.g., start button: `aria-label="Start vLLM server"`).
- Status indicators have `aria-live="polite"` for screen reader announcements.
- Toast notifications use `role="alert"`.
- Sidebar navigation uses `<nav aria-label="Main navigation">`.
- Modal dialogs use `role="dialog"` and `aria-modal="true"`.

**Focus management for htmx swaps:**
- After htmx content swap, focus the first heading or interactive element in the swapped content.
- Implementation: htmx `afterSwap` event listener that calls `.focus()` on the appropriate element.
- Restore focus to trigger element when modal closes.

**Keyboard navigation:**
- All interactive elements are reachable via Tab.
- Modal trap: Tab cycles within modal when open.
- Dropdown menus close on Escape.

**Color contrast:**
- All themes must meet WCAG 2.1 AA contrast ratio (4.5:1 for text, 3:1 for large text).
- Terminal themes (green, amber) need careful selection of background/foreground values.

**Reduced motion:**
- Respect `prefers-reduced-motion` media query: disable skeleton pulse animation, toast slide-in, etc.

---

## 4. Testing Strategy

### 4.1 API Test Scripts (`scripts/`)

Shell-based API tests adapted from llama-toolchest's pattern. Each script is standalone and can be run independently.

**File: `scripts/test-api.sh`** -- Master test runner
```bash
# Runs all test scripts in sequence, reports pass/fail
# Usage: ./scripts/test-api.sh [base_url] [api_key]
# Default: http://localhost:3000
```

**Individual test scripts:**

| Script | What It Tests |
|--------|---------------|
| `scripts/test-health.sh` | `GET /api/health` returns 200 |
| `scripts/test-settings.sh` | GET/PUT settings, validate round-trip |
| `scripts/test-settings-connection.sh` | Test connection and HF token endpoints |
| `scripts/test-models-registry.sh` | List models, get model details |
| `scripts/test-models-config.sh` | Update model config, validate persistence |
| `scripts/test-hf-search.sh` | HuggingFace search and model info |
| `scripts/test-service.sh` | Start/stop vLLM, check status transitions |
| `scripts/test-proxy.sh` | /v1/models, /v1/chat/completions through proxy |
| `scripts/test-proxy-tools.sh` | Tool use / function calling through proxy |
| `scripts/test-benchmark.sh` | Start benchmark, poll progress, get results |
| `scripts/test-monitor.sh` | GPU/CPU metrics endpoint |
| `scripts/test-sse.sh` | SSE stream connectivity (log stream, monitor stream) |

**Test script pattern (each script follows this structure):**
```bash
#!/bin/bash
BASE_URL="${1:-http://localhost:3000}"
API_KEY="${2:-}"
PASS=0; FAIL=0

assert_status() {
    local desc="$1" expected="$2" actual="$3"
    if [ "$expected" = "$actual" ]; then
        echo "PASS: $desc"
        PASS=$((PASS + 1))
    else
        echo "FAIL: $desc (expected $expected, got $actual)"
        FAIL=$((FAIL + 1))
    fi
}

# Test cases...

echo "Results: $PASS passed, $FAIL failed"
[ $FAIL -eq 0 ] || exit 1
```

**Tool use test (`scripts/test-proxy-tools.sh`):**

Tests OpenAI-compatible function calling through the proxy:
1. POST `/v1/chat/completions` with `tools` array (weather function schema).
2. Verify response contains `tool_calls` array.
3. Send follow-up message with `tool` role and function result.
4. Verify final response incorporates the tool result.

This test requires vLLM running with `--enable-auto-tool-choice` and `--tool-call-parser`, so it's conditional on service state.

### 4.2 Go Unit Tests

Unit tests live alongside their source files (`*_test.go` convention).

**Config parsing tests (`internal/config/config_test.go`):**
- Load valid YAML, verify all fields populated correctly.
- Load YAML with missing fields, verify defaults applied.
- Load YAML with invalid values, verify validation errors.
- Environment variable overrides: set env vars, verify they take precedence.
- Config merge: partial update preserves unset fields.
- Duration parsing: "5s", "1m30s", "300s" all parse correctly.
- Sensitive field sanitization: verify api_key and hf_token are masked in output.
- Marlin preference logic: given a GPTQ 4-bit model config with `prefer_marlin=true`, verify output quantization flag is "marlin".

**VRAM estimation tests (`internal/models/vram_test.go`):**
- Known models with known VRAM requirements:
  - Llama-3.1-8B FP16: ~16GB weights, test calculation matches.
  - Qwen2.5-72B AWQ 4-bit: ~36GB weights, test TP=2 fits in 2x 32GB.
  - Small model (1B params): fits in single GPU with room for KV cache.
- Edge cases:
  - Model with GQA (grouped query attention): fewer KV heads than Q heads.
  - FP8 quantization: 1 byte per parameter.
  - KV cache estimation at various context lengths.

**HF config parsing tests (`internal/models/hfconfig_test.go`):**
- Parse Llama config.json: verify architecture, layers, heads, hidden_size.
- Parse Mistral config.json: verify sliding window attention fields.
- Parse Qwen2 config.json: verify vocab size, tie_word_embeddings.
- Parse AWQ quantized model config: verify quant_config section.
- Parse GPTQ quantized model config: verify bits, group_size, desc_act.
- Missing fields: handle models with non-standard config gracefully.
- Tool calling detection from tokenizer_config.json chat_template.

**Model registry tests (`internal/models/registry_test.go`):**
- Add model, list models, get model by ID.
- Update model config, verify persistence.
- Delete model, verify removal from registry and correct return value.
- Startup scan: unregistered model directory gets auto-registered.
- Startup scan: registered model with missing directory gets flagged.
- Concurrent access: multiple goroutines reading/writing registry.

**Benchmark stats tests (`internal/benchmark/stats_test.go`):**
- Mean, min, max, p50, p95, p99 calculations with known datasets.
- Edge case: single data point.
- Edge case: all identical values.
- Edge case: empty dataset returns zeros (not panic).
- Tokens/sec calculation: total_tokens / elapsed_seconds.

**Process manager tests (`internal/process/manager_test.go`):**
- Mock vLLM command: test process start, health check polling, graceful stop.
- Auto-restart: simulate process crash, verify restart with backoff.
- Startup timeout: process never becomes healthy, verify timeout error.
- Command construction: given model config, verify correct `vllm serve` flags are generated.
- Tool use flags: verify `--enable-auto-tool-choice` and `--tool-call-parser hermes` are included when configured.
- Quantization flags: verify `--quantization marlin` when prefer_marlin and compatible model.

### 4.3 Integration Test

A single end-to-end integration test that validates the full flow.

**File: `scripts/integration-test.sh`**

Prerequisites: running container with GPU access.

**Steps:**
1. Verify container is healthy: `GET /api/health`.
2. Verify GPU detection: `GET /api/settings/gpu-info` returns at least one GPU.
3. Download a small test model: `POST /api/hf/download` with a small model (e.g., `Qwen/Qwen2.5-0.5B` or `TinyLlama/TinyLlama-1.1B-Chat-v1.0`).
4. Wait for download to complete (poll progress endpoint).
5. Verify model appears in registry: `GET /api/models`.
6. Configure model: `PUT /api/models/{id}/config` with basic settings.
7. Start vLLM: `POST /api/service/start`.
8. Wait for health check: poll `GET /api/service/health` until ready (up to startup_timeout).
9. Run inference: `POST /v1/chat/completions` with a simple prompt.
10. Verify response: check for non-empty `choices[0].message.content`.
11. Run a quick benchmark: `POST /api/benchmarks` with "quick" preset.
12. Wait for benchmark completion.
13. Verify benchmark results: `GET /api/benchmarks/{id}` has valid metrics.
14. Stop vLLM: `POST /api/service/stop`.
15. Verify stopped: `GET /api/service/status` shows stopped.
16. Cleanup: delete test model.

**Exit codes:**
- 0: all steps passed.
- 1: a step failed (prints which step and the response).

---

## 5. Documentation

### 5.1 README.md

Top-level `README.md` structure:

```
# vllm-toolchest

Containerized vLLM inference server with a web UI for AMD RDNA4 GPUs.

## Features
- Model management (download, configure, switch)
- vLLM process control with live logs
- Benchmarking with comparison
- Real-time GPU/CPU monitoring
- OpenAI-compatible API proxy with tool use / function calling
- Quantization support (AWQ, GPTQ, FP8, GGUF, BitsAndBytes, Marlin)

## Quick Start
  1. Clone the repo
  2. Run ./setup.sh
  3. Open http://localhost:3000

## Requirements
- Linux (Fedora 43 recommended)
- AMD RDNA4 GPU (RX 9700 XT) with 16-32GB VRAM
- Docker or Podman
- ROCm-compatible kernel driver

## Architecture
  [Brief description + diagram]

## API
  All endpoints are available at /v1/* (OpenAI-compatible) and /api/* (management).
  See docs/api.md for full reference.

## Configuration
  See .env.example for all environment variables.
  See docs/configuration.md for vllmctl.yaml reference.

## Development
  make dev   -- Run with hot reload
  make test  -- Run tests

## Screenshots
  [Screenshots of dashboard, model browser, server page, benchmarks]

## Credits
  Based on llama-toolchest (github.com/tmac1973/llama-toolchest)
  RDNA4 container approach inspired by kyuz0's vLLM RDNA4 work
```

### 5.2 .env.example

File: `.env.example` with every variable documented:

```bash
# =============================================================================
# vllm-toolchest Environment Configuration
# =============================================================================
# Copy to .env and customize. Generated automatically by setup.sh.

# --- GPU ---
GPU_TYPE=amd                          # amd | nvidia
GPU_COUNT=1                           # Number of GPUs
RENDER_GID=                           # Render group GID (auto-detected by setup.sh)
VIDEO_GID=                            # Video group GID (auto-detected by setup.sh)

# --- Container ---
COMPOSE_FILE=docker-compose.yml       # docker-compose.yml (ROCm) | docker-compose.cuda.yml (CUDA)

# --- Paths ---
MODEL_DIR=./models                    # Model storage directory (host path, bind-mounted)
DATA_DIR=./data                       # Persistent data (config, benchmarks)

# --- AMD ROCm ---
HSA_OVERRIDE_GFX_VERSION=12.0.1      # Override for RDNA4 gfx1201
PYTORCH_ROCM_ARCH=gfx1201            # PyTorch target architecture
VLLM_USE_TRITON_AWQ=1                # Use Triton AWQ kernels on ROCm (required)

# --- API ---
VLLMCTL_API_KEY=                      # API key for /v1/* endpoints (auto-generated by setup.sh)

# --- Ports ---
UI_PORT=3000                          # Web UI port
VLLM_PORT=8000                        # vLLM internal port (not exposed externally)

# --- HuggingFace ---
# VLLMCTL_HF_TOKEN=                  # HuggingFace token (or set via Settings UI)

# --- vLLM Defaults (override via Settings UI or env vars) ---
# VLLMCTL_VLLM_DEFAULTS_DTYPE=auto
# VLLMCTL_VLLM_DEFAULTS_GPU_MEMORY_UTILIZATION=0.90
# VLLMCTL_VLLM_DEFAULTS_ATTENTION_BACKEND=TRITON_FLASH_ATTN
# VLLMCTL_TOOL_USE_ENABLE_AUTO_TOOL_CHOICE=false
# VLLMCTL_TOOL_USE_TOOL_CALL_PARSER=
# VLLMCTL_QUANTIZATION_PREFER_MARLIN=true
```

### 5.3 API Documentation

File: `docs/api.md` -- Comprehensive endpoint reference.

**Structure:**
- For each endpoint: method, path, description, request parameters/body, response shape, example curl command.
- Organized by section: Health, Models, HuggingFace, Service, Benchmarks, Monitor, Settings, OpenAI Proxy.
- Note which endpoints require API key authentication.
- Note which endpoints support HTML (htmx) vs JSON response modes.
- Document SSE stream formats (event types, data shapes).

Alternatively, if we want machine-readable docs: an OpenAPI 3.0 spec YAML file at `docs/openapi.yaml` that can be served via Swagger UI. This is lower priority -- the markdown API docs are the primary reference. The OpenAPI spec can be generated later from the markdown or from Go handler annotations.

---

## 6. Edge Cases and Error Handling in Setup

### setup.sh Edge Cases

**No GPU detected:**
- Warn user: "No AMD or NVIDIA GPU detected. vLLM requires a GPU."
- Offer CPU-only mode (extremely slow, for testing only): set `VLLM_DEVICE=cpu` in `.env`.

**Mixed GPU vendors:**
- If both AMD and NVIDIA detected: ask user which to use. Do not try to use both.

**Docker rootless mode:**
- Detect rootless Docker: `docker info -f '{{.SecurityOptions}}'` contains `rootless`.
- GPU passthrough may not work in rootless mode -- warn user.

**Podman rootless:**
- CDI configuration requires root. Offer to run CDI setup with sudo.
- Alternative: `--device` flags in compose file instead of CDI.

**SELinux (Fedora):**
- If SELinux is enforcing: may need `:Z` suffix on volume mounts.
- Detect: `getenforce` returns "Enforcing".
- Auto-add `:Z` to bind mounts in compose file.

**Existing .env file:**
- If `.env` already exists: show diff of what would change, ask to overwrite or merge.
- Backup existing `.env` to `.env.backup.<timestamp>`.

**Port conflicts:**
- Check if port 3000 is already in use: `ss -tlnp | grep :3000`.
- If occupied: prompt user for alternative port.

**Insufficient disk space:**
- Check available space on MODEL_DIR partition.
- Warn if less than 50GB free (models can be large).

**Non-interactive mode:**
- Support `--yes` / `-y` flag to accept all defaults without prompting.
- Useful for CI/CD or scripted deployments.
- Also support `--gpu-type=amd`, `--runtime=docker`, `--model-dir=/path` flags for full non-interactive configuration.

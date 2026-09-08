.PHONY: build run dev docker docker-cuda docker-rocm docker-radiance docker-rebuild \
	up down up-cuda down-cuda up-rocm down-rocm up-radiance down-radiance \
	logs shell test reload clean

# ─── Variant + GPU auto-detection ───────────────────────────────────
# setup.sh records the chosen image variant in .env; honour it here so `make
# up` and `./setup.sh up` cannot disagree about which container they manage.
# Override on the command line: `make up VARIANT=radiance`.
VARIANT ?= $(shell sed -n 's/^VLLMCTL_VARIANT=//p' .env 2>/dev/null | head -1)

GPU_TYPE := $(shell \
	if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then \
		echo "cuda"; \
	elif [ -e /dev/kfd ]; then \
		echo "rocm"; \
	else \
		echo "cuda"; \
	fi)

# The image key mirrors setup.sh's image_key(): the radiance variant has its
# own compose file, everything else is keyed by GPU vendor.
IMAGE_KEY := $(GPU_TYPE)
ifeq ($(VARIANT),radiance)
	IMAGE_KEY := radiance
endif

COMPOSE_FILE := docker-compose.$(IMAGE_KEY).yml

# ─── Version ────────────────────────────────────────────────────────
# Stamped into the binary and shown under the sidebar brand, so a running
# instance can say which build it is. Falls back to "dev" outside a checkout.
VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
LDFLAGS := -X main.version=$(VERSION)

# ─── Local development ──────────────────────────────────────────────
build:
	go build -ldflags "$(LDFLAGS)" -o bin/vllmctl ./cmd/vllmctl

run: build
	./bin/vllmctl --config config.yaml

dev:
	go run -ldflags "$(LDFLAGS)" ./cmd/vllmctl --config config.yaml

# ─── Container (auto-detect GPU) ────────────────────────────────────
docker:
	@echo "GPU: $(GPU_TYPE)  variant: $(or $(VARIANT),generic) → $(COMPOSE_FILE)"
	docker compose -f $(COMPOSE_FILE) build

up:
	@echo "GPU: $(GPU_TYPE)  variant: $(or $(VARIANT),generic) → $(COMPOSE_FILE)"
	docker compose -f $(COMPOSE_FILE) up -d

down:
	docker compose -f $(COMPOSE_FILE) down

# ─── Container (explicit GPU target) ───────────────────────────────
docker-cuda:
	docker compose -f docker-compose.cuda.yml build

docker-rocm:
	docker compose -f docker-compose.rocm.yml build

docker-radiance:
	docker compose -f docker-compose.radiance.yml build

up-cuda:
	docker compose -f docker-compose.cuda.yml up -d

up-rocm:
	docker compose -f docker-compose.rocm.yml up -d

up-radiance:
	docker compose -f docker-compose.radiance.yml up -d

down-cuda:
	docker compose -f docker-compose.cuda.yml down

down-rocm:
	docker compose -f docker-compose.rocm.yml down

down-radiance:
	docker compose -f docker-compose.radiance.yml down

# ─── Common ────────────────────────────────────────────────────────
docker-rebuild:
	docker compose -f $(COMPOSE_FILE) down
	docker compose -f $(COMPOSE_FILE) build --no-cache
	docker compose -f $(COMPOSE_FILE) up -d

logs:
	docker compose -f $(COMPOSE_FILE) logs -f

shell:
	docker exec -it vllm-toolchest bash

test:
	go test ./...

# ─── Dev: rebuild Go binary and inject into running container ───────
# Compiles on host, copies into container, restarts the process.
# Much faster than rebuilding the entire image.
reload: build
	docker cp bin/vllmctl vllm-toolchest:/usr/local/bin/vllmctl
	docker restart vllm-toolchest
	@echo "Binary reloaded. UI at http://localhost:3000"

# ─── Cleanup ────────────────────────────────────────────────────────
clean:
	rm -rf bin/

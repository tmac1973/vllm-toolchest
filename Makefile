.PHONY: build run dev docker docker-rebuild up down \
	docker-amd docker-nvidia up-amd down-amd up-nvidia down-nvidia \
	logs shell test js-test reload clean env-example

# ─── Variant + vendor ───────────────────────────────────────────────
# setup.sh writes the chosen variant and its vendor to .env; read both from
# there so `make up` and `./setup.sh up` cannot disagree about which container
# they manage. This deliberately does not re-derive the vendor: that rule lives
# in setup.sh and the manifests, and a second copy here is what drifted last
# time. Override on the command line: `make up VENDOR=amd`.
VARIANT ?= $(shell sed -n 's/^VLLMCTL_VARIANT=//p' .env 2>/dev/null | head -1)
VENDOR  ?= $(shell sed -n 's/^VLLMCTL_VENDOR=//p' .env 2>/dev/null | head -1)

# Before the first install there is no .env. Fall back to a device probe so
# `make up` on a fresh checkout still does something sensible.
ifeq ($(VENDOR),)
VENDOR := $(shell \
	if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then \
		echo "nvidia"; \
	elif [ -e /dev/kfd ]; then \
		echo "amd"; \
	else \
		echo "nvidia"; \
	fi)
endif

COMPOSE_FILE := docker-compose.$(VENDOR).yml

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
	@echo "vendor: $(VENDOR)  variant: $(or $(VARIANT),generic) → $(COMPOSE_FILE)"
	docker compose -f $(COMPOSE_FILE) build

up:
	@echo "vendor: $(VENDOR)  variant: $(or $(VARIANT),generic) → $(COMPOSE_FILE)"
	docker compose -f $(COMPOSE_FILE) up -d

down:
	docker compose -f $(COMPOSE_FILE) down

# ─── Container (explicit vendor) ───────────────────────────────────
# One pair per vendor, not per variant: which variant is built depends on the
# VLLMCTL_* build inputs in .env, which setup.sh derives from the manifest.
docker-amd:
	docker compose -f docker-compose.amd.yml build

docker-nvidia:
	docker compose -f docker-compose.nvidia.yml build

up-amd:
	docker compose -f docker-compose.amd.yml up -d

up-nvidia:
	docker compose -f docker-compose.nvidia.yml up -d

down-amd:
	docker compose -f docker-compose.amd.yml down

down-nvidia:
	docker compose -f docker-compose.nvidia.yml down

# ─── Generated documentation ───────────────────────────────────────
# The feature switches are declared in variants/<id>.conf. Regenerate the
# block in .env.example from them rather than editing it by hand; a test
# fails if the two drift.
env-example:
	@./setup.sh --write-env-example .env.example

# ─── Common ────────────────────────────────────────────────────────
docker-rebuild:
	docker compose -f $(COMPOSE_FILE) down
	docker compose -f $(COMPOSE_FILE) build --no-cache
	docker compose -f $(COMPOSE_FILE) up -d

logs:
	docker compose -f $(COMPOSE_FILE) logs -f

shell:
	docker exec -it vllm-toolchest bash

test: js-test
	go test ./...

# The visualize page's chart logic runs against a node harness rather than a
# browser — the parts worth testing are decisions about the data, not drawing.
# Skipped where node is absent so `make test` still works on a bare box.
js-test:
	@command -v node >/dev/null 2>&1 || { echo "js-test: node not installed, skipping"; exit 0; }
	@node web/jstest/viz_test.js

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

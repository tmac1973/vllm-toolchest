.PHONY: build run dev docker docker-cuda docker-rebuild up down up-cuda down-cuda logs shell clean

# ─── GPU auto-detection ─────────────────────────────────────────────
GPU_TYPE := $(shell \
	if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then \
		echo "cuda"; \
	elif [ -e /dev/kfd ]; then \
		echo "rocm"; \
	else \
		echo "cuda"; \
	fi)

COMPOSE_FILE := docker-compose.yml
ifeq ($(GPU_TYPE),cuda)
	COMPOSE_FILE := docker-compose.cuda.yml
endif

# ─── Local development ──────────────────────────────────────────────
build:
	go build -o bin/vllmctl ./cmd/vllmctl

run: build
	./bin/vllmctl --config config.yaml

dev:
	go run ./cmd/vllmctl --config config.yaml

# ─── Container (auto-detect GPU) ────────────────────────────────────
docker:
	@echo "Detected GPU: $(GPU_TYPE) → $(COMPOSE_FILE)"
	docker compose -f $(COMPOSE_FILE) build

up:
	@echo "Detected GPU: $(GPU_TYPE) → $(COMPOSE_FILE)"
	docker compose -f $(COMPOSE_FILE) up -d

down:
	docker compose -f $(COMPOSE_FILE) down

# ─── Container (explicit GPU target) ───────────────────────────────
docker-cuda:
	docker compose -f docker-compose.cuda.yml build

docker-rocm:
	docker compose -f docker-compose.yml build

up-cuda:
	docker compose -f docker-compose.cuda.yml up -d

up-rocm:
	docker compose -f docker-compose.yml up -d

down-cuda:
	docker compose -f docker-compose.cuda.yml down

down-rocm:
	docker compose -f docker-compose.yml down

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

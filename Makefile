.PHONY: build run dev docker docker-rebuild up down logs shell clean

# ─── Local development ──────────────────────────────────────────────
build:
	go build -o bin/vllmctl ./cmd/vllmctl

run: build
	./bin/vllmctl --config config.yaml

dev:
	go run ./cmd/vllmctl --config config.yaml

# ─── Container ──────────────────────────────────────────────────────
docker:
	docker compose build

docker-rebuild:
	docker compose down
	docker compose build --no-cache
	docker compose up -d

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

shell:
	docker exec -it vllm-toolchest bash

# ─── Cleanup ────────────────────────────────────────────────────────
clean:
	rm -rf bin/

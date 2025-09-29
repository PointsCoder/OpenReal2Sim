# Makefile
name ?= openreal2sim

docker_build:
	# If it's an image-only service, this pulls it.
	docker compose -f docker/compose.yml pull $(name) || true
	# If it has a build context, this builds it (pull above is harmless).
	docker compose -f docker/compose.yml build $(name) || true

docker_run:
	# Runs the service; will pull if needed. Use run for one-off containers.
	docker compose -p "$$USER" -f docker/compose.yml run --rm $(name)

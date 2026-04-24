# Get version of CUDA and enable it for compilation if CUDA > 11.0
# This solves https://github.com/IDEA-Research/Grounded-Segment-Anything/issues/53
# and https://github.com/IDEA-Research/Grounded-Segment-Anything/issues/84
# when running in Docker
# Check if nvcc is installed
NVCC := $(shell which nvcc)
ifeq ($(NVCC),)
	# NVCC not found
	USE_CUDA := 0
	NVCC_VERSION := "not installed"
else
	NVCC_VERSION := $(shell nvcc --version | grep -oP 'release \K[0-9.]+')
	USE_CUDA := $(shell echo "$(NVCC_VERSION) > 11" | bc -l)
endif

# Add the list of supported ARCHs
ifeq ($(USE_CUDA), 1)
	TORCH_CUDA_ARCH_LIST := "3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
	BUILD_MESSAGE := "I will try to build the image with CUDA support"
else
	TORCH_CUDA_ARCH_LIST :=
	BUILD_MESSAGE := "CUDA $(NVCC_VERSION) is not supported"
endif


build-image:
	@echo $(BUILD_MESSAGE)
	docker build --build-arg USE_CUDA=$(USE_CUDA) \
	--build-arg TORCH_ARCH=$(TORCH_CUDA_ARCH_LIST) \
	-t gsa:v0 .
run:
ifeq (,$(wildcard ./sam_vit_h_4b8939.pth))
	wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
endif
ifeq (,$(wildcard ./groundingdino_swint_ogc.pth))
	wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
endif
	docker run --gpus all -it --rm --net=host --privileged \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-v "${PWD}":/home/appuser/Grounded-Segment-Anything \
	-e DISPLAY=$DISPLAY \
	--name=gsa \
	--ipc=host -it gsa:v0

# ============================================================================
# Test + lint targets — mirror CI exactly.
# Use `make ci-local` to run the same sequence CI runs, in the same order.
# ============================================================================

# pytest-xdist worker count. Pinned to 4 to match GitHub Actions free-tier
# vCPU count AND avoid "Different tests collected" errors observed at higher
# worker counts on dev machines. Override with `make test PYTEST_WORKERS=8`.
PYTEST_WORKERS ?= 4

# Common pytest flags: skip GPU + slow by default, fixed worker count,
# loadscope distribution groups tests by module for deterministic collection.
PYTEST_FAST = -m "not gpu and not slow" -n $(PYTEST_WORKERS) --dist=loadscope
PYTEST_SLOW = -m "slow and not gpu" -n $(PYTEST_WORKERS) --dist=loadscope

# Coverage scope: runtime code only. cli/ and experiment/ are high-churn
# entry-point / research scripts, excluded from coverage targets.
COV_FLAGS = --cov=core --cov=ml_engine --cov=augmentation --cov=api

.PHONY: test test-slow lint coverage ci-local

test:
	uv run pytest tests/unit tests/integration tests/contract $(PYTEST_FAST)

test-slow:
	uv run pytest tests $(PYTEST_SLOW)

lint:
	@# Ruff + mypy are both on a ramp-up period. The project has a large
	@# pre-existing lint baseline (3593 ruff findings at PR time, mostly
	@# trivial: trailing whitespace, unsorted imports, long lines). Rather
	@# than block shipping the CI pipeline on cleanup of unrelated files,
	@# both tools log their findings but do not fail the lint job. Cleanup
	@# is tracked in TODOS.md.
	@# When ruff/mypy baselines reach zero, drop the trailing `|| echo ...`
	@# to make these gating.
	uv run ruff check . || echo "ruff: continuing despite findings (ramp-up period)"
	uv run ruff format --check . || echo "ruff format: continuing despite findings (ramp-up period)"
	uv run mypy core ml_engine api --ignore-missing-imports || \
		echo "mypy: continuing despite findings (ramp-up period)"

coverage:
	uv run pytest tests/unit tests/integration tests/contract $(PYTEST_FAST) \
		$(COV_FLAGS) --cov-report=term-missing --cov-report=html:htmlcov
	@echo "Coverage HTML report: htmlcov/index.html"

# ci-local runs the EXACT commands CI runs, in the EXACT order.
# If this passes, PR CI will pass. If this fails, fix before pushing.
#
# The first step syncs both `lint` and `test` extras so ruff/mypy/pytest are
# all present. CI itself installs these per job, so this mirrors the
# aggregate install the developer's workspace needs.
ci-local:
	@echo "=== install (test + lint + cpu extras) ==="
	@# --extra cpu selects CPU torch so `make ci-local` behaves like PR CI.
	@# Developers working on GPU code should `uv sync --extra gpu --extra test`
	@# separately for their primary venv.
	uv sync --frozen --extra test --extra lint --extra cpu
	@echo "=== lint ==="
	$(MAKE) lint
	@echo "=== unit ==="
	uv run pytest tests/unit $(PYTEST_FAST) $(COV_FLAGS) --cov-report= --cov-context=test
	mv .coverage .coverage.unit
	@echo "=== contract + integration ==="
	uv run pytest tests/integration tests/contract $(PYTEST_FAST) $(COV_FLAGS) --cov-report=
	mv .coverage .coverage.contract
	@echo "=== coverage gate ==="
	uv run coverage combine .coverage.unit .coverage.contract
	uv run coverage report --show-missing
	@rm -f .coverage.unit .coverage.contract

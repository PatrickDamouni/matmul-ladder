# matmul-ladder -- builds on ARM64 macOS (Apple clang + Homebrew libomp)
# and ARM64 Linux (GCC/Clang with their bundled OpenMP).
#   macOS: Xcode command-line tools + `brew install libomp`
#   Linux: build-essential (bench needs Apple Accelerate, so it is macOS-only)

UNAME := $(shell uname -s)
WARN  := -Wall -Wextra -Werror

ifeq ($(UNAME),Darwin)
  CXX      := clang++
  LIBOMP   := $(shell brew --prefix libomp 2>/dev/null)
  OMPFLAGS := -Xpreprocessor -fopenmp -I$(LIBOMP)/include
  LDLIBS   := -L$(LIBOMP)/lib -lomp
else
  CXX      ?= g++
  OMPFLAGS := -fopenmp
  LDLIBS   := -fopenmp
endif

CXXFLAGS := -std=c++17 -O3 $(WARN) -Iinclude $(OMPFLAGS)

SRC := $(wildcard src/*.cpp)
HDR := $(wildcard include/*.h src/*.h)

.PHONY: all test bench bench-quick sweep asan torch-setup torch-test torch-bench clean

ifeq ($(UNAME),Darwin)
all: build/test build/bench
else
all: build/test
endif

build/test: $(SRC) tests/test_matmul.cpp $(HDR) | build
	$(CXX) $(CXXFLAGS) $(SRC) tests/test_matmul.cpp -o $@ $(LDLIBS)

build/bench: $(SRC) bench/bench.cpp $(HDR) | build
ifneq ($(UNAME),Darwin)
	$(error bench links Apple Accelerate and only builds on macOS)
endif
	$(CXX) $(CXXFLAGS) $(SRC) bench/bench.cpp -o $@ $(LDLIBS) -framework Accelerate

build/tile_sweep: $(SRC) bench/tile_sweep.cpp $(HDR) | build
	$(CXX) $(CXXFLAGS) $(SRC) bench/tile_sweep.cpp -o $@ $(LDLIBS)

# Same tests under AddressSanitizer + UndefinedBehaviorSanitizer.
build/test-asan: $(SRC) tests/test_matmul.cpp $(HDR) | build
	$(CXX) -std=c++17 -O1 -g $(WARN) -Iinclude $(OMPFLAGS) -fsanitize=address,undefined \
		-fno-sanitize-recover=all -fno-omit-frame-pointer \
		$(SRC) tests/test_matmul.cpp -o $@ $(LDLIBS)

test: build/test
	./build/test

asan: build/test-asan
	./build/test-asan

bench: build/bench | results
	./build/bench

bench-quick: build/bench | results
	./build/bench --quick

sweep: build/tile_sweep
	./build/tile_sweep

# PyTorch custom-op bridge (macOS arm64). Uses a local .venv; ninja must be on
# PATH for torch's JIT extension build, hence the absolute PATH prefix.
VENV  := $(CURDIR)/.venv
TORCH := cd torch_ext && PATH=$(VENV)/bin:$$PATH $(VENV)/bin/python

torch-setup:
	uv venv -q --python 3.14 $(VENV)
	uv pip install -q --python $(VENV)/bin/python -r torch_ext/requirements.txt

torch-test:
	$(TORCH) test_op.py

torch-bench:
	$(TORCH) bench_op.py

build results:
	mkdir -p $@

clean:
	rm -rf build torch_ext/__pycache__

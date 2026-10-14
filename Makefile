# matmul-ladder -- Apple Silicon (macOS) build.
# Needs: Xcode command-line tools + `brew install libomp`.

CXX      := clang++
LIBOMP   := $(shell brew --prefix libomp 2>/dev/null)
WARN     := -Wall -Wextra -Werror
OMPFLAGS := -Xpreprocessor -fopenmp -I$(LIBOMP)/include
CXXFLAGS := -std=c++17 -O3 $(WARN) -Iinclude $(OMPFLAGS)
LDLIBS   := -L$(LIBOMP)/lib -lomp

SRC := $(wildcard src/*.cpp)
HDR := $(wildcard include/*.h src/*.h)

.PHONY: all test bench bench-quick asan clean

all: build/test build/bench

build/test: $(SRC) tests/test_matmul.cpp $(HDR) | build
	$(CXX) $(CXXFLAGS) $(SRC) tests/test_matmul.cpp -o $@ $(LDLIBS)

build/bench: $(SRC) bench/bench.cpp $(HDR) | build
	$(CXX) $(CXXFLAGS) $(SRC) bench/bench.cpp -o $@ $(LDLIBS) -framework Accelerate

# Same tests under AddressSanitizer + UndefinedBehaviorSanitizer.
build/test-asan: $(SRC) tests/test_matmul.cpp $(HDR) | build
	$(CXX) -std=c++17 -O1 -g $(WARN) -Iinclude $(OMPFLAGS) -fsanitize=address,undefined \
		-fno-omit-frame-pointer $(SRC) tests/test_matmul.cpp -o $@ $(LDLIBS)

test: build/test
	./build/test

asan: build/test-asan
	./build/test-asan

bench: build/bench | results
	./build/bench

bench-quick: build/bench | results
	./build/bench --quick

build results:
	mkdir -p $@

clean:
	rm -rf build

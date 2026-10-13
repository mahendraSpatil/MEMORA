# Memora - context-aware memory eviction
#
#   make            build bin/memora and bin/memora_tests
#   make test       build and run the validation suite
#   make demo       one configuration, table on stdout
#   make results    every sweep to results/*.csv
#   make figures    charts from the CSVs (needs python3 + matplotlib)
#   make clean

CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Wpedantic -Iinclude
LDFLAGS  ?=

SRC      := $(filter-out src/main.cpp,$(wildcard src/*.cpp))
OBJ      := $(patsubst src/%.cpp,build/%.o,$(SRC))
MAIN_OBJ := build/main.o
TEST_OBJ := build/test_main.o

.PHONY: all test demo results figures clean

all: bin/memora bin/memora_tests

build/%.o: src/%.cpp
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -c $< -o $@

build/test_main.o: tests/test_main.cpp
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -c $< -o $@

bin/memora: $(OBJ) $(MAIN_OBJ)
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

bin/memora_tests: $(OBJ) $(TEST_OBJ)
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

test: bin/memora_tests
	./bin/memora_tests

demo: bin/memora
	./bin/memora --demo

results: bin/memora
	./bin/memora --all

figures:
	python3 tools/plot.py

clean:
	rm -rf build bin

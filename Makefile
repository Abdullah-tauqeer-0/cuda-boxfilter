# Directories
SRC_DIR := src
BIN_DIR := bin
INC_DIR := include

CPU_SRC := $(SRC_DIR)/box_filter_cpu.c
GPU_SRC := $(SRC_DIR)/box_filter_gpu.cu
UTILS   := $(SRC_DIR)/utils.c      # <-- NEW


CPU_BIN := $(BIN_DIR)/box_filter_cpu
GPU_BIN := $(BIN_DIR)/box_filter_gpu

CFLAGS  := -I$(INC_DIR) -I$(SRC_DIR) -O2 -std=c11
NVFLAGS := -I$(INC_DIR) -I$(SRC_DIR) -O2

.PHONY: all cpu gpu clean

all: cpu gpu

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

cpu: | $(BIN_DIR)
	gcc $(CPU_SRC) $(UTILS) -o $(CPU_BIN) $(CFLAGS) -lm
	@echo "Built $(CPU_BIN)"
gpu: | $(BIN_DIR)
	nvcc $(GPU_SRC) $(UTILS) -o $(GPU_BIN) $(NVFLAGS)
	@echo "Built $(GPU_BIN)"

clean:
	rm -rf $(BIN_DIR)

# -----------------------------------------------------------------
# Convenience target so "make build" builds *both* binaries
# -----------------------------------------------------------------
.PHONY: build
build: cpu gpu              # or just cpu    (pick what you want)

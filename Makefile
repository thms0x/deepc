# --- Setup ---
CC = gcc
CFLAGS = -O3 -Wall -Wextra -std=c11 -I./include
LDFLAGS = -lm

# --- Directories ---
SRC_DIR = src
INC_DIR = include
BUILD_DIR = build

# --- Files ---
TARGET = mnist_trainer.out

SRCS = $(wildcard $(SRC_DIR)/*.c)

OBJS = $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o, $(SRCS))

# --- Rules ---
all: $(BUILD_DIR) $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

.PHONY: all clean

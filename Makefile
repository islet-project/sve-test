CROSS ?= aarch64-linux-gnu-
CC := $(CROSS)gcc

CFLAGS := -O2 -g -Wall -Wextra -march=armv8.2-a+sve -static
LDFLAGS := -lpthread

TARGET := sve-test

all: $(TARGET)

$(TARGET): main.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean


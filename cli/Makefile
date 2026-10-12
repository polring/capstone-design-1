CC ?= gcc
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Werror
LDLIBS = -lm
BUILD_DIR = build

ifeq ($(OS),Windows_NT)
	EXE = .exe
else
	EXE =
endif

LOADER = $(BUILD_DIR)/loader$(EXE)
FIXTURE_TEST = $(BUILD_DIR)/fixture_test$(EXE)
# Every contract version the engine supports keeps its fixtures here.
FIXTURE_DIRS = $(wildcard fixtures/v*)

.PHONY: all engine postprocessor test test-fixtures test-postprocessor clean

all: engine postprocessor

engine: $(LOADER)

$(LOADER): engine/loader.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(LDLIBS) -o $@

# The test includes loader.c, so the engine's CLI handlers are unused there.
$(FIXTURE_TEST): tests/fixture_test.c engine/loader.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -Wno-unused-function $< $(LDLIBS) -o $@

$(BUILD_DIR):
	mkdir $(BUILD_DIR)

postprocessor:
	$(MAKE) -C postprocessor all

test: test-fixtures test-postprocessor

test-fixtures: $(FIXTURE_TEST)
	$(foreach dir,$(FIXTURE_DIRS),./$(FIXTURE_TEST) $(dir)/model.gguf $(dir)/cases.txt &&) true

test-postprocessor:
	$(MAKE) -C postprocessor test

clean:
	$(MAKE) -C postprocessor clean
	rm -rf $(BUILD_DIR)

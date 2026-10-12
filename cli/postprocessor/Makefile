CC ?= gcc
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Werror -I.
LDFLAGS ?= -lm

# Detect if sqlite3 is available
SQLITE_AVAILABLE := $(shell $(CC) -x c -E -include sqlite3.h /dev/null -o /dev/null 2>/dev/null && echo 1 || echo 0)
ifeq ($(SQLITE_AVAILABLE), 1)
	CFLAGS += -DHAVE_SQLITE3
	LDFLAGS += -lsqlite3
endif

SRCS = parser.c replacement.c schema.c core.c
OBJS = $(SRCS:.c=.o)
LIB_STATIC = libpostprocessor.a
TARGET_CLI = postprocessor
TEST_BIN = test_postprocessor

ifeq ($(OS),Windows_NT)
	TARGET_CLI := postprocessor.exe
	TEST_BIN := test_postprocessor.exe
	RM = del /Q /F
	NULL_DEVICE = nul
else
	RM = rm -f
	NULL_DEVICE = /dev/null
endif

.PHONY: all clean test

all: $(LIB_STATIC) $(TARGET_CLI)

$(LIB_STATIC): $(OBJS)
	ar rcs $@ $(OBJS)

$(TARGET_CLI): main.c $(LIB_STATIC)
	$(CC) $(CFLAGS) -o $@ main.c $(LIB_STATIC) $(LDFLAGS)

test: $(TEST_BIN)
	./$(TEST_BIN)

$(TEST_BIN): test_postprocessor.c $(LIB_STATIC)
	$(CC) $(CFLAGS) -o $@ test_postprocessor.c $(LIB_STATIC) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	$(RM) $(OBJS) $(LIB_STATIC) $(TARGET_CLI) $(TEST_BIN) 2>$(NULL_DEVICE) || true

CC      = gcc
CFLAGS  = -Wall -Wextra -std=c99 -g
LDFLAGS = -lm

.PHONY: all clean

all: compare

compare: compare.c
	$(CC) $(CFLAGS) -o compare compare.c $(LDFLAGS)

clean:
	rm -f compare

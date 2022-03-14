CC = g++ # clang++
WARNIGNS = -Wall -Wextra -Wshadow -Wfatal-errors -Wpedantic -Wno-uninitialized # -Wconversion
FLAGS = $(WARNIGNS) -std=c++20 # -O2 # -fsanitize=address

all:
	$(CC) $(FLAGS) main.cpp 

run: all
	./a.out
CC = nvcc
FLAGS = -lcufft 
# Alternatively for staticly linked cufft library '-lcufft_static -lculibos'
all:
	$(CC) $(FLAGS) src/main.cu

test:
	$(CC) $(FLAGS) src/cufft_test.cu

run: all
	./a.out
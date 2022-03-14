CC = nvcc
FLAGS = -lcufft # -I/usr/local/cuda/inc -L/usr/local/cuda/lib
# Alternatively for staticly linked cufft library '-lcufft_static -lculibos'
all:
	$(CC) $(FLAGS) cufft_test.cu

run: all
	./a.out
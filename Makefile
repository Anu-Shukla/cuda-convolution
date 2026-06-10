NVCC ?= nvcc

.PHONY: all clean

all: naive tiled constant cudnn_ref

naive: naive.cu
	$(NVCC) naive.cu -o naive -O3 --use_fast_math

tiled: tiled.cu
	$(NVCC) tiled.cu -o tiled -O3 --use_fast_math

constant: constant.cu
	$(NVCC) constant.cu -o constant -O3 --use_fast_math

cudnn_ref: cudnn_ref.cu
	$(NVCC) cudnn_ref.cu -o cudnn_ref -O3 -lcudnn

clean:
	rm -f naive tiled constant cudnn_ref

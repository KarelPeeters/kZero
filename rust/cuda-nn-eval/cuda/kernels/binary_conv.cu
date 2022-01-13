#include <assert.h>
#include <cuda_runtime.h>
#include <stdint.h>

typedef uint64_t u64;

template<typename T, int C, int H, int W>
__global__ void binaryConv3x3Kernel(
        const u64 *input, const u64 *filter, u64 *output
) {
    const int FILTER_SIZE = 3 * 3 * C * C;

    __shared__ u64 input_shared[H * W * C];
    __shared__ u64 filter_shared[FILTER_SIZE];

    int b = blockIdx.x;
    int h = threadIdx.x / W;
    int w = threadIdx.x % W;

    //TODO unroll loops
    //TODO memcpy?
    for (int c = 0; c < C; c++) {
        input_shared[h * W * C + w * C + c] = input[b * H * W * C + h * W * C + w * C + c];
    }

    if (blockDim.x > FILTER_SIZE) {
        if (threadIdx.x < FILTER_SIZE) {
            filter_shared[threadIdx.x] = filter[threadIdx.x];
        }
    } else {
        for (int i = 0; i < (FILTER_SIZE + blockDim.x - 1) / blockDim.x; i++) {
            //TODO continue here
            filter_shared[];
        }

        assert(false);
    }

    __syncthreads();

    for (int c = 0; c < C; c++) {
        u64 result = 0;
        for (int ci = 0; ci < 64; ci++) {
            u64 result_i = 0;
            for (int fh = 0; fh < 3; fh++) {
                for (int fw = 0; fw < 3; fw++) {

                }
            }
            result |= (result_i >= C * 32) << ci;
        }
        output[b * H * W * C + h * W * C + w * C + c] = result;
    }
}

extern "C" {
cudaError binaryConv(
        cudaStream_t stream,
        int batch_size, int cc, int w, int h,
        const u64 *input, u64 *output
) {
    int blockSize = w * h;
    int blocks = batch_size;

    binaryConvKernel < float ><<<blocks, blockSize, 0, stream>>>(batch_size, cc, w, h, input, output);

    return cudaGetLastError();
}
}

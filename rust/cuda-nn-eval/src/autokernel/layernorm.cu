#include "util.cu"
#include "welford.cu"

// de-dollar-ify template parameters
#define CACHE $CACHE$

const int THREADS_PER_BLOCK = $THREADS_PER_BLOCK$;

const int RANK = $RANK$;
const int STATIC_SIZE = $STATIC_SIZE$;
const int NORM_SIZE = $NORM_SIZE$;

const float EPS = $EPS$;
const float ALPHA_0 = $ALPHA_0$;
const float ALPHA_1 = $ALPHA_1$;
const float BETA = $BETA$;

// *CAREFUL* these arrays are actually of length RANK-1, but zero-sized arrays are not allowed in C++ so we pad them
const int STATIC_DENSE_STRIDES[RANK] = $STATIC_DENSE_STRIDES$;
const int STATIC_STRIDES[2][RANK] = $STATIC_STRIDES$;

const int NORM_STRIDES[2] = $NORM_STRIDES$;

__device__ float calculate_x(float *input0, float *input1, int offset_x) {
    float x = ALPHA_0 * input0[offset_x];
    if (ALPHA_1 != 0.0) {
        x += ALPHA_1 * input1[offset_x];
    }
    return x;
}

__device__ Welford wf_warp_reduce(Welford curr) {
    // TODO see if unrolling this is useful
    for (int offset = 16; offset > 0; offset /= 2) {
        Welford next = Welford(
                __shfl_down_sync(FULL_WARP_MASK, curr.count, offset),
                __shfl_down_sync(FULL_WARP_MASK, curr.mean, offset),
                __shfl_down_sync(FULL_WARP_MASK, curr.m2, offset)
        );
        curr = curr.combine(next);
    }
    return curr;
}

// TODO only do second reduction if there are actually multiple warps
// TODO   more generally support different warp counts

__global__ void layernorm_kernel(
        float *input0,
        float *input1,
        float *output
) {
    KernelInfo info = kernel_info();
    assert(info.threads_per_block == THREADS_PER_BLOCK);

    int static_index = info.block_id;
    assert(static_index < STATIC_SIZE);

    Array<int, 2> static_offsets = flat_index_to_offsets<RANK, 2>(static_index, STATIC_DENSE_STRIDES, STATIC_STRIDES);

    Welford wf_thread = Welford();

#if CACHE
    __shared__ float cache[NORM_SIZE];
#endif

    // calculate variance and mean per thread
    for (int i = info.thread_id; i < NORM_SIZE; i += THREADS_PER_BLOCK) {
        int offset_x = static_offsets[0] + i * NORM_STRIDES[0];
        float x = calculate_x(input0, input1, offset_x);
#if CACHE
        cache[i] = x;
#endif
        wf_thread.append(x);
    }

    // reduce across warp
    static_assert(THREADS_PER_BLOCK % 32 == 0, "THREADS_PER_BLOCK must be a multiple of 32");
    Welford wf_warp = wf_warp_reduce(wf_thread);

    // reduce across blocks
    const int WARPS_PER_BLOCK = THREADS_PER_BLOCK / 32;
    static_assert(WARPS_PER_BLOCK == 32, "WARPS_PER_BLOCK must be 32 (for now)");

    __shared__ char wf_buffer_bytes[WARPS_PER_BLOCK * sizeof(Welford)];
    Welford *wf_buffer = (Welford *) wf_buffer_bytes;

    int lane = info.lane_id;
    int warp = info.warp_id;

    if (lane == 0) {
        wf_buffer[warp] = wf_warp;
    }
    __syncthreads();

    // let first warp do the actual reduction
    if (warp == 0) {
        Welford wf_total = wf_warp_reduce(wf_buffer[lane]);

        if (lane == 0) {
            wf_buffer[0] = wf_total;
        }
    }
    __syncthreads();

    // broadcast result to all threads
    Welford wf_total = wf_buffer[0];
    float mean = wf_total.final_mean();
    float variance = wf_total.final_variance();

    // actually normalize and write to output
    float denom = sqrt(variance + EPS);

    for (int i = info.thread_id; i < NORM_SIZE; i += THREADS_PER_BLOCK) {
        int offset_x = static_offsets[0] + i * NORM_STRIDES[0];
        int offset_y = static_offsets[1] + i * NORM_STRIDES[1];

#if CACHE
        float x = cache[i];
#else
        float x = calculate_x(input0, input1, offset_x);
#endif

        float y = (x - mean) / denom;
        output[offset_y] = BETA * y;
    }
}
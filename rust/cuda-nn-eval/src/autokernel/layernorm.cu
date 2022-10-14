#include "util.cu"
#include "welford.cu"

// de-dollar-ify template parameters
#define CACHE $CACHE$

const int THREADS_PER_BLOCK = $THREADS_PER_BLOCK$;
static_assert(is_power_of_two(THREADS_PER_BLOCK),
"THREADS_PER_BLOCK must be a power of two");

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

// TODO add caching again for small enough sizes (and make sure it works for 64-bit addresses)
// Every block handles a single layernorm group.
// Uses Welford's algorithm to compute the mean and variance
//   (see https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford%27s_online_algorithm).
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
    for (int i = info.lane_id; i < NORM_SIZE; i += info.threads_per_block) {
        int offset_x = static_offsets[0] + i * NORM_STRIDES[0];
        float x = calculate_x(input0, input1, offset_x);
#if CACHE
        cache[i] = x;
#endif
        wf_thread.append(x);
    }

    // reduce variance and mean across the block
    int tid = info.thread_id;

    __shared__ char wf_reduction_bytes[THREADS_PER_BLOCK * sizeof(Welford)];
    Welford *wf_reduction = (Welford *) wf_reduction_bytes;

    wf_reduction[tid] = wf_thread;
    __syncthreads();

    assert(is_power_of_two(info.threads_per_block));
    for (int s = info.threads_per_block >> 1; s > 0; s >>= 1) {
        Welford wf_new;
        if (tid < s) {
            wf_new = wf_reduction[tid].combine(wf_reduction[tid + s]);
        }
        __syncthreads();
        if (tid < s) {
            wf_reduction[tid] = wf_new;
        }
        __syncthreads();
    }

    Welford wf_total = wf_reduction[0];
    float mean = wf_total.final_mean();
    float variance = wf_total.final_variance();

    // actually normalize and write to output
    float denom = sqrt(variance + EPS);

    for (int i = info.lane_id; i < NORM_SIZE; i += 32) {
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
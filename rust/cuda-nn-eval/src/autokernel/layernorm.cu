#include "util.cu"
#include "welford.cu"

// de-dollar-ify template parameters
#define CACHE $CACHE$
#define REDUCE_THEAD $REDUCE_THREAD$

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

    int static_index = info.global_warp_id;
    if (static_index >= STATIC_SIZE) {
        return;
    }

    Array<int, 2> static_offsets = flat_index_to_offsets<RANK, 2>(static_index, STATIC_DENSE_STRIDES, STATIC_STRIDES);

    Welford wf_thread = Welford();

#if CACHE
    __shared__ float cache[NORM_SIZE];
#endif

    // calculate variance and mean per thread
    for (int i = info.lane_id; i < NORM_SIZE; i += 32) {
        int offset_x = static_offsets[0] + i * NORM_STRIDES[0];
        float x = calculate_x(input0, input1, offset_x);

#if CACHE
        cache[i] = x;
#endif
        wf_thread.append(x);
    }

    // reduce variance and mean across the warp
#if REDUCE_THEAD
    // reduce using method that would generalize when we don't stay within a single warp
    int wid = info.warp_id;
    int tid = info.lane_id;

    const int MAX_WARP_COUNT = 4;
    assert(wid < MAX_WARP_COUNT);

    __shared__ char wf_reduction_bytes[MAX_WARP_COUNT][32 * sizeof(Welford)];
    Welford (*wf_reduction)[32] = (Welford (*)[32]) wf_reduction_bytes;

    wf_reduction[wid][tid] = wf_thread;
    __syncthreads();

    for (int s = 16; s > 0; s >>= 1) {
        Welford wf_new;
        if (tid < s) {
            wf_new = wf_reduction[wid][tid].combine(wf_reduction[wid][tid + s]);
        }
        __syncthreads();
        if (tid < s) {
            wf_reduction[wid][tid] = wf_new;
        }
        __syncthreads();
    }

    Welford wf_total = wf_reduction[wid][0];
#else
    // reduce using warp based primitives
    Welford wf_curr = wf_thread;

    for (int offset = 16; offset > 0; offset /= 2) {
        Welford wf_next = Welford(
                __shfl_down_sync(FULL_WARP_MASK, wf_curr.count, offset),
                __shfl_down_sync(FULL_WARP_MASK, wf_curr.mean, offset),
                __shfl_down_sync(FULL_WARP_MASK, wf_curr.m2, offset)
        );

        wf_curr = wf_curr.combine(wf_next);
    }

    // broadcast to all threads
    Welford wf_total = Welford(
            __shfl_sync(FULL_WARP_MASK, wf_curr.count, 0),
            __shfl_sync(FULL_WARP_MASK, wf_curr.mean, 0),
            __shfl_sync(FULL_WARP_MASK, wf_curr.m2, 0)
    );
#endif

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
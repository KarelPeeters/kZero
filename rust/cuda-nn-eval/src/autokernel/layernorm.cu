#include "util.cu"
#include "welford.cu"

// de-dollar-ify template parameters
#define CACHE $CACHE$
#define SINGLE_WARP_PER_BLOCK $SINGLE_WARP_PER_BLOCK$

const int THREADS_PER_BLOCK = $THREADS_PER_BLOCK$;
const int WARPS_PER_BLOCK = THREADS_PER_BLOCK / 32;

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

// Reduce into warp lane zero from the entire warp.
__device__ Welford wf_warp_reduce(Welford curr) {
    // TODO see if unrolling this is useful
    // TODO skip loop iterations if there are fewer warps
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

// Broadcast from warp lane zero to the entire warp.
__device__ Welford wf_broadcast(Welford wf) {
    return Welford(
            warp_broadcast(wf.count),
            warp_broadcast(wf.mean),
            warp_broadcast(wf.m2)
    );
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
    static_assert(WARPS_PER_BLOCK <= 32, "There cannot be more than 32 warps");
    static_assert(WARPS_PER_BLOCK >= 1, "There must be at least one full warp");

    Welford wf_total;
    static_assert(SINGLE_WARP_PER_BLOCK == (WARPS_PER_BLOCK == 1), "Single warp variables must match");

#if SINGLE_WARP_PER_BLOCK
    // TODO remove this special case once we're skipping loop iterations depending on warp count
    // there is only a single warp, we don't actually need to reduce further
    // just broadcast to all threads of the single warp
    wf_total = wf_broadcast(wf_warp);
#else
    int lane = info.lane_id;
    int warp = info.warp_id;

    __shared__ char wf_buffer_bytes[WARPS_PER_BLOCK * sizeof(Welford)];
    Welford *wf_buffer = (Welford *) wf_buffer_bytes;

    if (lane == 0) {
        assert(warp < WARPS_PER_BLOCK);
        wf_buffer[warp] = wf_warp;
    }
    __syncthreads();

    // do the actual reduction on the first warp
    if (warp == 0) {
        // pad with additional values if the buffer is not full-sized
        Welford wf_tmp = Welford();
        if (lane < WARPS_PER_BLOCK) {
            wf_tmp = wf_buffer[lane];
        }

        Welford wf_total = wf_warp_reduce(wf_tmp);

        if (lane == 0) {
            wf_buffer[0] = wf_total;
        }
    }
    __syncthreads();

    // broadcast result to all threads
    wf_total = wf_buffer[0];
#endif

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
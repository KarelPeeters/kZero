// Implementation based on FlashAttention
//   paper: https://arxiv.org/abs/2205.14135
//   repository: https://github.com/HazyResearch/flash-attention

#include "util.cu"

// TODO proper stride support
// TODO properly implement reduce operations
// TODO mask and scale
// TODO more general bank conflict fix, this is basically hardcoded to D being a multiple of 32

// de-dollar-ify template parameters

// sequence length
const int S_QO = $S_QO$;
const int S_KV = $S_KV$;

// data size
const int D_QK = $D_QK$;
const int D_VO = $D_VO$;

// block size
const int G_QO = $G_QO$;
const int B_QO = $B_QO$;
const int B_KV = $B_KV$;

// TODO relax this requirement?
static_assert(true && (S_QO % G_QO == 0) && (G_QO % B_QO == 0), "S_QO blocks must divide it");
static_assert(true && (S_KV % B_KV == 0), "S_KV block size must divide it");
static_assert(true && (D_VO % B_KV == 0), "B_KV must divide D_VO");

const int GC_QO = ceil_div(S_QO, G_QO);
const int BC_QO = ceil_div(G_QO, B_QO);
const int BC_KV = ceil_div(S_KV, B_KV);

const int SCRATCH_SIZE = $SCRATCH_SIZE$;
static_assert(true && SCRATCH_SIZE == 2 * S_QO, "Scratch size mismatch");

// this is not a top-level function because the IDE can't deal with the __launch_bounds__ it would require
__device__ void attention_kernel_inner(
        float *global_q, float *global_k, float *global_v,
        float *global_o, float *scratch
) {
    KernelInfo info = kernel_info();
    float *global_max = scratch;
    float *global_sum = scratch + S_QO;

    assert(info.block_count == GC_QO);
    assert(info.threads_per_block == B_QO * B_KV);
    int thread_qo_i = info.thread_id / B_KV;
    int thread_kv_j = info.thread_id % B_KV;
    bool is_first_q_thread = thread_kv_j == 0;

    // offset operands
    int g_qo_offset = info.block_id * G_QO;
    global_q += g_qo_offset * D_QK;
    global_o += g_qo_offset * D_VO;
    global_max += g_qo_offset;
    global_sum += g_qo_offset;

    // zero-initialize output and scratch
    for (int i = info.thread_id; i < G_QO * D_VO; i += info.threads_per_block) {
        global_o[i] = 0.0;
    }
    for (int i = info.thread_id; i < G_QO; i += info.threads_per_block) {
        global_max[i] = -1.0 / 0.0;
        global_sum[i] = 0.0;
    }

    __syncthreads();

    // local memory
    __shared__ float block_q[B_QO * D_QK];
    __shared__ float block_k[B_KV * (D_QK + 1)];
    __shared__ float block_v[B_KV * D_VO];
    __shared__ float block_o[B_QO * D_VO];
    // TODO try fusing old/new to reduce shared mem usage
    __shared__ float block_max_old[B_QO];
    __shared__ float block_max_new[B_QO];
    __shared__ float block_sum_old[B_QO];
    __shared__ float block_sum_new[B_QO];
    __shared__ float block_logits[B_QO][B_KV];

    // main processing
    for (int block_kv_j = 0; block_kv_j < BC_KV; block_kv_j++) {
        /* Load inputs */
        // load k, v
        // TODO merge loops if D_QK == D_VO?
        for (int i = info.thread_id; i < B_KV * D_QK; i += info.threads_per_block) {
            int offset = block_kv_j * B_KV * D_QK;
            int pad_i = (i / D_QK) * (D_QK + 1) + (i % D_QK);
            block_k[pad_i] = global_k[offset + i];
        }
        for (int i = info.thread_id; i < B_KV * D_VO; i += info.threads_per_block) {
            int offset = block_kv_j * B_KV * D_VO;
            block_v[i] = global_v[offset + i];
        }
        __syncthreads();

        for (int block_qo_i = 0; block_qo_i < BC_QO; block_qo_i++) {
            // load q, o
            // TODO merge loops if D_QK == D_VO?
            for (int i = info.thread_id; i < B_QO * D_QK; i += info.threads_per_block) {
                int offset = block_qo_i * B_QO * D_QK;
                block_q[i] = global_q[offset + i];
            }
            for (int i = info.thread_id; i < B_QO * D_VO; i += info.threads_per_block) {
                int offset = block_qo_i * B_QO * D_VO;
                block_o[i] = global_o[offset + i];
            }

            // load max, sum
            if (is_first_q_thread) {
                int offset = block_qo_i * B_QO;
                block_max_old[thread_qo_i] = global_max[offset + thread_qo_i];
                block_sum_old[thread_qo_i] = global_sum[offset + thread_qo_i];
            }
            __syncthreads();

            /* Compute deltas */
            // compute logits, each thread does one row/col dot product
            {
                float curr_logit = 0.0;
                for (int d = 0; d < D_QK; d++) {
                    float q_value = block_q[thread_qo_i * D_QK + d]; // bank broadcast
                    float k_value = block_k[thread_kv_j * (D_QK + 1) + d]; // avoid bank conflict
                    curr_logit += q_value * k_value;
                }
                block_logits[thread_qo_i][thread_kv_j] = curr_logit;
            }
            __syncthreads();

            // TODO after this point block_q is never used again, we could store other stuff in there
            // TODO check similar things for other shared memory blocks
            //    => the goal is to fit more blocks on a SM for when we add batching/multiple heads

            // compute new max per query
            if (is_first_q_thread) {
                float curr_max = block_max_old[thread_qo_i];
                for (int j = 0; j < B_KV; j++) {
                    curr_max = max(curr_max, block_logits[thread_qo_i][j]);
                }
                block_max_new[thread_qo_i] = curr_max;
            }
            __syncthreads();

            // compute exp(logit-max)
            block_logits[thread_qo_i][thread_kv_j] = expf(
                    block_logits[thread_qo_i][thread_kv_j] - block_max_new[thread_qo_i]
            );
            __syncthreads();

            // compute new sum per query
            if (is_first_q_thread) {
                float scalar_old = expf(block_max_old[thread_qo_i] - block_max_new[thread_qo_i]);
                float curr_sum = scalar_old * block_sum_old[thread_qo_i];
                for (int j = 0; j < B_KV; j++) {
                    curr_sum += block_logits[thread_qo_i][j];
                }
                block_sum_new[thread_qo_i] = curr_sum;
            }
            __syncthreads();

            // compute output
            // every thread calculates (D_VO / B_KV) output values
            static_assert(true && (D_VO % B_KV == 0), "B_KV must divide D_VO");

            float scale_old = block_sum_old[thread_qo_i]
                              * expf(block_max_old[thread_qo_i] - block_max_new[thread_qo_i]);
            float scale_shared = 1.0f / block_sum_new[thread_qo_i];

            for (int d = thread_kv_j; d < D_VO; d += B_KV) {
                float o_delta_curr = 0.0;
                for (int j = 0; j < B_KV; j++) {
                    o_delta_curr += block_logits[thread_qo_i][j] * block_v[j * D_VO + d];
                }
                block_o[thread_qo_i * D_VO + d] =
                        scale_shared * (scale_old * block_o[thread_qo_i * D_VO + d] + o_delta_curr);
            }
            __syncthreads();

            /* Store outputs */
            // store o
            for (int i = info.thread_id; i < B_QO * D_VO; i += info.threads_per_block) {
                int offset = block_qo_i * B_QO * D_VO;
                global_o[offset + i] = block_o[i];
            }

            // store max, sum
            if (is_first_q_thread) {
                int offset = block_qo_i * B_QO;
                global_max[offset + thread_qo_i] = block_max_new[thread_qo_i];
                global_sum[offset + thread_qo_i] = block_sum_new[thread_qo_i];
            }

            __syncthreads();
        }
    }
}

// launch_bounds ensures the compiler doesn't use too many registers
__global__ void __launch_bounds__(B_QO * B_KV)

attention_kernel(
        float *global_q, float *global_k, float *global_v,
        float *global_o, float *scratch
) {
    attention_kernel_inner(global_q, global_k, global_v, global_o, scratch);
}

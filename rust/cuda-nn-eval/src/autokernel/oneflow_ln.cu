template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int cols_per_thread,
        int thread_group_width, int rows_per_access, bool padding>
__global__ void LayerNormWarpImpl(LOAD load, STORE store, const int64_t rows, const int64_t cols,
                                  const double epsilon, ComputeType *mean,
                                  ComputeType *inv_variance) {
    static_assert(cols_per_thread % pack_size == 0, "");
    static_assert(thread_group_width <= kWarpSize, "");
    static_assert(kWarpSize % thread_group_width == 0, "");
    constexpr int num_packs = cols_per_thread / pack_size;
    assert(cols <= cols_per_thread * thread_group_width);

    ComputeType buf[rows_per_access][cols_per_thread];
    const int64_t global_thread_group_id = blockIdx.x * blockDim.y + threadIdx.y;
    const int64_t num_global_thread_group = gridDim.x * blockDim.y;
    const int64_t lane_id = threadIdx.x;

    for (int64_t row = global_thread_group_id * rows_per_access;
         row < rows; row += num_global_thread_group * rows_per_access) {
        ComputeType thread_mean[rows_per_access];
        ComputeType thread_m2[rows_per_access];
        ComputeType thread_count[rows_per_access];

#pragma unroll
        for (int row_id = 0; row_id < rows_per_access; ++row_id) {
            thread_mean[row_id] = 0;
            thread_m2[row_id] = 0;
            thread_count[row_id] = 0;
            ComputeType *row_buf = buf[row_id];

#pragma unroll
            for (int pack_id = 0; pack_id < num_packs; ++pack_id) {
                const int col = (pack_id * thread_group_width + lane_id) * pack_size;
                const int pack_offset = pack_id * pack_size;
                if (!padding || col < cols) {
                    load.template load<pack_size>(row_buf + pack_offset, row + row_id, col);

#pragma unroll
                    for (int i = 0; i < pack_size; ++i) {
                        WelfordCombine(row_buf[pack_offset + i], thread_mean + row_id, thread_m2 + row_id,
                                       thread_count + row_id);
                    }
                } else {

#pragma unroll
                    for (int i = 0; i < pack_size; ++i) { row_buf[pack_offset + i] = 0; }
                }
            }
        }

        ComputeType warp_mean[rows_per_access];
        ComputeType warp_m2[rows_per_access];
        ComputeType warp_count[rows_per_access];

#pragma unroll
        for (int row_id = 0; row_id < rows_per_access; ++row_id) {
            int global_row_id = row + row_id;
            ComputeType *row_buf = buf[row_id];
            WelfordWarpAllReduce<ComputeType, thread_group_width>(
                    thread_mean[row_id], thread_m2[row_id], thread_count[row_id], warp_mean + row_id,
                    warp_m2 + row_id, warp_count + row_id);
            ComputeType row_mean = warp_mean[row_id];
            ComputeType row_variance =
                    max(Div(warp_m2[row_id], warp_count[row_id]), static_cast<ComputeType>(0.0));
            ComputeType row_inv_var = Rsqrt(row_variance + static_cast<ComputeType>(epsilon));
            if (lane_id == 0) {
                mean[global_row_id] = row_mean;
                inv_variance[global_row_id] = row_inv_var;
            }

#pragma unroll
            for (int i = 0; i < cols_per_thread; ++i) {
                row_buf[i] = (row_buf[i] - row_mean) * row_inv_var;
            }

#pragma unroll
            for (int i = 0; i < num_packs; ++i) {
                const int col = (i * thread_group_width + lane_id) * pack_size;
                if (!padding || col < cols) {
                    store.template store<pack_size>(row_buf + i * pack_size, global_row_id, col);
                }
            }
        }
    }
}
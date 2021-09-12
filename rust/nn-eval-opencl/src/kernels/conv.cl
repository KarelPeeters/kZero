/*
Required defines:
    function settings:
        RELU if defined apply relu to the output
        RES if defined add a skip connection

    shape settings:
        F: int, filter size
        HF: int, half filter size

        S: int, input and output image width and height
        C: int, input and output channels
*/

//TODO make batch size dynamic again when local buffers start working properly
//TODO support bias
//TODO support res coming from different kernel
//TODO support pre-activation
//TODO support different input and output channels again

#define INDEX3(v, i0, i1, i2, s1, s2) v[i0*(s1*s2) + i1*(s2) + i2]
#define INDEX4(v, i0, i1, i2, i3, s1, s2, s3) v[i0*(s1*s2*s3) + i1*(s2*s3) + i2*s3 + i3]

kernel void convolution(
    const int batch_size,
    // B x C x S x S
    const global float *restrict input,
    // C x C x F x F
    const global float *restrict filter,
    // B x C x S x S
    global float *restrict output
) {
    // local copy of the input
    // C x F x F
    local float input_local[C * F * F];

    // each kernel computes a single output value
    int co = get_global_id(0);
    int ox = get_global_id(1);
    int oy = get_global_id(2);

    // ... and copies a single input channel
    int copy_ci = get_local_id(0);

    for (int b = 0; b < batch_size; b++) {
        // copy input into local memory
        #pragma unroll
        for (int fx = 0; fx < F; fx++) {
            #pragma unroll
            for (int fy = 0; fy < F; fy++) {
                int ix = ox + fx - HF;
                int iy = oy + fy - HF;

                if (0 <= ix && ix < S && 0 <= iy && iy < S) {
                    float value = INDEX3(input, copy_ci, ix, iy, S, S);
                    INDEX3(input_local, copy_ci, fx, fy, F, F) = value;
                }
            }
        }

        // wait for all threads to finish copying
        barrier(CLK_LOCAL_MEM_FENCE);

        // do the actual computation
        float total = 0.0;

        for (int ci = 0; ci < C; ci++) {
            #pragma unroll
            for (int fx = 0; fx < F; fx++) {
                #pragma unroll
                for (int fy = 0; fy < F; fy++) {
                    int ix = ox + fx - HF;
                    int iy = oy + fy - HF;

                    if (0 <= ix && ix < S && 0 <= iy && iy < S) {
                        float i = INDEX3(input_local, ci, fx, fy, F, F);
                        float f = INDEX4(filter, co, ci, fx, fy, C, F, F);
                        total += f * i;
                    }
                }
            }
        }

        #ifdef RES
        total += INDEX3(input_local, co, HF, HF, S, S);
        #endif

        #ifdef RELU
        total = max(total, 0.0f);
        #endif

        INDEX4(output, b, co, ox, oy, C, S, S) = total;
    }
}
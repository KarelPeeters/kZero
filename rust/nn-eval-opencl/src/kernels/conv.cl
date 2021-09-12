/*
Required defines:
    function settings:
        RELU if defined apply relu to the output
        RES if defined add a skip connection

    shape settings:
        F: int, filter size
        HF: int, half filter size

        S: int, input and output image width and height
        CI: int, input channels
        CO: int, output channels
*/

#define INDEX4(v, i0, i1, i2, i3, s1, s2, s3) v[i0*(s1*s2*s3) + i1*(s2*s3) + i2*s3 + i3]

kernel void convolution(
    const int batch_size,
    // B x CI x S x S
    const global float *input,
    // CO x CI x F x F
    const global float *filter,
    // B x CO x S x S
    global float *output
) {
    // each kernel computes a single output value
    int co = get_global_id(0);
    int ox = get_global_id(1);
    int oy = get_global_id(2);

    for (int b = 0; b < batch_size; b++) {
        float total = 0.0;

        for (int ci = 0; ci < CI; ci++) {
            #pragma unroll
            for (int fx = 0; fx < F; fx++) {
                #pragma unroll
                for (int fy = 0; fy < F; fy++) {
                    int ix = ox + fx - HF;
                    int iy = oy + fy - HF;

                    if (0 <= ix && ix < S && 0 <= iy && iy < S) {
                        float i = INDEX4(input, b, ci, ix, iy, CI, S, S);
                        float f = INDEX4(filter, co, ci, fx, fy, CI, F, F);
                        total += f * i;
                    }
                }
            }
        }

        #ifdef RES
        total += INDEX4(input, b, co, HF, HF, CI, S, S);
        #endif

        #ifdef RELU
        total = max(total, 0.0f);
        #endif

        INDEX4(output, b, co, ox, oy, CO, S, S) = total;
    }
}
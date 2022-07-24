
static void convolution_pack1ton_int8_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_int8, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
{
    const int packn = csrr_vlenb();
    const word_type vl = vsetvl_e8m1(packn);

    int w = bottom_blob.w;
    int channels = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w * dilation_h - kernel_w * dilation_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }

    // num_output
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        int* outptr = top_blob.channel(p);

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                vint32m4_t _sum = vmv_v_x_i8m4(0, vl);

                const signed char* kptr = weight_data_int8.channel(p);

                // channels
                for (int q = 0; q < channels; q++)
                {
                    const Mat m = bottom_blob.channel(q);
                    const signed char* sptr = m.row<const signed char>(i * stride_h) + j * stride_w;

                    for (int k = 0; k < maxk; k++) // 29.23
                    {
                        char val = sptr[space_ofs[k]];
                        vint8m1_t _w = vle8_v_i8m1(kptr, vl);
                        vint16m2_t _s0 = vwmul_vx_i16m2(_w, val, vl);
                        _sum = vwadd_wv_i32m4(_sum, _s0, vl);

                        kptr += packn;
                    }
                }

                vse32_v_i32m4(outptr + j * packn, _sum, vl);
            }

            outptr += outw * packn;
        }
    }
}
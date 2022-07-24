// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

static void convolution_packnto1_int8_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_int8, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
{
    const int packn = csrr_vlenb();
    const word_type vl = vsetvl_e32m1(packn);

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
                    const signed int* sptr = m.row<const signed char>(i * stride_h) + j * stride_w * packn;

                    for (int k = 0; k < maxk; k++)
                    {
                        vint8m1_t _val = vle8_v_i8m1(sptr + space_ofs[k] * packn, vl);
                        vint8m1_t _w = vle8_v_i8m1(kptr, vl);
                        vint16m2_t _s8 = vwmul_vv_i16m2(_val, _w, vl);

                        _sum = vwadd_wv_i32m4(_sum, _s8, vl);

                        kptr += packn;
                    }
                }

                sum = vfmv_f_s_f32m1_f32(vfredusum_vs_f32m1_f32m1(vfloat32m1_t(), _sum, vfmv_s_f_f32m1(vfloat32m1_t(), sum, vl), vl));

                outptr[j] = sum;
            }

            outptr += outw;
        }
    }
}
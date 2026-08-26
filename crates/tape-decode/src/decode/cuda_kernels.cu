struct ComplexF {
    float x;
    float y;
};

__device__ __forceinline__ ComplexF complex_sub(ComplexF a, ComplexF b) {
    return {a.x - b.x, a.y - b.y};
}

__device__ __forceinline__ float atan_approx(float x) {
    const float a1 = 0.99997726f;
    const float a3 = -0.33262347f;
    const float a5 = 0.19354346f;
    const float a7 = -0.11643287f;
    const float a9 = 0.05265332f;
    const float a11 = -0.0117212f;
    const float x2 = x * x;
    return x * (a1 + x2 * (a3 + x2 * (a5 + x2 * (a7 + x2 * (a9 + x2 * a11)))));
}

__device__ __forceinline__ float atan2_tape(float y, float x) {
    const float half_pi = 1.57079632679489661923f;
    const float pi = 3.14159265358979323846f;
    x += copysignf(1.1754943508222875e-38f, x);
    const bool swap = fabsf(x) < fabsf(y);
    const float input = (swap ? x : y) / (swap ? y : x);
    float result = atan_approx(input);
    const float quadrant = input >= 0.0f ? half_pi : -half_pi;
    result = swap ? quadrant - result : result;
    if (x >= 0.0f) return result;
    return y >= 0.0f ? pi + result : -pi + result;
}

__device__ __forceinline__ float unwrap_pair(ComplexF a, ComplexF b, float freq, float offset) {
    const float tau = 6.28318530717958647692f;
    float diff = atan2_tape(b.y, b.x) - atan2_tape(a.y, a.x);
    diff -= floorf(diff / tau) * tau;
    return diff * freq / tau - offset;
}

extern "C" __global__ void filter_real(
    ComplexF* spectra,
    const float* gains,
    int half_bins,
    int total_bins
) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= total_bins) return;
    const float gain = gains[i % half_bins];
    spectra[i].x *= gain;
    spectra[i].y *= gain;
}

extern "C" __global__ void filter_complex(
    ComplexF* spectra,
    const ComplexF* gains,
    int half_bins,
    int total_bins
) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= total_bins) return;
    const int bin = i % half_bins;
    const ComplexF a = spectra[i];
    const ComplexF b = gains[bin];
    spectra[i].x = fmaf(a.x, b.x, -(a.y * b.y));
    spectra[i].y = (bin == 0 || bin == half_bins - 1)
        ? 0.0f
        : fmaf(a.x, b.y, a.y * b.x);
}

// Build the three output spectra in one pass so cuFFT can execute all inverse
// transforms as one larger batch. Layout is channel-major: video, delayed
// video, then burst; each channel contains all block spectra contiguously.
extern "C" __global__ void prepare_output_spectra(
    const ComplexF* demod,
    const ComplexF* raw,
    const ComplexF* video_gains,
    const ComplexF* video05_gains,
    const float* burst_gains,
    ComplexF* output,
    int half_bins,
    int spectrum_bins
) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = 3 * spectrum_bins;
    if (i >= total) return;
    const int channel = i / spectrum_bins;
    const int source_index = i - channel * spectrum_bins;
    const int bin = source_index % half_bins;
    if (channel == 2) {
        const ComplexF value = raw[source_index];
        const float gain = burst_gains[bin];
        output[i] = {value.x * gain, value.y * gain};
        return;
    }
    const ComplexF value = demod[source_index];
    const ComplexF gain = channel == 0 ? video_gains[bin] : video05_gains[bin];
    output[i].x = fmaf(value.x, gain.x, -(value.y * gain.y));
    output[i].y = (bin == 0 || bin == half_bins - 1)
        ? 0.0f
        : fmaf(value.x, gain.y, value.y * gain.x);
}

// Apply one SOS biquad over a field. Each lane first reduces a contiguous chunk
// to the affine state transform s' = M*s + v. A block-wide prefix scan then
// supplies the exact entering state for every chunk, after which lanes replay
// their local samples and emit the section output. The reverse flag changes
// logical traversal only; data remains in physical order for the next section.
extern "C" __global__ void sos_section_scan(
    const float* input,
    float* output,
    const float* initial_value,
    int len,
    int chunk,
    int reverse,
    float b0,
    float neg_a1,
    float neg_a2,
    float bff1,
    float bff2,
    float zi0_base,
    float zi1_base
) {
    const int lane = (int)threadIdx.x;
    int begin = lane * chunk;
    int end = begin + chunk;
    if (end > len) end = len;

    // Transform for this lane's chunk, initialized to identity.
    float m00 = 1.0f;
    float m01 = 0.0f;
    float m10 = 0.0f;
    float m11 = 1.0f;
    float v0 = 0.0f;
    float v1 = 0.0f;
    for (int logical = begin; logical < end; ++logical) {
        const int index = reverse ? len - 1 - logical : logical;
        const float sample = input[index];
        const float old_m00 = m00;
        const float old_m01 = m01;
        const float old_v0 = v0;
        m00 = fmaf(neg_a1, old_m00, m10);
        m01 = fmaf(neg_a1, old_m01, m11);
        m10 = neg_a2 * old_m00;
        m11 = neg_a2 * old_m01;
        v0 = fmaf(neg_a1, old_v0, fmaf(bff1, sample, v1));
        v1 = fmaf(neg_a2, old_v0, bff2 * sample);
    }

    __shared__ float scan_m00[1024];
    __shared__ float scan_m01[1024];
    __shared__ float scan_m10[1024];
    __shared__ float scan_m11[1024];
    __shared__ float scan_v0[1024];
    __shared__ float scan_v1[1024];
    scan_m00[lane] = m00;
    scan_m01[lane] = m01;
    scan_m10[lane] = m10;
    scan_m11[lane] = m11;
    scan_v0[lane] = v0;
    scan_v1[lane] = v1;
    __syncthreads();

    // Inclusive scan in chronological order: current transform after prefix.
    for (int offset = 1; offset < 1024; offset <<= 1) {
        float p00 = 1.0f;
        float p01 = 0.0f;
        float p10 = 0.0f;
        float p11 = 1.0f;
        float pv0 = 0.0f;
        float pv1 = 0.0f;
        if (lane >= offset) {
            p00 = scan_m00[lane - offset];
            p01 = scan_m01[lane - offset];
            p10 = scan_m10[lane - offset];
            p11 = scan_m11[lane - offset];
            pv0 = scan_v0[lane - offset];
            pv1 = scan_v1[lane - offset];
        }
        const float c00 = scan_m00[lane];
        const float c01 = scan_m01[lane];
        const float c10 = scan_m10[lane];
        const float c11 = scan_m11[lane];
        const float cv0 = scan_v0[lane];
        const float cv1 = scan_v1[lane];
        __syncthreads();
        if (lane >= offset) {
            scan_m00[lane] = fmaf(c00, p00, c01 * p10);
            scan_m01[lane] = fmaf(c00, p01, c01 * p11);
            scan_m10[lane] = fmaf(c10, p00, c11 * p10);
            scan_m11[lane] = fmaf(c10, p01, c11 * p11);
            scan_v0[lane] = fmaf(c00, pv0, fmaf(c01, pv1, cv0));
            scan_v1[lane] = fmaf(c10, pv0, fmaf(c11, pv1, cv1));
        }
        __syncthreads();
    }

    const float initial = initial_value[0];
    float state0 = zi0_base * initial;
    float state1 = zi1_base * initial;
    if (lane > 0) {
        const float base0 = state0;
        const float base1 = state1;
        state0 = fmaf(
            scan_m00[lane - 1], base0,
            fmaf(scan_m01[lane - 1], base1, scan_v0[lane - 1])
        );
        state1 = fmaf(
            scan_m10[lane - 1], base0,
            fmaf(scan_m11[lane - 1], base1, scan_v1[lane - 1])
        );
    }

    for (int logical = begin; logical < end; ++logical) {
        const int index = reverse ? len - 1 - logical : logical;
        const float sample = input[index];
        const float old_state0 = state0;
        output[index] = fmaf(b0, sample, old_state0);
        state0 = fmaf(neg_a1, old_state0, fmaf(bff1, sample, state1));
        state1 = fmaf(neg_a2, old_state0, bff2 * sample);
    }
}

extern "C" __global__ void capture_last(
    const float* input,
    float* value,
    int len
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) value[0] = input[len - 1];
}

extern "C" __global__ void analytic_expand(
    const ComplexF* half,
    ComplexF* full,
    int n,
    int half_bins,
    int total
) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= total) return;
    const int block = i / n;
    const int bin = i - block * n;
    ComplexF value = {0.0f, 0.0f};
    if (bin < half_bins) {
        value = half[block * half_bins + bin];
        if (bin > 0 && bin < half_bins - 1) {
            value.x *= 2.0f;
            value.y *= 2.0f;
        }
    }
    full[i] = value;
}

extern "C" __global__ void demod_envelope(
    const ComplexF* hilbert,
    float* demod,
    float* raw_envelope,
    int n,
    int total,
    float freq,
    float offset
) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= total) return;
    const int local = i % n;
    const int base = i - local;
    demod[i] = local == 0 ? -offset : unwrap_pair(hilbert[i - 1], hilbert[i], freq, offset);
    int envelope_source = local - 4;
    if (envelope_source < 0) envelope_source += n;
    raw_envelope[i] = fabsf(hilbert[base + envelope_source].x) / (float)n;
}

extern "C" __global__ void demod_diffed(
    const ComplexF* hilbert,
    float* output,
    int n,
    int total,
    float freq,
    float offset
) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= total) return;
    const int local = i % n;
    if (local == 0) {
        output[i] = -offset;
        return;
    }
    const ComplexF zero = {0.0f, 0.0f};
    const ComplexF prev = local == 1 ? zero : complex_sub(hilbert[i - 1], hilbert[i - 2]);
    const ComplexF curr = complex_sub(hilbert[i], hilbert[i - 1]);
    output[i] = unwrap_pair(prev, curr, freq, offset);
}

__device__ __forceinline__ float sos1_step(
    float sample,
    float b0,
    float recurrence,
    float feed_forward,
    float* state
) {
    const float output = fmaf(b0, sample, *state);
    *state = fmaf(recurrence, *state, feed_forward * sample);
    return output;
}

extern "C" __global__ void filter_envelope_scan(
    const float* raw,
    float* work,
    float* packed,
    int n,
    int usable,
    int cut,
    int blocks,
    float b0,
    float recurrence,
    float feed_forward,
    float zi0_base
) {
    const int signal = (int)blockIdx.x;
    const int lane = (int)threadIdx.x;
    if (signal >= blocks) return;
    const int edge = 6;
    const int chunk = 33;
    const int total = n + edge;
    const float* input = raw + signal * n;
    float* filtered = work + signal * total;
    __shared__ float scan_a[1024];
    __shared__ float scan_b[1024];
    __shared__ float initial_state;

    if (lane == 0) {
        const float left_end = input[0];
        float state = zi0_base * (2.0f * left_end - input[edge]);
        for (int i = edge; i >= 1; --i) {
            sos1_step(
                2.0f * left_end - input[i], b0, recurrence, feed_forward, &state
            );
        }
        initial_state = state;
    }

    int begin = lane * chunk;
    int end = begin + chunk;
    if (end > total) end = total;
    float transform_a = 1.0f;
    float transform_b = 0.0f;
    const float right_end = input[n - 1];
    for (int i = begin; i < end; ++i) {
        const float sample = i < n
            ? input[i]
            : 2.0f * right_end - input[n - 1 - (i - n + 1)];
        transform_b = fmaf(recurrence, transform_b, feed_forward * sample);
        transform_a *= recurrence;
    }
    scan_a[lane] = transform_a;
    scan_b[lane] = transform_b;
    __syncthreads();
    for (int offset = 1; offset < 1024; offset <<= 1) {
        float previous_a = 1.0f;
        float previous_b = 0.0f;
        const float current_a = scan_a[lane];
        const float current_b = scan_b[lane];
        if (lane >= offset) {
            previous_a = scan_a[lane - offset];
            previous_b = scan_b[lane - offset];
        }
        __syncthreads();
        if (lane >= offset) {
            scan_a[lane] = current_a * previous_a;
            scan_b[lane] = fmaf(current_a, previous_b, current_b);
        }
        __syncthreads();
    }
    float state = initial_state;
    if (lane > 0) {
        state = fmaf(scan_a[lane - 1], initial_state, scan_b[lane - 1]);
    }
    for (int i = begin; i < end; ++i) {
        const float sample = i < n
            ? input[i]
            : 2.0f * right_end - input[n - 1 - (i - n + 1)];
        filtered[i] = sos1_step(sample, b0, recurrence, feed_forward, &state);
    }
    __syncthreads();

    if (lane == 0) {
        float backward_state = zi0_base * filtered[total - 1];
        for (int padding = 0; padding < edge; ++padding) {
            sos1_step(
                filtered[total - 1 - padding],
                b0,
                recurrence,
                feed_forward,
                &backward_state
            );
        }
        initial_state = backward_state;
    }

    begin = lane * chunk;
    end = begin + chunk;
    if (end > n) end = n;
    transform_a = 1.0f;
    transform_b = 0.0f;
    for (int i = begin; i < end; ++i) {
        const float sample = filtered[n - 1 - i];
        transform_b = fmaf(recurrence, transform_b, feed_forward * sample);
        transform_a *= recurrence;
    }
    scan_a[lane] = transform_a;
    scan_b[lane] = transform_b;
    __syncthreads();
    for (int offset = 1; offset < 1024; offset <<= 1) {
        float previous_a = 1.0f;
        float previous_b = 0.0f;
        const float current_a = scan_a[lane];
        const float current_b = scan_b[lane];
        if (lane >= offset) {
            previous_a = scan_a[lane - offset];
            previous_b = scan_b[lane - offset];
        }
        __syncthreads();
        if (lane >= offset) {
            scan_a[lane] = current_a * previous_a;
            scan_b[lane] = fmaf(current_a, previous_b, current_b);
        }
        __syncthreads();
    }
    state = initial_state;
    if (lane > 0) {
        state = fmaf(scan_a[lane - 1], initial_state, scan_b[lane - 1]);
    }
    for (int i = begin; i < end; ++i) {
        const int source = n - 1 - i;
        const float output = sos1_step(
            filtered[source], b0, recurrence, feed_forward, &state
        );
        if (source >= cut && source < n - cut) {
            packed[signal * usable + source - cut] = output;
        }
    }
}

extern "C" __global__ void mark_candidates(
    const float* demod,
    unsigned char* candidates,
    int total,
    float threshold
) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < total) candidates[i] = demod[i] > threshold;
}

extern "C" __global__ void repair_spikes(
    float* demod,
    const float* diffed,
    const unsigned char* candidates,
    unsigned char* block_flags,
    int n,
    int blocks,
    float threshold
) {
    const int block = (int)blockIdx.x;
    if (block >= blocks) return;
    const int lane = (int)threadIdx.x;
    const int base = block * n;
    __shared__ float maxima[256];
    float lane_max = -3.402823466e+38f;
    for (int i = 20 + lane; i < n - 20; i += 256) {
        lane_max = fmaxf(lane_max, demod[base + i]);
    }
    maxima[lane] = lane_max;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) maxima[lane] = fmaxf(maxima[lane], maxima[lane + stride]);
        __syncthreads();
    }
    if (lane == 0) block_flags[block] = maxima[0] > threshold;
    if (maxima[0] <= threshold || lane != 0) return;
    for (int i = 0; i < n; ++i) {
        if (!candidates[base + i]) continue;
        const int start = i > 8 ? i - 8 : 0;
        int end = i + 30;
        if (end > n - 1) end = n - 1;
        float normal_max = demod[base + start];
        float diffed_max = diffed[base + start];
        for (int j = start + 1; j < end; ++j) {
            normal_max = fmaxf(normal_max, demod[base + j]);
            diffed_max = fmaxf(diffed_max, diffed[base + j]);
        }
        if (diffed_max < normal_max) {
            for (int j = start; j < end; ++j) demod[base + j] = diffed[base + j];
        }
    }
}

extern "C" __global__ void pack_luma(
    const float* video,
    const float* video05,
    float* packed_video,
    float* packed_video05,
    int n,
    int usable,
    int cut,
    int video05_shift,
    int total,
    float inv_n,
    float ire0
) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= total) return;
    const int block = i / usable;
    const int local = i - block * usable + cut;
    int shifted = local + video05_shift;
    if (shifted >= n) shifted -= n;
    packed_video[i] = video[block * n + local] * inv_n + ire0;
    packed_video05[i] = video05[block * n + shifted] * inv_n + ire0;
}

extern "C" __global__ void pack_luma_combined(
    const float* outputs,
    float* packed_video,
    float* packed_video05,
    int real_bins,
    int n,
    int usable,
    int cut,
    int video05_shift,
    int total,
    float inv_n,
    float ire0
) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= total) return;
    const int block = i / usable;
    const int local = i - block * usable + cut;
    int shifted = local + video05_shift;
    if (shifted >= n) shifted -= n;
    packed_video[i] = outputs[block * n + local] * inv_n + ire0;
    packed_video05[i] = outputs[real_bins + block * n + shifted] * inv_n + ire0;
}

extern "C" __global__ void burst_means(
    const float* burst,
    float* means,
    int n,
    int blocks,
    float inv_n
) {
    const int block = (int)blockIdx.x;
    if (block >= blocks) return;
    const int lane = (int)threadIdx.x;
    __shared__ float lane_sums[16];
    float sum = 0.0f;
    if (lane < 16) {
        for (int i = lane; i < n; i += 16) {
            sum += burst[block * n + i] * inv_n;
        }
        lane_sums[lane] = sum;
    }
    __syncthreads();
    if (lane == 0) {
        sum = lane_sums[0];
        for (int i = 1; i < 16; ++i) sum += lane_sums[i];
        means[block] = sum / (float)n;
    }
}

extern "C" __global__ void burst_means_combined(
    const float* outputs,
    float* means,
    int real_bins,
    int n,
    int blocks,
    float inv_n
) {
    const int block = (int)blockIdx.x;
    if (block >= blocks) return;
    const int lane = (int)threadIdx.x;
    const float* burst = outputs + 2 * real_bins;
    __shared__ float lane_sums[16];
    float sum = 0.0f;
    if (lane < 16) {
        for (int i = lane; i < n; i += 16) {
            sum += burst[block * n + i] * inv_n;
        }
        lane_sums[lane] = sum;
    }
    __syncthreads();
    if (lane == 0) {
        sum = lane_sums[0];
        for (int i = 1; i < 16; ++i) sum += lane_sums[i];
        means[block] = sum / (float)n;
    }
}

extern "C" __global__ void pack_burst(
    const float* burst,
    const float* means,
    float* packed,
    int n,
    int usable,
    int cut,
    int shift,
    int total,
    float inv_n
) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= total) return;
    const int block = i / usable;
    const int local = i - block * usable + cut;
    int source = local - shift;
    source %= n;
    if (source < 0) source += n;
    packed[i] = burst[block * n + source] * inv_n - means[block];
}

extern "C" __global__ void pack_burst_combined(
    const float* outputs,
    const float* means,
    float* packed,
    int real_bins,
    int n,
    int usable,
    int cut,
    int shift,
    int total,
    float inv_n
) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= total) return;
    const int block = i / usable;
    const int local = i - block * usable + cut;
    int source = local - shift;
    source %= n;
    if (source < 0) source += n;
    const float* burst = outputs + 2 * real_bins;
    packed[i] = burst[block * n + source] * inv_n - means[block];
}

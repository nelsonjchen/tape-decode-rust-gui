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

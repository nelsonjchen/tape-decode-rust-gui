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
    const int block = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (block >= blocks) return;
    const int base = block * n;
    float interior_max = -3.402823466e+38f;
    for (int i = 20; i < n - 20; ++i) interior_max = fmaxf(interior_max, demod[base + i]);
    if (interior_max <= threshold) return;
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
    const int block = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (block >= blocks) return;
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += burst[block * n + i] * inv_n;
    means[block] = sum / (float)n;
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

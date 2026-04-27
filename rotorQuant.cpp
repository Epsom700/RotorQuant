#include "rotorQuant.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <iostream>
#include <stdexcept>
#include <Accelerate/Accelerate.h>
using namespace std;


RotorQuant::RotorQuant(int n, int num_levels, double sigma)
    : rotation_(n), lloyd_max_(num_levels, sigma), n_(n)
{
    // Build f32 caches once.
    const auto& H = rotation_.hadamard();
    hadamard_f32_.resize(static_cast<size_t>(n) * n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            hadamard_f32_[static_cast<size_t>(i) * n + j] = static_cast<float>(H[i][j]);

    const auto& flips = rotation_.random_flips();
    flips_f32_.resize(n);
    for (int i = 0; i < n; i++) flips_f32_[i] = static_cast<float>(flips[i]);

    const auto& bp = lloyd_max_.breakpoints();
    breakpoints_f32_.resize(bp.size());
    for (size_t i = 0; i < bp.size(); i++) breakpoints_f32_[i] = static_cast<float>(bp[i]);

    const auto& c = lloyd_max_.centroids();
    centroids_f32_.resize(c.size());
    for (size_t i = 0; i < c.size(); i++) centroids_f32_[i] = static_cast<float>(c[i]);
}
    
vector<int> RotorQuant::encode(const vector<double>& activations){
    vector<double> rotated_activations = rotation_.rotate(activations); 
    vector<int> quantized_vector = lloyd_max_.quantize_vector(rotated_activations);
    return quantized_vector; 
}

vector<double> RotorQuant::decode(const vector<int>& bins){
    vector<double> deQuantized_bins = lloyd_max_.deQuantize_vector(bins); 
    vector<double> rotated_bins = rotation_.inverse_rotate(deQuantized_bins); 
    return rotated_bins; 
}

vector<vector<int>> RotorQuant::encode_2d(const vector<vector<double>>& activations){
    vector<vector<int>> encoded_2d_vectors(activations.size()); 
    #pragma omp parallel for
    for (int i=0; i < activations.size(); i++){
        encoded_2d_vectors[i] = encode(activations[i]); 
        
    }
    return encoded_2d_vectors;
}


vector<vector<double>> RotorQuant::decode_2d(const vector<vector<int>>& bins){
    vector<vector<double>> decoded_2d_vectors(bins.size());
    #pragma omp parallel for
    for (int i = 0; i < bins.size(); i ++){
        decoded_2d_vectors[i] = decode(bins[i]);

    }
    return decoded_2d_vectors;
}

void RotorQuant::encode_decode_2d_inplace(double* data, int rows, int cols){
    #pragma omp parallel for
    for (int i = 0; i < rows; i++){
        std::vector<double> row(data + i*cols, data + (i+1)*cols);
        std::vector<int> encoded = encode(row);
        std::vector<double> decoded = decode(encoded);
        std::copy(decoded.begin(), decoded.end(), data + i*cols);
    }
}

// Batched float32 round-trip using Apple Accelerate.
//
// Math (per-row, treating x as a column vector):
//   forward  : r = H * (flips ⊙ x)         then quantize r elementwise
//   inverse  : y = flips ⊙ (H^T * dequant(r))
//
// In row-major batch form (rows of X are samples):
//   X *= flips                   (broadcast)
//   R  = X @ H^T                 (sgemm)
//   R  = quantize/dequantize(R)  (bucketize + LUT, elementwise)
//   Y  = R @ H                   (sgemm)
//   Y *= flips                   (broadcast)
//
// Uses cblas_sgemm (Accelerate) which leverages AMX/NEON on Apple Silicon.
void RotorQuant::encode_decode_batch_f32(float* data, int rows, int cols){
    if (cols != n_) throw std::runtime_error("RotorQuant: cols must equal n");
    const int N = rows * cols;
    const float* H = hadamard_f32_.data();
    const float* flips = flips_f32_.data();
    const float* bp = breakpoints_f32_.data();   // size L+1
    const float* cent = centroids_f32_.data();   // size L
    const int L = static_cast<int>(centroids_f32_.size());

    // 1. data *= flips (broadcast over rows)
    for (int r = 0; r < rows; r++) {
        float* row = data + r * cols;
        for (int c = 0; c < cols; c++) row[c] *= flips[c];
    }

    // 2. rotated = data @ H^T  (rows x cols)
    std::vector<float> rotated(static_cast<size_t>(N));
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                rows, cols, cols,
                1.0f,
                data, cols,
                H, cols,
                0.0f,
                rotated.data(), cols);

    // 3. Elementwise quantize+dequantize on rotated.
    //    bp has -inf at index 0 and +inf at index L; interior breakpoints are bp[1..L-1].
    //    Bucket index in [0, L-1] is the largest k with bp[k] <= v.
    for (int i = 0; i < N; i++) {
        float v = rotated[i];
        int lo = 0, hi = L - 1;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (bp[mid + 1] > v) hi = mid; else lo = mid + 1;
        }
        rotated[i] = cent[lo];
    }

    // 4. data = rotated @ H
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                rows, cols, cols,
                1.0f,
                rotated.data(), cols,
                H, cols,
                0.0f,
                data, cols);

    // 5. data *= flips
    for (int r = 0; r < rows; r++) {
        float* row = data + r * cols;
        for (int c = 0; c < cols; c++) row[c] *= flips[c];
    }
}

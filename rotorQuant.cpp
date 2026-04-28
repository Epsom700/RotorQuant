#include "rotorQuant.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <iostream>
#include <stdexcept>
using namespace std;


RotorQuant::RotorQuant(int n, int num_levels, double sigma)
    : rotation_(n), lloyd_max_(num_levels, sigma), n_(n)
{
    // Build f32 caches once.
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

void RotorQuant::fwht_f32(float* data, int n) {
    // In-place Fast Walsh-Hadamard Transform (FWHT) for n a power of 2.
    for (int len = 1; len < n; len *= 2) {
        for (int i = 0; i < n; i += 2 * len) {
            for (int j = 0; j < len; j++) {
                float u = data[i + j];
                float v = data[i + j + len];
                data[i + j] = u + v;
                data[i + j + len] = u - v;
            }
        }
    }
    for (int i = 0; i < n; i++) data[i] /= sqrtf(static_cast<float>(n));
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

// Batched float32 round-trip using in-place FWHT.
//
// Math (per-row, treating x as a column vector):
//   forward  : r = FWHT(flips ⊙ x)         then quantize r elementwise
//   inverse  : y = flips ⊙ FWHT(dequant(r))
//
// In row-major batch form (rows of X are samples):
//   X *= flips                   (broadcast)
//   X  = FWHT(X)                 (in-place, per row, O(n log n))
//   X  = quantize/dequantize(X)  (bucketize + LUT, elementwise)
//   X  = FWHT(X)                 (in-place, per row)
//   X *= flips                   (broadcast)

void RotorQuant::encode_decode_batch_f32(float* data, int rows, int cols){
    if (cols != n_) throw std::runtime_error("RotorQuant: cols must equal n");
    const int N = rows * cols;  // size n
    const float* flips = flips_f32_.data();
    const float* bp = breakpoints_f32_.data();   // size L+1
    const float* cent = centroids_f32_.data();   // size L
    const int L = static_cast<int>(centroids_f32_.size());

    // 1. Multiply each column by its flip sign (apply random sign flips per-dimension)
    for (int r = 0; r < rows; r++) {
        float* row = data + r * cols;
        for (int c = 0; c < cols; c++) row[c] *= flips[c];
        fwht_f32(row, cols);  // in-place FWHT on the row
    }



    // 2. Elementwise quantize + dequantize the rotated coefficients.
    //    'bp' contains breakpoints (-inf .. +inf) and 'cent' contains centroids for each bucket.
    //    For each value v choose the bucket index k (largest k with bp[k] <= v) and write cent[k].
    for (int i = 0; i < N; i++) {
        float v = data[i];
        int lo = 0, hi = L - 1;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (bp[mid + 1] > v) hi = mid; else lo = mid + 1;
        }
        data[i] = cent[lo];
    }

    // 3. Inverse transform: apply FWHT again and re-apply flips to return to original domain.
    for (int r = 0; r < rows; r++) {
        float* row = data + r * cols;
        fwht_f32(row, cols);  // in-place FWHT on the row (H is symmetric)
        for (int c = 0; c < cols; c++) row[c] *= flips[c];
    }
}

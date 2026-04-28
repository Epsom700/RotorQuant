#pragma once
#include <vector>
#include "rotation.h"
#include "lloyd_max.h"


class RotorQuant{
    public:
        RotorQuant(int n, int num_levels, double sigma); 
        ~RotorQuant() = default; 
        std::vector<int> encode(const std::vector<double>& activations); 
        std::vector<double> decode(const std::vector<int>& bins); 
        std::vector<std::vector<int>> encode_2d(const std::vector<std::vector<double>>& hidden_dimensions);
        std::vector<std::vector<double>> decode_2d(const std::vector<std::vector<int>>& binned_2d);
        void encode_decode_2d_inplace(double* data, int rows, int cols);
        void encode_decode_batch_f32(float* data, int rows, int cols);

    private:
        Rotation rotation_;
        LloydMax lloyd_max_;
        int n_;    // n*n row-major
        std::vector<float> flips_f32_;        // n
        std::vector<float> breakpoints_f32_;  // L+1 (with -inf/+inf at ends)
        std::vector<float> centroids_f32_; 
        static void fwht_f32(float* data, int n);   // L
};
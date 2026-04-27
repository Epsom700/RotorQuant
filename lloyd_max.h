#pragma once 
#include <vector> 
#include <cmath> 
#include <string> 

class LloydMax{
    public: 
        LloydMax(int num_levels, double sigma); 
        ~LloydMax() = default;
        int quantize(double x); 
        double deQuantize(int bins); 
        std::vector<int> quantize_vector(const std::vector<double>& x);
        std::vector<double> deQuantize_vector(const std::vector<int>& bins);
        const std::vector<double>& centroids() const { return centroids_; }
        const std::vector<double>& breakpoints() const { return breakpoints_; }
    private:
        void run_lloyd_max(); 

        int num_levels_; 
        double sigma_; 
        double tol_;
        int max_iter_;  
        std::vector<double> centroids_; 
        std::vector<double> breakpoints_; 
}; 
#define _USE_MATH_DEFINES
#include "lloyd_max.h"
#include <vector> 
#include <cmath> 
#include <string> 
#include <iostream>
#include <random> 
#include <algorithm> 
using namespace std; 

LloydMax::LloydMax(int num_levels, double sigma)
    : num_levels_(num_levels), sigma_(sigma), tol_(1e-6), max_iter_(100) 
{
    run_lloyd_max();
}

double gaussian_pdf(double x){
    return (1.0/sqrt(2.0*M_PI)) * exp(-0.5* x*x);
}

double gaussian_cdf(double x){ 
    return (0.5 * (1+ erf(x/sqrt(2.0)))); 
}

double conditional_expectations(double a, double b){
    double num = gaussian_pdf(a) - gaussian_pdf(b);
    double dom = gaussian_cdf(b) - gaussian_cdf(a);
    if (dom < 1e-10) return (a+b)/2.0; 
    return num/dom;
}

vector<double> initialize_centroids(double min_val, double max_val, int num_levels){
    vector<double> centroids; 
    double step = (max_val - min_val)/(num_levels); 
    for (int i = 0; i < num_levels; i++){ 
        double val = min_val + (i+0.5) * step; 
        centroids.push_back(val); 
    }
    return centroids;
}

vector<double> update_breakpoints(const vector<double>& centroids){
    vector<double> breakpoints;
    int k = centroids.size(); 
    breakpoints.push_back(-INFINITY); 
    for (int i =0; i < k-1; i++){ 
        double b = (centroids[i] + centroids[i+1])/2.0;
        breakpoints.push_back(b);
    }
    breakpoints.push_back(INFINITY); 
    return breakpoints;
}

int binary_search(const vector<double>& X, double val){
    int low=0; 
    int high = X.size()-2; 
    while (low < high){ 
        int mid = low + (high-low)/2; 
        if (X[mid+1] > val) high = mid; 
        else low = mid+1; 
    }
    return low; 
}



void LloydMax::run_lloyd_max(){ 
    double min_val = -3.0 * sigma_; 
    double max_val = 3.0 * sigma_; 
    vector<double> centroids = initialize_centroids(min_val, max_val, num_levels_);
    for (int iter = 0; iter < max_iter_; iter++){
        vector<double> breakpoints = update_breakpoints(centroids);
        vector<double> new_centroids(num_levels_);
        for (int i = 0; i < num_levels_; i++){
            double a = breakpoints[i];
            double b = breakpoints[i+1];
            new_centroids[i] = conditional_expectations(a, b);
        }
        double diff = 0;
        for (int i = 0; i < num_levels_; i++){
            diff += abs(new_centroids[i] - centroids[i]);
        }
        centroids = new_centroids;
        if (diff < tol_) break;
    }
    centroids_ = centroids;
    breakpoints_ = update_breakpoints(centroids);
}

int LloydMax::quantize(double x){
    int val = binary_search(breakpoints_, x); 
    return val; 
}

double LloydMax::deQuantize(int index){
    if (index < 0 || index >= num_levels_) throw std::out_of_range("Bin value/index out of range"); 
    return centroids_[index]; 
}

vector<int> LloydMax::quantize_vector(const vector<double>& data){
    vector<int> quantized_data_index; 
    for (double x: data){
        int index = quantize(x); 
        quantized_data_index.push_back(index); 
    }
    return quantized_data_index;
}

vector<double> LloydMax::deQuantize_vector(const vector<int>& indexes){
    vector<double> centroids_vectors; 
    for (int index : indexes){
        double centroid = deQuantize(index); 
        centroids_vectors.push_back(centroid); 
    
    }
    return centroids_vectors; 
}


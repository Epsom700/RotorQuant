#pragma once
#include <vector> 
#include <string> 
#include <algorithm> 
#include <cmath> 


class Rotation{
    public: 
        Rotation(int n); 
        ~Rotation() = default; 
        std::vector<double> rotate(const std::vector<double>& activations);
        std::vector<double> inverse_rotate(const std::vector<double>& activations);
        const std::vector<std::vector<double>>& hadamard() const { return hadamard_; }
        const std::vector<double>& random_flips() const { return random_flips_; }

    private:
        
        std::vector<std::vector<double>> hadamard_; 
        std::vector<std::vector<double>> generate_hadamard(int n); 
        std::vector<double> random_flips_; 

        
    
    
};

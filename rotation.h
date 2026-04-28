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
        const std::vector<double>& random_flips() const { return random_flips_; }

    private:
        
        int n_; 
        static void fwht(std::vector<double>& data);
        std::vector<double> random_flips_; 

        
    
    
};

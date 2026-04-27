
#include "rotation.h"
#include <vector>
#include <iostream> 
#include <string>
#include <cmath> 
#include <algorithm> 
#include <random>
using namespace std;


Rotation::Rotation(int n)
{
    if ((n & (n-1)) !=0) throw std::invalid_argument("n must be a power of 2"); 
    hadamard_ = generate_hadamard(n); 
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0,1);
    random_flips_.resize(hadamard_.size());
    std::generate(random_flips_.begin(), random_flips_.end(), [&]() {
        return dist(gen)== 1 ? 1.0 : -1.0; //maps to -1/1; 
    });

}



vector<vector<double>> Rotation::generate_hadamard(int n){
    if (n == 1){
        return {{1.0}};

    }

    vector<vector<double>> H_prev = generate_hadamard(n/2); 
    vector<vector<double>> H(n, vector<double>(n));
    for (int i = 0; i < n/2; i++){
        for (int j = 0; j < n/2; ++j ){
            H[i][j] = H_prev[i][j] * (1/std::sqrt(2)); 
            H[i][j+n/2] = H_prev[i][j] * (1/std::sqrt(2)); 
            H[i+n/2][j] = H_prev[i][j] * (1/std::sqrt(2));
            H[i+n/2][j+n/2] = H_prev[i][j] * -(1/std::sqrt(2)); 

        }
    }
    return H; 
}


vector<double> Rotation::rotate(const vector<double>& activations){
    const auto& R = hadamard_;
    int r = R.size();
    int c = R[0].size();
    int o = activations.size();
    if (r!=o){
        throw runtime_error("The dimensions should match");
    }
    vector<double> X(c);
    for (int i = 0; i < r; i++){
        double acc = 0.0;
        const auto& Ri = R[i];
        for(int j = 0; j < c; j++){
            acc += Ri[j] * (activations[j]*random_flips_[j]);
        }
        X[i] = acc;
    }
    return X;
}

vector<double> Rotation::inverse_rotate(const vector<double>& activations){
    const auto& R = hadamard_;
    int r = R.size();
    int c = R[0].size();
    int o = activations.size();
    if (r!=o){
        throw runtime_error("The dimensions should match");
    }
    vector<double> H_t(r);
    for (int i = 0; i < c; i++){
        for (int j = 0 ; j < r ; j++){
            H_t[i] += R[j][i] * activations[j];
        }
    }
    for (int i = 0; i < r; i++){
        H_t[i] *= random_flips_[i];
    }
    return H_t;
}

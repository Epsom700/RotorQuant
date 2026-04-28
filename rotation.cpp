
#include "rotation.h"
#include <vector>
#include <iostream> 
#include <string>
#include <cmath> 
#include <algorithm> 
#include <random>
using namespace std;


Rotation::Rotation(int n)
    : n_(n)
{
    if ((n & (n-1)) !=0) throw std::invalid_argument("n must be a power of 2");  
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0,1);
    random_flips_.resize(n_);
    std::generate(random_flips_.begin(), random_flips_.end(), [&]() {
        return dist(gen)== 1 ? 1.0 : -1.0; //maps to -1/1; 
    });

}



// vector<vector<double>> Rotation::generate_hadamard(int n){
//     if (n == 1){
//         return {{1.0}};

//     }

//     vector<vector<double>> H_prev = generate_hadamard(n/2); 
//     vector<vector<double>> H(n, vector<double>(n));
//     for (int i = 0; i < n/2; i++){
//         for (int j = 0; j < n/2; ++j ){
//             H[i][j] = H_prev[i][j] * (1/std::sqrt(2)); 
//             H[i][j+n/2] = H_prev[i][j] * (1/std::sqrt(2)); 
//             H[i+n/2][j] = H_prev[i][j] * (1/std::sqrt(2));
//             H[i+n/2][j+n/2] = H_prev[i][j] * -(1/std::sqrt(2)); 

//         }
//     }
//     return H; 
// }


void Rotation::fwht(vector<double>& data){
    int n = data.size(); 
    int h = 1; 
    while (h < n){
        for (int i = 0; i < n; i += h*2){
            for (int j = i; j< i + h; j++){
                double x = data[j]; 
                double y = data[j+h]; 
                data[j] = x+y; 
                data[j+h] = x-y; 
            }
        }
        h*=2; 
    }
    for (double& v : data) v /= std::sqrt((double)n);
}

vector<double> Rotation::rotate(const vector<double>& activations){
    int a_size = activations.size(); 
    if (a_size != n_) throw runtime_error("activation size should be valid"); 
    vector<double> flipped(a_size);
    for (int i = 0; i < a_size; i++){
        flipped[i] = activations[i] * random_flips_[i]; 
    }
    fwht(flipped);
    return flipped;
}

vector<double> Rotation::inverse_rotate(const vector<double>& activations){
    int a_size = activations.size(); 
    if (a_size != n_) throw runtime_error("activation size should be valid"); 
    vector<double> temp = activations;
    fwht(temp); //in-place inverse FWHT
    vector<double> iflipped(a_size);
    for (int i = 0; i < a_size; i++){
        iflipped[i] = temp[i] * random_flips_[i]; 
    }
    return iflipped;
}

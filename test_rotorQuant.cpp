#include "rotorQuant.h"
#include <vector>
#include <iostream>
#include <cmath> 
using namespace std; 

double compute_mse(const vector<double>& original, const vector<double>& reconstructed){
    double mse = 0; 
    for (int i = 0; i < original.size(); i++){
        mse += pow((original[i] - reconstructed[i]), 2); 
    }
    return mse / original.size(); 
}

int main(){
    // 1D Test
    cout << "=== 1D Test ===" << endl;
    vector<double> activations = {0.5, -1.2, 0.8, -0.3, 1.5, -0.7, 0.2, -1.0};
    RotorQuant rq(8, 8, 1.0); 

    vector<int> binned = rq.encode(activations); 
    vector<double> reconstructed = rq.decode(binned); 

    cout << "Original:      ";
    for (double v : activations) cout << v << " ";
    cout << endl;

    cout << "Reconstructed: ";
    for (double v : reconstructed) cout << v << " ";
    cout << endl;

    cout << "Bin indices:   ";
    for (int i : binned) cout << i << " ";
    cout << endl;

    cout << "MSE: " << compute_mse(activations, reconstructed) << endl;
    cout << "Original size: " << activations.size() * sizeof(double) << " bytes" << endl;
    cout << "Encoded size:  " << binned.size() * sizeof(int) << " bytes" << endl;

    // 2D Test
    cout << "\n=== 2D Test ===" << endl;
    vector<vector<double>> activations_2d = {
        {0.5, -1.2,  0.8, -0.3,  1.5, -0.7,  0.2, -1.0},
        {1.1, -0.4,  0.6, -1.3,  0.3, -0.9,  1.4, -0.2},
        {-0.8, 0.7, -1.1,  0.4, -0.5,  1.2, -0.3,  0.9}
    };

    vector<vector<int>> binned_2d = rq.encode_2d(activations_2d);
    vector<vector<double>> reconstructed_2d = rq.decode_2d(binned_2d);

    double total_mse = 0;
    for (int i = 0; i < activations_2d.size(); i++){
        double row_mse = compute_mse(activations_2d[i], reconstructed_2d[i]);
        total_mse += row_mse;
        cout << "Row " << i << " MSE: " << row_mse << endl;
    }
    cout << "Average MSE: " << total_mse / activations_2d.size() << endl;

    // Memory footprint
    int original_bytes = 0;
    int encoded_bytes = 0;
    for (int i = 0; i < activations_2d.size(); i++){
        original_bytes += activations_2d[i].size() * sizeof(double);
        encoded_bytes += binned_2d[i].size() * sizeof(int);
    }
    cout << "Original size: " << original_bytes << " bytes" << endl;
    cout << "Encoded size:  " << encoded_bytes << " bytes" << endl;
    cout << "Reduction:     " << (1.0 - (double)encoded_bytes/original_bytes) * 100 << "%" << endl;

    return 0; 
}
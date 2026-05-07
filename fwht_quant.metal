#include <metal_stdlib>
using namespace metal;

kernel void fwht_quant(
    device float* data        [[ buffer(0) ]],
    device const float* flips [[ buffer(1) ]],
    device const float* bp    [[ buffer(2) ]],
    device const float* cent  [[ buffer(3) ]],
    constant int& L           [[ buffer(4) ]],
    constant int& cols        [[ buffer(5) ]],
    uint2 tgid [[ threadgroup_position_in_grid ]],
    uint2 tid2 [[ thread_position_in_threadgroup ]], 
    uint  lane_id [[ thread_index_in_simdgroup ]])
{
    uint tid = tid2.x;
    uint row_base = tgid.x * cols;
    float x[8];

    // 1. load + apply flips
    for (uint i = 0; i < 8; i++) {
        uint g = tid * 8 + i;
        x[i] = data[row_base + g] * flips[g];
        
    }

    for (uint i =0; i< 7; i+=2){
        float t0 = x[i], t1 = x[i+1]; 
        x[i] = t0 + t1;
        x[i+1] = t0 - t1;
    }
    for (uint base = 0; base < 8; base +=4){
        for (uint j = 0; j < 2; j++){
            float t0 = x[base + j], t1 = x[base + j + 2];
            x[base + j] = t0 + t1;
            x[base + j + 2] = t0 - t1;
        }
    }
    for (uint i = 0; i < 4; i++){
        float t0 = x[i], t1 = x[i + 4];
        x[i] = t0 + t1;
        x[i + 4] = t0 - t1;
    }

    for (uint i = 0; i < 8; i++){
        data[row_base + tid * 8 + i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_device);
    // 2. forward FWHT

    for (uint len = 8; len <= 128; len *= 2) {
        uint stride = len / 8;
        uint k = lane_id;  // ← fixed
        bool lower = (k % (2 * stride)) < stride;
        uint partner_lane = lower ? (k + stride) : (k - stride);

        float new_x[8];
        for (uint i = 0; i < 8; i++) {
            float partner_val = simd_shuffle(x[i], partner_lane);
            new_x[i] = lower ? (x[i] + partner_val) : (partner_val - x[i]);
        }
        for (uint i = 0; i < 8; i++) x[i] = new_x[i];
    }

    // After len = 128, x[] holds the right values in registers, but device memory is stale.
    // We need to write back before the cross-SIMD stages can read.
    for (uint i = 0; i < 8; i++)
        data[row_base + tid * 8 + i] = x[i];
    threadgroup_barrier(mem_flags::mem_device);

    for (uint len = 256; len < (uint)cols; len *= 2) {
        // Phase A: read old data, compute new values into registers
        for (uint i = 0; i < 8; i++) {
            uint g = tid * 8 + i;
            uint pos_in_block = g % (2 * len);
            if (pos_in_block < len) {
                float u = data[row_base + g];
                float v = data[row_base + g + len];
                x[i] = u + v;
            } else {
                float u = data[row_base + g - len];
                float v = data[row_base + g];
                x[i] = u - v;
            }
        }
        // *** FIX: ensure ALL threads finished reading before ANY thread writes ***
        threadgroup_barrier(mem_flags::mem_device);

        // Phase B: write computed values back
        for (uint i = 0; i < 8; i++)
            data[row_base + tid * 8 + i] = x[i];
        threadgroup_barrier(mem_flags::mem_device);
    }

    // 3. normalize
    float scale = 1.0f / sqrt((float)cols);
    for (uint i = 0; i < 8; i++) x[i] *= scale;


    // 4. quantize
    for (uint i = 0; i < 8; i++) {
        float v = x[i];
        int lo = 0, hi = L - 1;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (bp[mid + 1] > v) hi = mid;
            else lo = mid + 1;
        }
        x[i] = cent[lo];
    }

    // 5. write quantized back (x already matches what we just wrote, no reload needed)    

    for (uint i =0; i< 7; i+=2){
        float t0 = x[i], t1 = x[i+1]; 
        x[i] = t0 + t1;
        x[i+1] = t0 - t1;
    }
    for (uint base = 0; base < 8; base +=4){
        for (uint j = 0; j < 2; j++){
            float t0 = x[base + j], t1 = x[base + j + 2];
            x[base + j] = t0 + t1;
            x[base + j + 2] = t0 - t1;
        }
    }
    for (uint i = 0; i < 4; i++){
        float t0 = x[i], t1 = x[i + 4];
        x[i] = t0 + t1;
        x[i + 4] = t0 - t1;
    }

    for (uint i = 0; i < 8; i++){
        data[row_base + tid * 8 + i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_device); 


    for (uint len = 8; len <= 128; len *= 2) {
        uint stride = len / 8;
        uint k = lane_id;  // ← fixed
        bool lower = (k % (2 * stride)) < stride;
        uint partner_lane = lower ? (k + stride) : (k - stride);

        float new_x[8];
        for (uint i = 0; i < 8; i++) {
            float partner_val = simd_shuffle(x[i], partner_lane);
            new_x[i] = lower ? (x[i] + partner_val) : (partner_val - x[i]);
        }
        for (uint i = 0; i < 8; i++) x[i] = new_x[i];
    }

    for (uint i = 0; i < 8; i++){
        data[row_base + tid * 8 + i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_device); 

    // 6. inverse FWHT (same race fix as forward)
    for (uint len = 256; len < (uint)cols; len *=2 ) {
        // Phase A: read + compute
        for (uint i = 0; i < 8; i++) {
            uint g = tid * 8 + i;
            uint pos_in_block = g % (2 * len);
            if (pos_in_block < len) {
                float u = data[row_base + g];
                float v = data[row_base + g + len];
                x[i] = u + v;
            } else {
                float u = data[row_base + g - len];
                float v = data[row_base + g];
                x[i] = u - v;
            }
        }
        // *** FIX: same barrier as forward ***
        threadgroup_barrier(mem_flags::mem_device);

        // Phase B: write
        for (uint i = 0; i < 8; i++)
            data[row_base + tid * 8 + i] = x[i];
        threadgroup_barrier(mem_flags::mem_device);
    }

    // 7. normalize + inverse flips + write back
    float scale2 = 1.0f / sqrt((float)cols);
    for (uint i = 0; i < 8; i++) {
        uint g = tid * 8 + i;
        x[i] *= scale2 * flips[g];
        data[row_base + g] = x[i];
    }
}

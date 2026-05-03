#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <torch/extension.h>
#include <ATen/mps/MPSStream.h>

// Persistent Metal objects — created once, reused every call
static id<MTLDevice>              _device     = nil;
static id<MTLComputePipelineState> _pipeline  = nil;
static id<MTLLibrary>             _library    = nil;

static void initMetal(){
    if (_pipeline) return ; 
    _device = MTLCreateSystemDefaultDevice(); 
    NSString* libPath = @"fwht_quant.metallib"; 
    NSError* err = nil; 
    _library = [_device newLibraryWithURL: [NSURL fileURLWithPath:libPath] error: &err]; 
    if (!_library){
        NSLog(@"Failed to create Metal library: %@", err);
        return;
    }
    id<MTLFunction> func = [_library newFunctionWithName: @"fwht_quant"]; 
    _pipeline = [_device newComputePipelineStateWithFunction: func error: &err];
    if (!_pipeline){
        NSLog(@"Failed to create Metal pipeline: %@", err);
        return;
    }
}

void fwht_quant_metal(torch::Tensor data, 
                     torch::Tensor flips, 
                     torch::Tensor bp, 
                     torch::Tensor cent, 
                     int L, 
                     int cols)                     
{
    initMetal(); 
    auto stream = at::mps::getCurrentMPSStream(); 
    id<MTLCommandBuffer> cmdBuf = stream -> commandBuffer(); 

    //Get raw metal buffer pointers from the pyTorch tensors; 
    id<MTLBuffer> dataBuf = (id<MTLBuffer>)data.storage().data(); 
    id<MTLBuffer> flipsBuf = (id<MTLBuffer>)flips.storage().data(); 
    id<MTLBuffer> bpBuf = (id<MTLBuffer>)bp.storage().data(); 
    id<MTLBuffer> centBuf = (id<MTLBuffer>)cent.storage().data(); 


    // Create Compute encoder
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder]; 
    [enc setComputePipelineState: _pipeline]; 

    // Bind Buffers;
    [enc setBuffer:dataBuf offset:0 atIndex:0]; 
    [enc setBuffer:flipsBuf offset:0 atIndex:1]; 
    [enc setBuffer:bpBuf offset:0 atIndex:2]; 
    [enc setBuffer:centBuf offset:0 atIndex:3]; 


    // Bind Scalar constants
    [enc setBytes: &L length:sizeof(int) atIndex:4]; 
    [enc setBytes: &cols length:sizeof(int) atIndex:5]; 

    //Dispatch - one threadgroup per row, 1024 threads per group

    MTLSize grid = MTLSizeMake(data.size(0), 1, 1); 
    MTLSize groupSize = MTLSizeMake(1024, 1, 1); 
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:groupSize]; 
    [enc endEncoding]; 

    stream->synchronize(at::mps::SyncType::COMMIT_AND_WAIT); 
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fwht_quant_metal", &fwht_quant_metal, "FWHT quantize via Metal");
}
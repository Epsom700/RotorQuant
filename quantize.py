import numpy as np
import torch
import rotorquant
import math

class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, rq):
        original_dtype = x.dtype
        original_shape = x.shape
        original_device = x.device
        D = original_shape[-1]
        # bf16/fp16 -> fp32 numpy on CPU; numpy() requires CPU+contig
        arr = (
            x.detach()
             .to(dtype=torch.float32, device="cpu")
             .reshape(-1, D)
             .contiguous()
             .numpy()
        )
        # M4-optimized: batched Accelerate (sgemm) + bucketize, in-place on f32 buffer
        rq.encode_decode_batch_f32(arr)
        out = torch.from_numpy(arr).to(device=original_device, dtype=original_dtype)
        return out.reshape(original_shape)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None  # straight-through estimator


class RotorQuantLayer(torch.nn.Module):
    def __init__(self, layer, actual_dim, num_levels, sigma):
        super().__init__()
        print(f"Initializing RotorQuantLayer with n={actual_dim}, num_levels={num_levels}, sigma={sigma}")
        self.padded_dim = 2**math.ceil(math.log2(actual_dim))
        self.actual_dim = actual_dim
        self.pad_size = self.padded_dim - self.actual_dim
        self.rq = rotorquant.RotorQuant(self.padded_dim, num_levels, sigma)
        self.layer = layer

    def forward(self, x):
        x = self.layer(x)
        if self.pad_size > 0:
            x = torch.nn.functional.pad(x, (0, self.pad_size))  # then pad
        x = STEQuantize.apply(x, self.rq)  # quantize padded
        if self.pad_size > 0:
            x = x[..., :self.actual_dim]

        return x



def inject_rotorquant(model, num_levels, sigma):
    act_dim = model.config.intermediate_size
    for i in range(len(model.model.layers)):
        original_act = model.model.layers[i].mlp.act_fn
        model.model.layers[i].mlp.act_fn = RotorQuantLayer(original_act, act_dim, num_levels, sigma)
    return model
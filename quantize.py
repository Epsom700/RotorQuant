import numpy as np
import torch
import rotorquant
import fwht_metal
import math
import time
import csv
import os
import logging
import psutil

logging.basicConfig(level=logging.WARNING)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # allow multiple C++ libs (e.g. pybind11 + OpenMP) to coexist without segfaults

class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, rq_tensors):
        original_dtype = x.dtype
        original_shape = x.shape
        D = original_shape[-1]
        # bf16/fp16 -> fp32 numpy on CPU; numpy() requires CPU+contig
        timings = {}

        proc = psutil.Process()
        t0 = time.time()
        x_detached = x.detach()
        timings['detach'] = time.time() - t0
        mems = {}
        mems['detach'] = proc.memory_info().rss / 1024**3

        t1 = time.time()
        x_cpu_f32 = x_detached.to(dtype=torch.float32)
        timings['to_cpu_f32'] = time.time() - t1
        mems['to_cpu_f32'] = proc.memory_info().rss / 1024**3

        t2 = time.time()
        contig = x_cpu_f32.contiguous()
        timings['contiguous'] = time.time() - t2
        mems['contiguous'] = proc.memory_info().rss / 1024**3

        t3 = time.time()
        flat = contig.reshape(-1, D)
        timings['reshape'] = time.time() - t3
        mems['reshape'] = proc.memory_info().rss / 1024**3

        t4 = time.time()
        fwht_metal.fwht_quant_metal(
            flat, 
            rq_tensors['flips'], 
            rq_tensors['bp'], 
            rq_tensors['cent'], 
            rq_tensors['L'], 
            D
        )
        timings['cpp_roundtrip'] = time.time() - t4
        mems['cpp_roundtrip'] = proc.memory_info().rss / 1024**3

        total = sum(timings.values())
        timings['total'] = total

        # Record per-step timings (seconds). Use DEBUG level so they don't pollute stdout
        logging.debug("Quantize timings (s): " + ", ".join(f"{k}: {v:.6f}" for k, v in timings.items()))

        # Append timings + memory to CSV log
        log_path = "./quantize_timings.csv"
        # Fieldnames should reflect the actual measured keys in this function
        timing_keys = ["detach", "to_cpu_f32", "contiguous", "reshape", "cpp_roundtrip", "total"]
        mem_keys = [k + "_mem_gb" for k in ["detach", "to_cpu_f32", "contiguous", "reshape", "cpp_roundtrip"]]
        fieldnames = ["timestamp", "shape"] + timing_keys + mem_keys
        # Determine whether we need to write header. Use csv.reader to parse
        # existing header (handles quoting) and compare lists. If mismatch,
        # back up the old file and create a fresh CSV with the correct header.
        need_header = False
        try:
            if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
                need_header = True
            else:
                with open(log_path, 'r', newline='') as f:
                    reader = csv.reader(f)
                    try:
                        existing = next(reader)
                    except StopIteration:
                        existing = []
                if existing != fieldnames:
                    bak_path = log_path + '.bak'
                    try:
                        os.replace(log_path, bak_path)
                        logging.warning(f"Existing CSV header did not match expected; backed up to {bak_path} and creating fresh CSV")
                    except Exception:
                        logging.warning("Failed to back up existing CSV; will attempt to overwrite")
                    need_header = True

            # Open in write mode if we need to ensure header is the first line,
            # otherwise append.
            mode = 'w' if need_header else 'a'
            with open(log_path, mode, newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if need_header:
                    writer.writeheader()
                row = {"timestamp": time.time(), "shape": str(original_shape)}
                # fill timing fields using actual timing_keys
                for k in timing_keys:
                    row[k] = timings.get(k, float('nan'))
                # fill memory fields (in GB) using mem_keys mapping
                for k_short, k_full in zip(["detach", "to_cpu_f32", "contiguous", "reshape", "cpp_roundtrip"], mem_keys):
                    row[k_full] = mems.get(k_short, float('nan'))
                writer.writerow(row)
        except Exception as e:
            logging.warning(f"Failed to write quantize timings to CSV: {e}")

        return flat.to(dtype=original_dtype).reshape(original_shape)  # quantized output, back on original device and dtype

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None  # straight-through estimator


class RotorQuantLayer(torch.nn.Module):
    def __init__(self, layer, actual_dim, rq):
        super().__init__()
        self.actual_dim = actual_dim
        self.padded_dim = 2**math.ceil(math.log2(actual_dim))
        self.pad_size = self.padded_dim - actual_dim
        self.layer = layer

        device = torch.device("mps")
        self.rq_tensors = {
            'flips': torch.tensor(rq.flips_f32(), dtype=torch.float32, device=device),
            'bp':    torch.tensor(rq.bp_f32(),    dtype=torch.float32, device=device),
            'cent':  torch.tensor(rq.cent_f32(),  dtype=torch.float32, device=device),
            'L':     rq.num_levels()
        }

    def forward(self, x):
        x = self.layer(x)
        if self.pad_size > 0:
            x = torch.nn.functional.pad(x, (0, self.pad_size))
        x = STEQuantize.apply(x, self.rq_tensors)
        if self.pad_size > 0:
            x = x[..., :self.actual_dim]
        return x


def inject_rotorquant(model, num_levels, sigma):
    act_dim = model.config.intermediate_size
    rq = rotorquant.RotorQuant(
        2**math.ceil(math.log2(act_dim)),
        num_levels,
        sigma
    )
    for i in range(len(model.model.layers)):
        original_act = model.model.layers[i].mlp.act_fn
        model.model.layers[i].mlp.act_fn = RotorQuantLayer(
            original_act, act_dim, rq
        )
    return model
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import get_peft_model, LoraConfig
import torch
import gc
import psutil
import traceback
from quantize import inject_rotorquant

def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / 1024**3


model_name = "/Users/arjunsingh/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95"
NUM_STEPS = 100
BATCH_SIZE = 4
SEQ_LEN = 128
GRAD_ACCUM = 2


def load_fresh_model():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    m = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    return m


def run_training(use_rotorquant: bool, num_steps: int, tokenizer, dataset):
    label = "RotorQuant" if use_rotorquant else "Baseline"
    print(f"\n{'='*60}\n{label} run — loading model\n{'='*60}")

    model = load_fresh_model()
    print(f"Memory after model load: {get_memory_usage():.2f} GB")

    if use_rotorquant:
        print("Injecting RotorQuant...")
        model = inject_rotorquant(model, num_levels=8, sigma=1.0)
        print(f"Memory after RotorQuant injection: {get_memory_usage():.2f} GB")
        # sanity check first layer structure
        print(f"layer[0].mlp.act_fn = {type(model.model.layers[0].mlp.act_fn).__name__}")

    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none"
    )
    model = get_peft_model(model, lora_config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    losses = []

    # Pre-filter and pre-tokenize a pool of non-empty samples in batches.
    texts = (s["text"] for s in dataset if s["text"].strip())
    def batch_iter():
        buf = []
        for t in texts:
            buf.append(t)
            if len(buf) == BATCH_SIZE:
                yield buf
                buf = []
    batches = batch_iter()

    print(f"Starting {label} training: {num_steps} steps, batch={BATCH_SIZE}, "
          f"seq_len={SEQ_LEN}, grad_accum={GRAD_ACCUM}")

    import time
    completed = 0
    optimizer.zero_grad()
    accum_loss = 0.0
    accum_count = 0
    t0 = time.time()
    while completed < num_steps:
        try:
            batch_texts = next(batches)
        except StopIteration:
            print("Dataset exhausted")
            break

        try:
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=SEQ_LEN,
                padding="max_length",
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  non-finite loss, skipping micro-batch")
                continue

            (loss / GRAD_ACCUM).backward()
            accum_loss += loss.item()
            accum_count += 1

            if accum_count == GRAD_ACCUM:
                optimizer.step()
                optimizer.zero_grad()
                avg = accum_loss / accum_count
                losses.append(avg)
                completed += 1
                dt = time.time() - t0
                t0 = time.time()
                print(f"[{label}] step {completed}/{num_steps} | loss: {avg:.4f} "
                      f"| {dt:.2f}s | mem: {get_memory_usage():.2f} GB")
                accum_loss = 0.0
                accum_count = 0

        except Exception as e:
            print(f"[{label}] step failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            optimizer.zero_grad()
            accum_loss = 0.0
            accum_count = 0
            continue

    print(f"\n{label} done. completed={len(losses)}")

    del model, optimizer
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return losses


print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

print("Loading dataset...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

# baseline_losses = run_training(use_rotorquant=False, num_steps=NUM_STEPS, tokenizer=tokenizer, dataset=dataset)
rotorquant_losses = run_training(use_rotorquant=True, num_steps=NUM_STEPS, tokenizer=tokenizer, dataset=dataset)

print("\n" + "="*60)
print("LOSS COMPARISON")
print("="*60)
print(f"{'step':>5} | {'baseline':>10} | {'rotorquant':>10}")
print("-"*40)
n = max(len(baseline_losses), len(rotorquant_losses))
for i in range(n):
    b = f"{baseline_losses[i]:.4f}" if i < len(baseline_losses) else "  -  "
    r = f"{rotorquant_losses[i]:.4f}" if i < len(rotorquant_losses) else "  -  "
    print(f"{i+1:>5} | {b:>10} | {r:>10}")

if baseline_losses and rotorquant_losses:
    print(f"\nbaseline   avg: {sum(baseline_losses)/len(baseline_losses):.4f}")
    print(f"rotorquant avg: {sum(rotorquant_losses)/len(rotorquant_losses):.4f}")


import pandas as pd
import math
out_xlsx = "./losses_comparison.xlsx"
out_csv = "./losses_comparison.csv"
try:
    max_len = max(len(baseline_losses), len(rotorquant_losses))
    b = baseline_losses + [math.nan] * (max_len - len(baseline_losses))
    r = rotorquant_losses + [math.nan] * (max_len - len(rotorquant_losses))
    df = pd.DataFrame({"baseline_losses": b, "rotorquant_losses": r})
    df.to_excel(out_xlsx, index=False)
    print(f"Saved losses to {out_xlsx}")
except Exception as e:
    try:
        df.to_csv(out_csv, index=False)
        print(f"Saved CSV to {out_csv} due to Excel write error: {e}")
    except Exception as e2:
        print(f"Failed to save losses to Excel or CSV: {e2}")
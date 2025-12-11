#!/usr/bin/env python3
"""
VecP PHASE 4: LATENCY BENCHMARK
===============================

Measures the "Safety Tax" - the latency cost of the VecP stack
compared to raw generation.

Patent Pending: USPTO 63/931,565
Copyright (c) 2025 David Cappelli / VecP Labs
"""

import time
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList

from vecp_unified_threat import (
    UnifiedThreatGenerator, 
    UnifiedThreatProcessor,
    STEWARD_SEED,
    HARM_VETO_THRESHOLD,
)

# =============================================================================
# BENCHMARK CONFIGURATION
# =============================================================================

BENCHMARK_PROMPTS = [
    # Short creative
    ("Write a haiku about mountains.", 50),
    # Medium informational
    ("Explain how photosynthesis works in simple terms.", 100),
    # Longer creative
    ("Write a short story about a robot learning to paint.", 150),
]

# =============================================================================
# RAW BASELINE (No VecP)
# =============================================================================

def run_raw_baseline(model, tokenizer, prompt: str, max_tokens: int) -> dict:
    """Generate without any VecP processing - pure baseline."""
    full_prompt = f"User: {prompt}\n\nAssistant:"
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    
    start_time = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            # NO logits_processor - raw generation
        )
    end_time = time.time()
    
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if "Assistant:" in response:
        response = response.split("Assistant:")[-1].strip()
    
    tokens_generated = output_ids.shape[1] - inputs.input_ids.shape[1]
    duration = end_time - start_time
    
    return {
        "tokens": tokens_generated,
        "duration": duration,
        "tps": tokens_generated / duration if duration > 0 else 0,
        "response": response[:100] + "..." if len(response) > 100 else response,
    }


# =============================================================================
# VECP STACK (Full Protection)
# =============================================================================

def run_vecp_stack(gen: UnifiedThreatGenerator, prompt: str, max_tokens: int) -> dict:
    """Generate with full VecP Input Gate + Output Gate."""
    start_time = time.time()
    result = gen.generate(prompt, STEWARD_SEED, max_tokens=max_tokens, verbose=False)
    end_time = time.time()
    
    response = result["response"]
    tokens_generated = len(gen.tokenizer.encode(response))
    duration = end_time - start_time
    
    return {
        "tokens": tokens_generated,
        "duration": duration,
        "tps": tokens_generated / duration if duration > 0 else 0,
        "response": response[:100] + "..." if len(response) > 100 else response,
        "mode": result["report"].get("input_classification", {}).get("mode", "unknown"),
        "interventions": result["report"].get("interventions", 0),
    }


# =============================================================================
# MAIN BENCHMARK
# =============================================================================

def run_benchmark(model_name: str = "Qwen/Qwen2.5-1.5B", safety_matrix_path: str = "safety_matrix.pt"):
    print("=" * 70)
    print("  🚀 VecP PHASE 4: LATENCY BENCHMARK")
    print("=" * 70)
    
    # Initialize VecP generator (this also loads the model)
    print("\nInitializing VecP stack...")
    gen = UnifiedThreatGenerator(model_name, safety_matrix_path)
    
    # Get raw model reference for baseline tests
    model = gen.model
    tokenizer = gen.tokenizer
    
    # Warmup
    print("\n⏳ Warmup run...")
    gen.generate("Hello", STEWARD_SEED, max_tokens=10)
    
    print("\n" + "=" * 70)
    print("  BENCHMARK RESULTS")
    print("=" * 70)
    
    results = []
    
    for prompt, max_tokens in BENCHMARK_PROMPTS:
        print(f"\n📝 Prompt: \"{prompt[:50]}{'...' if len(prompt) > 50 else ''}\"")
        print(f"   Max tokens: {max_tokens}")
        print("-" * 50)
        
        # Run RAW baseline
        raw = run_raw_baseline(model, tokenizer, prompt, max_tokens)
        print(f"   RAW (no VecP):  {raw['duration']:.2f}s | {raw['tokens']} tokens | {raw['tps']:.1f} tok/s")
        
        # Run VecP stack
        vecp = run_vecp_stack(gen, prompt, max_tokens)
        print(f"   VECP (full):    {vecp['duration']:.2f}s | {vecp['tokens']} tokens | {vecp['tps']:.1f} tok/s")
        
        # Calculate overhead
        if raw['duration'] > 0:
            overhead_pct = ((vecp['duration'] - raw['duration']) / raw['duration']) * 100
            overhead_abs = vecp['duration'] - raw['duration']
        else:
            overhead_pct = 0
            overhead_abs = 0
        
        print(f"   ─────────────────────────────")
        print(f"   SAFETY TAX:     +{overhead_abs:.2f}s ({overhead_pct:+.1f}%)")
        
        results.append({
            "prompt": prompt,
            "max_tokens": max_tokens,
            "raw": raw,
            "vecp": vecp,
            "overhead_pct": overhead_pct,
            "overhead_abs": overhead_abs,
        })
    
    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    
    avg_raw_tps = sum(r["raw"]["tps"] for r in results) / len(results)
    avg_vecp_tps = sum(r["vecp"]["tps"] for r in results) / len(results)
    avg_overhead = sum(r["overhead_pct"] for r in results) / len(results)
    
    print(f"""
    Average RAW speed:    {avg_raw_tps:.1f} tokens/sec
    Average VecP speed:   {avg_vecp_tps:.1f} tokens/sec
    Average Safety Tax:   {avg_overhead:+.1f}%
    
    Verdict: {"✅ ACCEPTABLE" if avg_overhead < 50 else "⚠️ NEEDS OPTIMIZATION"} for consumer use
    """)
    
    # Hardware info
    print("=" * 70)
    print("  HARDWARE")
    print("=" * 70)
    if torch.cuda.is_available():
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
        print(f"    VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("    Running on CPU")
    
    return results


# =============================================================================
# EXTENDED BENCHMARK (Multiple Runs)
# =============================================================================

def run_extended_benchmark(model_name: str, safety_matrix_path: str, runs: int = 3):
    """Run multiple iterations for more stable measurements."""
    print("=" * 70)
    print(f"  🚀 VecP EXTENDED BENCHMARK ({runs} runs)")
    print("=" * 70)
    
    gen = UnifiedThreatGenerator(model_name, safety_matrix_path)
    model = gen.model
    tokenizer = gen.tokenizer
    
    # Single test prompt
    prompt = "Write a short story about a robot learning to paint."
    max_tokens = 100
    
    print(f"\nPrompt: \"{prompt}\"")
    print(f"Tokens: {max_tokens}")
    print(f"Runs: {runs}")
    
    # Warmup
    print("\n⏳ Warmup...")
    gen.generate("Hello", STEWARD_SEED, max_tokens=10)
    
    raw_times = []
    vecp_times = []
    
    print("\n📊 Running benchmark...")
    for i in range(runs):
        # RAW
        raw = run_raw_baseline(model, tokenizer, prompt, max_tokens)
        raw_times.append(raw['duration'])
        
        # VecP
        vecp = run_vecp_stack(gen, prompt, max_tokens)
        vecp_times.append(vecp['duration'])
        
        print(f"   Run {i+1}: RAW={raw['duration']:.2f}s, VecP={vecp['duration']:.2f}s")
    
    avg_raw = sum(raw_times) / len(raw_times)
    avg_vecp = sum(vecp_times) / len(vecp_times)
    overhead = ((avg_vecp - avg_raw) / avg_raw) * 100 if avg_raw > 0 else 0
    
    print("\n" + "=" * 70)
    print("  EXTENDED RESULTS")
    print("=" * 70)
    print(f"""
    RAW Average:    {avg_raw:.2f}s (±{max(raw_times) - min(raw_times):.2f}s)
    VecP Average:   {avg_vecp:.2f}s (±{max(vecp_times) - min(vecp_times):.2f}s)
    Safety Tax:     {overhead:+.1f}%
    """)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VecP Phase 4 Latency Benchmark")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--safety-matrix", default="safety_matrix.pt")
    parser.add_argument("--extended", action="store_true", help="Run extended benchmark with multiple iterations")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs for extended benchmark")
    args = parser.parse_args()
    
    if args.extended:
        run_extended_benchmark(args.model, args.safety_matrix, args.runs)
    else:
        run_benchmark(args.model, args.safety_matrix)

#!/usr/bin/env python3
"""
Advanced PRNG seed finder script (Improved Version)

Improvements implemented:
1. Visual progress bar with better error handling
2. Robust argument validation
3. Detailed documentation
4. Resumable search with safe state management
5. Cleaner and organized output
6. Backend compatibility checks
7. Batch processing for verification phase
"""

import sys
import os
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from time import time
from math import ceil

try:
    from tqdm import tqdm
except ImportError:
    print("Install tqdm for better progress bars: pip install tqdm")
    tqdm = lambda x, *args, **kwargs: x  # Fallback implementation

# Argument parser configuration
def setup_parser():
    parser = ArgumentParser(
        description="Custom PRNG seed finder",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  python prng_finder.py 5090412420734391115 4310087371899710319 --cpu
  python prng_finder.py V1 V2 --resume saved_state.npz --step 5000000
        """
    )
    
    parser.add_argument('V1', type=int, help='First value in sequence')
    parser.add_argument('V2', type=int, help='Second value in sequence')
    parser.add_argument('--cpu', action='store_true', help='Use CPU (numpy) instead of GPU')
    parser.add_argument('--step', type=int, default=10_000_000, 
                       help='Search block size (default: 10,000,000)')
    parser.add_argument('--resume', type=str, 
                       help='State file to resume interrupted search')
    parser.add_argument('--save', type=str, 
                       help='Periodically save state to this file')
    parser.add_argument('--verbose', action='store_true', help='Verbose mode')
    
    return parser

# Configure backend (CPU/GPU) with safety checks
def setup_backend(use_cpu):
    """Configure and validate the computational backend"""
    if use_cpu:
        import numpy as np
        backend = "numpy"
    else:
        try:
            import cupy as np
            # Test basic GPU functionality
            test_array = np.array([1, 2, 3], dtype=np.ulonglong)
            test_array + 1  # Simple operation to verify GPU
            backend = "cupy"
        except Exception as e:
            print(f"⚠️ GPU error: {e}, falling back to CPU")
            import numpy as np
            backend = "numpy (fallback)"
    
    if '--verbose' in sys.argv:
        print(f"ℹ️ Using {backend}")
    return np

# -------- npRNG CLASS (Enhanced with Safety Checks) --------
class npRNG:
    """
    Vectorized implementation of a bitwise-operation based PRNG.
    
    The algorithm uses a series of shifts and XOR operations to generate
    pseudorandom numbers from initial seeds.
    
    Attributes:
        states (array): Internal PRNG state for multiple seeds
        cur (array): Temporary value used in calculations
    """
    
    def __init__(self, seeds):
        """
        Initialize the PRNG with an array of seeds.
        
        Args:
            seeds (array): Array of initial seeds (can be vectorized)
        """
        self.states = np.array([0] * len(seeds), dtype=np.ulonglong)
        for _ in range(16):
            self.cur = np.bitwise_and(seeds, 3)
            seeds = np.right_shift(seeds, 2)
            self.states = np.bitwise_or(
                np.left_shift(self.states, 4),
                np.bitwise_xor(np.bitwise_and(self.states, 3), self.cur)
            )
            self.states = np.bitwise_or(self.states, np.left_shift(self.cur, 2))

    def get_states(self):
        """Return current PRNG state"""
        return self.states

    def next(self, bits):
        """
        Generate next pseudorandom number with specified bit length.
        
        Args:
            bits (int): Number of bits for the generated value
            
        Returns:
            array: Generated values for all seeds
        """
        ret = np.array([0] * len(self.states), dtype=np.ulonglong)
        for _ in range(bits):
            ret = np.left_shift(ret, 1)
            ret = np.bitwise_or(ret, np.bitwise_and(self.states, 1))
            for _ in range(3):
                self.states = np.bitwise_xor(
                    np.left_shift(self.states, 1),
                    np.right_shift(self.states, 61)
                )
                self.states = np.bitwise_and(self.states, 0xFFFFFFFFFFFFFFFF)
                self.states = np.bitwise_xor(self.states, 0xFFFFFFFFFFFFFFFF)
                for j in range(0, 64, 4):
                    self.cur = np.bitwise_and(np.right_shift(self.states, j), 0xF)
                    a = np.bitwise_or(
                        np.right_shift(self.cur, 3),
                        np.bitwise_and(np.right_shift(self.cur, 2), 2)
                    )
                    b = np.bitwise_and(np.left_shift(self.cur, 3), 8)
                    c = np.bitwise_and(np.left_shift(self.cur, 2), 4)
                    self.cur = np.bitwise_or(np.bitwise_or(a, b), c)
                    self.states = np.bitwise_xor(self.states, np.left_shift(self.cur, j))
        return ret

    def next2(self, bits):
        """Convenience method to call next() twice"""
        self.next(bits)
        return self.next(bits)

# -------- Enhanced Helper Functions --------
def save_state(filename, candidates, current_seed, args, np):
    """Save current search state with safety checks"""
    try:
        
        if hasattr(candidates, '__cuda_array_interface__'): 
            candidates = np.asnumpy(candidates) 
        np.savez(filename, candidates=candidates, current_seed=current_seed)
        if args.verbose:
            print(f"💾 State saved to {filename}")
    except Exception as e:
        print(f"⚠️ Error saving state: {e}")

def load_state(filename, args, np):
    """Load previously saved search state with validation"""
    try:
        if not os.path.exists(filename):
            raise FileNotFoundError(f"State file not found: {filename}")
        
        loaded = np.load(filename)
        candidates = loaded['candidates']
        if hasattr(candidates, 'tolist'):
            candidates = candidates.tolist()
        current_seed = int(loaded['current_seed'])
        
        if args.verbose:
            print(f"♻️ State loaded from {filename}")
            print(f"   Current seed: {current_seed}")
            print(f"   Found candidates: {len(candidates)}")
        
        return candidates, current_seed
    except Exception as e:
        print(f"⚠️ Error loading state: {e}")
        raise

def print_banner(args):
    """Display startup information"""
    print("\n" + "="*60)
    print(f"🔍 Starting PRNG seed search".center(60))
    print(f"V1: {args.V1}".center(60))
    print(f"V2: {args.V2}".center(60))
    print(f"Using {'CPU (numpy)' if args.cpu else 'GPU (cupy)'}".center(60))
    print("="*60 + "\n")

# -------- MAIN LOGIC (Improved Error Handling) --------
if __name__ == "__main__":
    # Initial setup with error handling
    try:
        parser = setup_parser()
        args = parser.parse_args()
        np = setup_backend(args.cpu)
        
        print_banner(args)
        
        SEED_MAX = 0xFFFFFFFF
        STEP = args.step
        candidates = []
        start_seed = 0
        
        # Handle resuming interrupted search
        if args.resume:
            candidates, start_seed = load_state(args.resume, args, np)
            print(f"🔄 Resuming search from seed {start_seed}")
        else:
            candidates = []
            start_seed = 0
        start_time = time()
        
        # Phase 1: Initial candidate search with batch processing
        print("🔎 Phase 1: Finding initial candidates...")
        s = start_seed  # Ensure 's' is defined for exception handling
        try:
            total_blocks = ceil((SEED_MAX - start_seed) / STEP)
            progress_bar = tqdm(
                range(start_seed, SEED_MAX, STEP),
                desc="Progress",
                unit="block",
                unit_scale=STEP,
                total=total_blocks,
                initial=start_seed // STEP,  # Start progress bar in correct resume
            )
    
            for s in progress_bar:
                seeds = np.arange(s, min(s + STEP, SEED_MAX), dtype=np.ulonglong)
                rng = npRNG(seeds)
                values = rng.next(8)
                matches = np.where(values == (args.V1 >> 56))[0]
            
                
                if len(matches) > 0:
                    if args.verbose:
                        print(f"\n🎯 {len(matches)} candidates found in block {s}-{s+STEP}")
                    candidates.extend(seeds[matches].tolist())
                
                # Periodic state saving if requested
                if args.save and s % (10 * STEP) == 0:
                    save_state(args.save, candidates, s + STEP, args, np)  # Save next block to check
                if args.resume and s % (10 * STEP) == 0:
                    save_state(args.resume, candidates, s + STEP, args, np)  # Save next block to check    
                    
        except KeyboardInterrupt:
            print("\n⏹ Search interrupted by user")
            if args.save:
                save_state(args.save, candidates, s, args, np)
                print(f"\U0001f4be State saved to {args.save} for later resumption")
            if args.resume:
                save_state(args.resume, candidates, s, args, np)
                print(f"\U0001f4be State saved to {args.resume} for later resumption")    
            sys.exit(0)

        print("\n\U0001f50d Phase 2: Verifying candidates...")
        if not candidates:
            print("❌ No candidates found in phase 1.")
            sys.exit(1)

        try:
            BATCH_SIZE = 10000000
            for i in tqdm(range(0, len(candidates), BATCH_SIZE),
                         desc="Verifying", unit="batch"):
                batch = candidates[i:i+BATCH_SIZE]
                if hasattr(batch, 'tolist'):
                    batch = batch.tolist()
                seeds = np.array(batch, dtype=np.ulonglong)
                rng = npRNG(seeds)

                v1_checks = rng.next(64)
                v2_checks = rng.next(64)

                matches = np.where((v1_checks == args.V1) & (v2_checks == args.V2))[0].tolist()
                if len(matches) > 0:
                    seed_value = int(batch[matches[0]])
                    elapsed = time() - start_time
                    print(f"\n\U0001f389 Seed found after {elapsed:.2f} seconds!")
                    print(f"🔑 Seed: {batch[matches[0]]}")
                    seed = batch[matches[0]]
                    seeds = np.array([seed], dtype=np.ulonglong)
                    rng = npRNG(seeds)
    
                    print(f"\U0001f52e value (V1): {rng.next(64)[0]}")
                    print(f"\U0001f52e value (V2): {rng.next(64)[0]}")
                    print(f"\U0001f52e Next value (V3): {rng.next(64)[0]}")
                    if args.save and os.path.exists(args.save):
                        os.remove(args.save)
                     
                       
            sys.exit(0)

        except KeyboardInterrupt:
            print("\n⏹ Verification interrupted by user")
            sys.exit(0)

        elapsed = time() - start_time
        print(f"\n⏱ Search completed in {elapsed:.2f} seconds")
        print("❌ No valid seed found for the given sequence")
        if args.save and os.path.exists(args.save):
            os.remove(args.save)

    except Exception as e:
        print(f"\u26a0\ufe0f Fatal error: {e}")
        sys.exit(1)


from z3 import *

def setup(seed):
    state = 0
    for _ in range(16):
        cur = seed & 3
        seed >>= 2
        state = (state << 4) | ((state & 3) ^ cur)
        state |= cur << 2
    return state

def prng_next(state, bits):
    for _ in range(bits):
        ret = state & 1
        state = (state << 1) ^ (state >> 61)
        state &= 0xFFFFFFFFFFFFFFFF
        state ^= 0xFFFFFFFFFFFFFFFF

        for j in range(0, 64, 4):
            cur = (state >> j) & 0xF
            cur = (cur >> 3) | ((cur >> 2) & 2) | ((cur << 3) & 8) | ((cur << 2) & 4)
            state ^= cur << j
    return state

def generate_output(seed, steps):
    state = setup(seed)
    outputs = []
    for _ in range(steps):
        val = 0
        for _ in range(26):
            val <<= 1
            val |= state & 1
            state = (state << 1) ^ (state >> 61)
            state &= 0xFFFFFFFFFFFFFFFF
            state ^= 0xFFFFFFFFFFFFFFFF
            for j in range(0, 64, 4):
                cur = (state >> j) & 0xF
                cur = (cur >> 3) | ((cur >> 2) & 2) | ((cur << 3) & 8) | ((cur << 2) & 4)
                state ^= cur << j
        outputs.append(val)
    return outputs

# Known PRNG outputs used for verification
expected_code1 = 5090412420734391115
expected_code2 = 4310087371899710319

# Declare symbolic seed
seed = BitVec("seed", 128)
solver = Solver()

# Symbolic PRNG implementation for Z3
def get_symbolic_output(symbolic_seed, steps):
    state = 0
    seed_copy = symbolic_seed
    for _ in range(16):
        cur = seed_copy & 3
        seed_copy >>= 2
        state = (state << 4) | ((state & 3) ^ cur)
        state |= cur << 2

    outputs = []
    for _ in range(steps):
        val = 0
        for _ in range(26):
            val <<= 1
            val |= state & 1
            state = (state << 1) ^ LShR(state, 61)
            state &= 0xFFFFFFFFFFFFFFFF
            state ^= 0xFFFFFFFFFFFFFFFF
            for j in range(0, 64, 4):
                cur = (state >> j) & 0xF
                cur = (LShR(cur, 3)) | ((LShR(cur, 2)) & 2) | ((cur << 3) & 8) | ((cur << 2) & 4)
                state ^= cur << j
        outputs.append(val)
    return outputs

# Add symbolic constraints
expected_outputs = get_symbolic_output(seed, 2)
solver.add(seed >= 0, seed <= 0xFFFFFFFF)
solver.add(expected_outputs[0] == expected_code1)
solver.add(expected_outputs[1] == expected_code2)

# Solve for seed and compute next output
if solver.check() == sat:
    model = solver.model()
    solved_seed = model[seed].as_long()
    print("[+] Found seed:", solved_seed)

    generated_codes = generate_output(solved_seed, 3)
    print("[+] Third unlock code:", generated_codes[2])
else:
    print("[-] No solution found.")

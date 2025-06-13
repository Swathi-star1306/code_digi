```python
# Spread spectrum
import numpy as np
import matplotlib.pyplot as plt

# System Parameters
data_length = 4                # Number of bits
chips_per_bit = 8              # PN sequence length
bit_rate = 1e3                 # 1 kbps
chip_rate = bit_rate * chips_per_bit
carrier_freq = 20e3            # 20 kHz carrier
sample_rate = 160e3            # 160 kHz sampling rate
samples_per_chip = int(sample_rate / chip_rate)

# Generate random binary data
def generate_data(length):
    return np.random.randint(0, 2, length)

import numpy as np
import matplotlib.pyplot as plt

# System Parameters
data_length = 4                # Number of bits
chips_per_bit = 8              # PN sequence length
bit_rate = 1e3                 # 1 kbps
chip_rate = bit_rate * chips_per_bit
carrier_freq = 20e3            # 20 kHz carrier
sample_rate = 160e3            # 160 kHz sampling rate
samples_per_chip = int(sample_rate / chip_rate)

# Generate random binary data
def generate_data(length):
    return np.random.randint(0, 2, length)

# Generate PN sequence: ±1 chips
def generate_pn_sequence(length):
    return np.random.choice([-1, 1], length)

# BPSK mapping: 0 → -1, 1 → +1
def bpsk_modulate(bit):
    return 2 * bit - 1

# DSSS spreading
def dsss_spread(data, pn_sequence):
    spread = []
    for bit in data:
        bpsk_bit = bpsk_modulate(bit)
        spread.extend(bpsk_bit * pn_sequence)
    return np.array(spread)

# BPSK carrier modulation of spread signal
def carrier_modulate(spread_signal, carrier_freq, sample_rate, samples_per_chip):
    total_samples = len(spread_signal) * samples_per_chip
    t = np.arange(total_samples) / sample_rate
    carrier_wave = np.cos(2 * np.pi * carrier_freq * t)

    # Repeat each chip to match carrier sampling
    chip_samples = np.repeat(spread_signal, samples_per_chip)
    return chip_samples * carrier_wave, t

# Main function
if __name__ == "__main__":
    # Generate input
    data = generate_data(data_length)
    pn_seq = generate_pn_sequence(chips_per_bit)

    print("Original Data Bits:     ", data)
    print("PN Sequence:            ", pn_seq)
    
    # DSSS spreading
    spread_signal = dsss_spread(data, pn_seq)

    # BPSK Carrier modulation
    bpsk_waveform, t = carrier_modulate(spread_signal, carrier_freq, sample_rate, samples_per_chip)

    # Plot DSSS spread signal (chip values)
    plt.figure(figsize=(12, 3))
    plt.plot(spread_signal, drawstyle='steps-mid')
    plt.title("DSSS Spread Signal (Baseband)")
    plt.xlabel("Chip Index")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()

    # Plot BPSK modulated carrier waveform
    plt.figure(figsize=(12, 3))
    plt.plot(t, bpsk_waveform)
    plt.title("BPSK Modulated Waveform (Carrier)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
```

```python
import numpy as np

pb = []           # Parity matrix rows
Ik = []           # Identity matrix
p = []
m = []
h_dis = []
r_code = []
err = []

# Input for matrix dimensions
col = int(input("Enter the number of parity bits: "))
row = int(input("Enter the number of message bits: "))

# Input the parity matrix (P)
print("\nEnter the parity matrix rows:")
for i in range(row):
    p = list(map(int, input(f"Row {i+1}: ").split()))
    if len(p) != col:
        raise ValueError(f"Each row must have {col} elements.")
    pb.append(p)

# Convert to numpy array
p_mat = np.array(pb, dtype=int)
Ik = np.eye(row, dtype=int)

# Generator Matrix G = [I | P]
g_mat = np.hstack((Ik, p_mat))

# Codeword length n and message bits k
k = g_mat.shape[0]
n = g_mat.shape[1]

# Generate all possible message vectors (2^k)
all_messages = np.array([[1 if (i >> (k - j - 1)) & 1 else 0 for j in range(k)] for i in range(2 ** k)])

# Generate codewords: c = m * G mod 2
codewords = np.mod(np.dot(all_messages, g_mat), 2)

# Hamming weights
hamming_weights = [np.sum(row) for row in codewords]
d_min = np.min(hamming_weights[1:])

# Parity-check matrix H = [P^T | I]
p_t = p_mat.T
h_check = np.hstack((p_t, np.eye(col, dtype=int)))
ht = h_check.T  # Transpose of H

# Output Generator Matrix
print("\n**********")
print("Generator Matrix [G = I | P]:")
for row in g_mat:
    print("".join(map(str, row)))

# Output Codewords with Proper Formatting
print("\n**********")
print("{:<15} {:<15} {:<15}".format("Message Bits", "Codeword", "Hamming Weight"))
for i in range(len(all_messages)):
    msg_str = "".join(map(str, all_messages[i]))
    code_str = "".join(map(str, codewords[i]))
    weight = hamming_weights[i]
    print("{:<15} {:<15} {:<15}".format(msg_str, code_str, weight))

# Minimum Hamming distance
print("\n**********")
print(f"Minimum Hamming Distance: {d_min}")

# Output Parity Check Matrix
print("\n**********")
print("Parity Check Matrix [H = P^T | I]:")
for row in h_check:
    print("".join(map(str, row)))

# Output Transpose of Parity Check Matrix
print("\n**********")
print("Transpose of Parity Check Matrix (H^T):")
for row in ht:
    print("".join(map(str, row)))

# Receive codeword
rc = list(map(int, input("\nEnter the received codeword: ").split()))
if len(rc) != n:
    raise ValueError("Received codeword length must match codeword length n.")
r_c = np.array([rc])

# Syndrome calculation: s = r * H^T mod 2
syndrome = np.mod(np.dot(r_c, ht), 2).flatten()

# Find error position
err = np.zeros(n, dtype=int)
for i in range(n):
    if np.array_equal(syndrome, ht[i]):
        err[i] = 1
        break

print("\n**********")
print("Syndrome:", "".join(map(str, syndrome)))
print("Error vector:", "".join(map(str, err)))

# Correct the error
corrected = (r_c.flatten() + err) % 2
print("Corrected Codeword:", "".join(map(str, corrected)))

# Optional: Syndrome Table (first few entries)
print("\n**********")
print("Syndrome Matrix:")
for i in range(n):
    s = ht[i]
    ev = np.eye(n, dtype=int)[i]
    print(f"{' '.join(map(str, s))}  {' '.join(map(str, ev))}")

print("**********")
```



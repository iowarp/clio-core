#!/usr/bin/env python3
"""
NeuroPress-style dense neural network trainer for compression-metric prediction.

Reads a unified benchmark CSV and trains a 5-layer dense network
(8 inputs -> 64 -> 64 -> 64 -> 64 -> 8 outputs) using PyTorch or numpy.

Emits model.nnwt in the exact little-endian binary format that the C++
NeuroPressNNPredictor::Load() expects.

Input mapping from 11-feature vector to 8-input:
  1. algo_id = library_config_id // 10 % 8
  2. quant = 0 (fixed)
  3. shuffle = 0 (fixed)
  4. error_bound = 0 (fixed)
  5. data_size = chunk_size_bytes
  6. entropy = shannon_entropy
  7. mad
  8. second_derivative_mean

Output (8): [compression_time_ms, decompression_time_ms, compression_ratio,
  psnr_db, 0, 0, 0, 0]

Note: decompression_time_ms is synthesized as mean of compression_time_ms
if not in CSV.
"""

import argparse
import struct
import sys
import os


def main():
    parser = argparse.ArgumentParser(
        description="Train NeuroPress-style dense NN compression predictor"
    )
    parser.add_argument("--csv", required=True, help="Input CSV path")
    parser.add_argument("--out-dir", required=True, help="Output directory for models")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        print("Error: pandas and numpy required", file=sys.stderr)
        return 1

    # Read CSV
    try:
        df = pd.read_csv(args.csv)
    except FileNotFoundError:
        print(f"Error: CSV file not found: {args.csv}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error reading CSV: {e}", file=sys.stderr)
        return 1

    # Check required columns
    required_cols = [
        "chunk_size_bytes",
        "shannon_entropy",
        "mad",
        "second_derivative_mean",
        "library_config_id",
        "compression_ratio",
        "psnr_db",
        "compression_time_ms",
        "success",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"Error: Missing CSV columns: {missing}", file=sys.stderr)
        return 1

    if df.empty:
        print("Error: CSV is empty", file=sys.stderr)
        return 1

    # Filter success == 1
    df = df[df["success"] == 1].copy()
    if df.empty:
        print("Error: No rows with success=1", file=sys.stderr)
        return 1

    print(f"Loaded {len(df)} samples")

    # Build 8-input NeuroPress feature vector
    X_8 = np.zeros((len(df), 8), dtype=np.float32)
    X_8[:, 0] = (df["library_config_id"].values.astype(np.int32) // 10) % 8  # algo_id
    X_8[:, 1] = 0.0  # quant
    X_8[:, 2] = 0.0  # shuffle
    X_8[:, 3] = 0.0  # error_bound
    X_8[:, 4] = df["chunk_size_bytes"].values  # data_size
    X_8[:, 5] = df["shannon_entropy"].values  # entropy
    X_8[:, 6] = df["mad"].values  # mad
    X_8[:, 7] = df["second_derivative_mean"].values  # second_derivative

    # Build 8-output label vector
    Y_8 = np.zeros((len(df), 8), dtype=np.float32)
    Y_8[:, 0] = df["compression_time_ms"].values  # compression_time_ms
    # decompression_time_ms: use mean of compression_time if not available
    if "decompression_time_ms" in df.columns:
        Y_8[:, 1] = df["decompression_time_ms"].values
    else:
        Y_8[:, 1] = df["compression_time_ms"].mean()
    Y_8[:, 2] = df["compression_ratio"].values  # compression_ratio
    Y_8[:, 3] = df["psnr_db"].values  # psnr_db
    Y_8[:, 4:8] = 0.0  # padding zeros

    # Standardize inputs and outputs
    x_means = X_8.mean(axis=0).astype(np.float32)
    x_stds = X_8.std(axis=0).astype(np.float32)
    x_stds[x_stds < 1e-8] = 1.0  # Avoid division by zero

    y_means = Y_8.mean(axis=0).astype(np.float32)
    y_stds = Y_8.std(axis=0).astype(np.float32)
    y_stds[y_stds < 1e-8] = 1.0  # Avoid division by zero

    X_norm = (X_8 - x_means) / x_stds
    Y_norm = (Y_8 - y_means) / y_stds

    # Try PyTorch training
    torch_available = False
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim

        torch_available = True
        print("Using PyTorch for training")

        device = torch.device("cpu")

        # Define model
        class DenseNet(nn.Module):
            def __init__(self):
                super(DenseNet, self).__init__()
                self.fc1 = nn.Linear(8, 64)
                self.fc2 = nn.Linear(64, 64)
                self.fc3 = nn.Linear(64, 64)
                self.fc4 = nn.Linear(64, 64)
                self.fc5 = nn.Linear(64, 8)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                x = torch.relu(self.fc3(x))
                x = torch.relu(self.fc4(x))
                x = self.fc5(x)
                return x

        model = DenseNet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.MSELoss()

        X_tensor = torch.from_numpy(X_norm).to(device)
        Y_tensor = torch.from_numpy(Y_norm).to(device)

        print(f"Training for {args.epochs} epochs...")
        for epoch in range(args.epochs):
            optimizer.zero_grad()
            Y_pred = model(X_tensor)
            loss = criterion(Y_pred, Y_tensor)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch + 1}: loss={loss.item():.6f}")

        # Extract weights and biases
        weights_list = []
        biases_list = []
        for layer in [model.fc1, model.fc2, model.fc3, model.fc4, model.fc5]:
            w = layer.weight.detach().cpu().numpy().astype(np.float32)  # [out, in]
            b = layer.bias.detach().cpu().numpy().astype(np.float32)
            weights_list.append(w)
            biases_list.append(b)

    except ImportError:
        print("PyTorch not available, using numpy fallback")
        torch_available = False

    if not torch_available:
        # Numpy-based training: simple SGD
        print("Using numpy SGD for training")

        # Initialize weights with He initialization
        rng = np.random.RandomState(42)

        def he_init(in_size, out_size):
            return rng.normal(0, np.sqrt(2.0 / in_size), (out_size, in_size)).astype(
                np.float32
            )

        weights_list = [
            he_init(8, 64),
            he_init(64, 64),
            he_init(64, 64),
            he_init(64, 64),
            he_init(64, 8),
        ]
        biases_list = [
            np.zeros(64, dtype=np.float32),
            np.zeros(64, dtype=np.float32),
            np.zeros(64, dtype=np.float32),
            np.zeros(64, dtype=np.float32),
            np.zeros(8, dtype=np.float32),
        ]

        print(f"Training for {args.epochs} epochs...")

        for epoch in range(args.epochs):
            loss_total = 0.0

            for i in range(len(X_norm)):
                x = X_norm[i : i + 1]
                y = Y_norm[i : i + 1]

                # Forward pass
                activations = [x]
                for j, (w, b) in enumerate(zip(weights_list, biases_list)):
                    z = np.dot(activations[-1], w.T) + b
                    if j < len(weights_list) - 1:
                        a = np.maximum(0, z)  # ReLU
                    else:
                        a = z  # Linear output
                    activations.append(a)

                y_pred = activations[-1]
                loss = np.mean((y_pred - y) ** 2)
                loss_total += loss

                # Backward pass (simple)
                delta = y_pred - y
                for j in range(len(weights_list) - 1, -1, -1):
                    dW = np.dot(delta.T, activations[j])
                    db = np.sum(delta, axis=0)
                    weights_list[j] -= args.lr * dW / len(X_norm)
                    biases_list[j] -= args.lr * db / len(X_norm)

                    if j > 0:
                        delta = np.dot(delta, weights_list[j])
                        delta[activations[j] <= 0] = 0  # ReLU derivative

            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch + 1}: loss={loss_total / len(X_norm):.6f}")

    # Extract feature bounds
    x_mins = X_8.min(axis=0).astype(np.float32)
    x_maxs = X_8.max(axis=0).astype(np.float32)

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Write model.nnwt in binary format
    model_path = os.path.join(args.out_dir, "model.nnwt")

    with open(model_path, "wb") as f:
        # Header (24 bytes)
        magic = 0x4E4E5754  # 'NNWT'
        version = 2
        num_layers = 5
        input_dim = 8
        hidden_dim = 64
        output_dim = 8

        f.write(struct.pack("<I", magic))
        f.write(struct.pack("<I", version))
        f.write(struct.pack("<I", num_layers))
        f.write(struct.pack("<I", input_dim))
        f.write(struct.pack("<I", hidden_dim))
        f.write(struct.pack("<I", output_dim))

        # Normalization (32 floats)
        for val in x_means:
            f.write(struct.pack("<f", val))
        for val in x_stds:
            f.write(struct.pack("<f", val))
        for val in y_means:
            f.write(struct.pack("<f", val))
        for val in y_stds:
            f.write(struct.pack("<f", val))

        # Weights and biases (layer by layer, row-major weights)
        for w, b in zip(weights_list, biases_list):
            # Write weights in row-major order (flatten)
            for val in w.flatten():
                f.write(struct.pack("<f", val))
            # Write biases
            for val in b:
                f.write(struct.pack("<f", val))

        # Feature bounds (16 floats)
        for val in x_mins:
            f.write(struct.pack("<f", val))
        for val in x_maxs:
            f.write(struct.pack("<f", val))

    print(f"Saved model to {model_path}")

    # Verify the binary file
    expected_header_size = 24
    expected_norm_size = 32 * 4
    expected_weights_biases_size = (
        8 * 64 * 4 + 64 * 4 +
        64 * 64 * 4 + 64 * 4 +
        64 * 64 * 4 + 64 * 4 +
        64 * 64 * 4 + 64 * 4 +
        64 * 8 * 4 + 8 * 4
    )
    expected_bounds_size = 16 * 4
    expected_total_size = (
        expected_header_size
        + expected_norm_size
        + expected_weights_biases_size
        + expected_bounds_size
    )

    actual_size = os.path.getsize(model_path)
    if actual_size == expected_total_size:
        print(f"OK: File size {actual_size} matches expected {expected_total_size}")
    else:
        print(
            f"WARNING: File size {actual_size} != expected {expected_total_size}",
            file=sys.stderr,
        )
        return 1

    # Verify header by re-reading
    with open(model_path, "rb") as f:
        magic_read = struct.unpack("<I", f.read(4))[0]
        version_read = struct.unpack("<I", f.read(4))[0]
        if magic_read == 0x4E4E5754 and version_read == 2:
            print("OK: Header magic and version verified")
        else:
            print(f"ERROR: Header mismatch", file=sys.stderr)
            return 1

    print("NeuroPress NN training complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())

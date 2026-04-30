import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import argparse
import datetime
from torch.utils.data import DataLoader, random_split
from .model import Surrogate, SurrogateDataset

def cosine_distance(x, y):
    return 1 - F.cosine_similarity(x, y, dim=-1)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to samples.parquet")
    parser.add_argument("--embeddings", type=str, required=True, help="Path to embeddings.npy")
    parser.add_argument("--out", type=str, required=True, help="Output directory for runs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume")
    args = parser.parse_args()

    # Load data
    df = pd.read_parquet(args.dataset)
    # Correctly identify parameter columns
    # Select only columns starting with 'p_' as per s01/s02 convention
    param_cols = [c for c in df.columns if c.startswith("p_")]
    
    # Ensure they are numeric (float32)
    params_df = df[param_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    params = torch.tensor(params_df.values.astype(np.float32), dtype=torch.float32)
    
    notes = torch.tensor(df["note"].values.astype(np.float32) / 127.0, dtype=torch.float32)
    latents = torch.tensor(np.load(args.embeddings), dtype=torch.float32)

    dataset = SurrogateDataset(params, notes, latents)
    
    # Split: 80/10/10 reproducible by hash or simple split (using seed for now)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size], 
                                            generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, pin_memory=True)

    input_dim = len(param_cols) + 1
    model = Surrogate(input_dim=input_dim).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Setup Output Dir
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.out, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}")

    best_val_loss = float("inf")

    # 4090 optimization: AMP
    scaler = torch.amp.GradScaler("cuda", enabled=(args.device == "cuda"))

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0
        for b_params, b_notes, b_latents in train_loader:
            b_params, b_notes, b_latents = b_params.to(args.device), b_notes.to(args.device), b_latents.to(args.device)
            
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=(args.device == "cuda")):
                pred = model(b_params, b_notes)
                mse = F.mse_loss(pred, b_latents)
                cos = cosine_distance(pred, b_latents).mean()
                loss = mse + 0.3 * cos
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for b_params, b_notes, b_latents in val_loader:
                b_params, b_notes, b_latents = b_params.to(args.device), b_notes.to(args.device), b_latents.to(args.device)
                pred = model(b_params, b_notes)
                mse = F.mse_loss(pred, b_latents)
                cos = cosine_distance(pred, b_latents).mean()
                val_loss += (mse + 0.3 * cos).item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        print(f"Epoch {epoch}: Train={avg_train:.6f}, Val={avg_val:.6f}")

        # Save Checkpoint
        check_path = os.path.join(run_dir, "checkpoint_latest.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": avg_val
        }, check_path)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), os.path.join(run_dir, "state_dict.pt"))
            # Export ONNX as well
            dummy_p = torch.randn(1, len(param_cols)).to(args.device)
            dummy_n = torch.tensor([0.5]).to(args.device)
            torch.onnx.export(model, (dummy_p, dummy_n), 
                             os.path.join(run_dir, "surrogate.onnx"),
                             input_names=["params", "note"], output_names=["encodec_latent"],
                             dynamic_axes={"params": {0: "batch"}, "note": {0: "batch"}, "encodec_latent": {0: "batch"}})

    # Save manifest
    manifest = {
        "timestamp": timestamp,
        "input_dim": input_dim,
        "param_cols": param_cols,
        "best_val_loss": best_val_loss,
        "epochs": args.epochs
    }
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=4)

if __name__ == "__main__":
    train()

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
from .model import Surrogate, SurrogateDataset, SurrogateMRSTFTHead
import defaults as _defs

def cosine_distance(x, y):
    return 1 - F.cosine_similarity(x, y, dim=-1)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",         type=str, default=str(_defs.S03_PARQUET),
                        help="Path to samples.parquet")
    parser.add_argument("--embeddings",      type=str, default=str(_defs.S04_EMBEDDINGS),
                        help="Path to embeddings .npy (encodec_embeddings.npy or clap_embeddings.npy)")
    parser.add_argument("--out",             type=str, default=str(_defs.S05_RUNS_DIR),
                        help="Output directory for runs")
    parser.add_argument("--epochs",          type=int, default=100)
    parser.add_argument("--batch-size",      type=int, default=256)
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--device",          type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume",          type=str, help="Path to checkpoint to resume")
    parser.add_argument("--hidden-dim",      type=int, default=1024)
    parser.add_argument("--film",            dest="film", action="store_true",  default=True)
    parser.add_argument("--no-film",         dest="film", action="store_false")
    parser.add_argument("--embed-model",     choices=["encodec", "clap"], default="encodec",
                        help="Embedding model used for target latents")
    parser.add_argument("--mrstft-features", type=str, default=None,
                        help="Path to mrstft_features.npy for auxiliary spectral loss")
    args = parser.parse_args()

    # Load data
    df = pd.read_parquet(args.dataset)
    param_cols = [c for c in df.columns if c.startswith("p_")]

    params_df = df[param_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    params  = torch.tensor(params_df.values.astype(np.float32), dtype=torch.float32)
    notes   = torch.tensor(df["note"].values.astype(np.float32) / 127.0, dtype=torch.float32)
    latents = torch.tensor(np.load(args.embeddings), dtype=torch.float32)

    mrstft = None
    if args.mrstft_features:
        mrstft = torch.tensor(np.load(args.mrstft_features), dtype=torch.float32)

    dataset = SurrogateDataset(params, notes, latents, mrstft)

    train_size = int(0.8 * len(dataset))
    val_size   = int(0.1 * len(dataset))
    test_size  = len(dataset) - train_size - val_size
    train_ds, val_ds, test_ds = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, pin_memory=True)

    input_dim  = len(param_cols) + 1
    output_dim = 128 if args.embed_model == "encodec" else 512
    model      = Surrogate(input_dim=input_dim, output_dim=output_dim,
                           hidden_dim=args.hidden_dim, use_film=args.film).to(args.device)

    mrstft_head = None
    if args.mrstft_features:
        mrstft_head = SurrogateMRSTFTHead(args.hidden_dim).to(args.device)

    params_to_opt = list(model.parameters())
    if mrstft_head is not None:
        params_to_opt += list(mrstft_head.parameters())
    optimizer = torch.optim.AdamW(params_to_opt, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir   = os.path.join(args.out, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if mrstft_head is not None and "mrstft_head_state_dict" in checkpoint:
            mrstft_head.load_state_dict(checkpoint["mrstft_head_state_dict"])
        print(f"Resuming from epoch {start_epoch}")

    best_val_loss = float("inf")
    scaler = torch.amp.GradScaler("cuda", enabled=(args.device == "cuda"))

    for epoch in range(start_epoch, args.epochs):
        model.train()
        if mrstft_head is not None:
            mrstft_head.train()
        train_loss = 0

        for batch in train_loader:
            if mrstft is not None:
                b_params, b_notes, b_latents, b_mrstft = batch
                b_mrstft = b_mrstft.to(args.device)
            else:
                b_params, b_notes, b_latents = batch
                b_mrstft = None

            b_params, b_notes, b_latents = (
                b_params.to(args.device),
                b_notes.to(args.device),
                b_latents.to(args.device),
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=(args.device == "cuda")):
                pred = model(b_params, b_notes)
                mse  = F.mse_loss(pred, b_latents)
                cos  = cosine_distance(pred, b_latents).mean()
                loss = mse + 0.3 * cos

                if mrstft_head is not None and b_mrstft is not None:
                    hidden      = model.forward_features(b_params, b_notes)
                    pred_mrstft = mrstft_head(hidden)
                    loss        = loss + 0.1 * F.mse_loss(pred_mrstft, b_mrstft)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        # Validation (primary loss only — mrstft head is training-only)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                b_params, b_notes, b_latents = batch[0], batch[1], batch[2]
                b_params, b_notes, b_latents = (
                    b_params.to(args.device),
                    b_notes.to(args.device),
                    b_latents.to(args.device),
                )
                pred = model(b_params, b_notes)
                mse  = F.mse_loss(pred, b_latents)
                cos  = cosine_distance(pred, b_latents).mean()
                val_loss += (mse + 0.3 * cos).item()

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)
        print(f"Epoch {epoch}: Train={avg_train:.6f}, Val={avg_val:.6f}")

        # Checkpoint (includes mrstft head state for resume)
        check_path = os.path.join(run_dir, "checkpoint_latest.pt")
        ckpt = {
            "epoch":               epoch,
            "model_state_dict":    model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss":            avg_val,
        }
        if mrstft_head is not None:
            ckpt["mrstft_head_state_dict"] = mrstft_head.state_dict()
        torch.save(ckpt, check_path)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            # Save only the surrogate — mrstft head is not used at inference time
            torch.save(model.state_dict(), os.path.join(run_dir, "state_dict.pt"))
            print(f"  → new best, saved state_dict.pt")

        scheduler.step()

    # Save manifest
    manifest = {
        "timestamp":    timestamp,
        "input_dim":    input_dim,
        "param_cols":   param_cols,
        "hidden_dim":   args.hidden_dim,
        "use_film":     args.film,
        "output_dim":   output_dim,
        "embed_model":  args.embed_model,
        "best_val_loss": best_val_loss,
        "epochs":       args.epochs,
    }
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=4)

    # Export ONNX once, from the best weights, after training completes.
    model.load_state_dict(torch.load(os.path.join(run_dir, "state_dict.pt"),
                                     map_location=args.device, weights_only=True))
    model.eval()
    dummy_p = torch.randn(1, len(param_cols)).to(args.device)
    dummy_n = torch.tensor([0.5]).to(args.device)
    batch   = torch.export.Dim("batch")
    torch.onnx.export(
        model, (dummy_p, dummy_n),
        os.path.join(run_dir, "surrogate.onnx"),
        input_names=["params", "note"],
        output_names=["embedding"],
        dynamic_shapes={"params": {0: batch}, "note": {0: batch}},
    )
    print("ONNX exported.")

if __name__ == "__main__":
    train()

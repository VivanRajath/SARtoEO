import os
import csv
import argparse
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Plot training loss curves from training_log.csv")
    parser.add_argument("--csv_path", type=str, default="outputs/training_log.csv",
                        help="Path to the training log CSV file")
    parser.add_argument("--output_path", type=str, default="outputs/loss_curve.png",
                        help="Path to save the generated loss curve plot")
    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        print(f"Error: Log file not found at '{args.csv_path}'")
        return

    print(f"Reading log from: {args.csv_path}")
    history = []
    with open(args.csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            history.append({
                "epoch": int(row["epoch"]),
                "train_g": float(row["train_g_total"]),
                "train_g_l1": float(row["train_g_l1_only"]),
                "train_d": float(row["train_d_loss"]),
                "val_g": float(row["val_g_total"]),
                "val_g_l1": float(row["val_g_l1_only"]),
                "val_d": float(row["val_d_loss"])
            })

    if not history:
        print("Error: No history data found in CSV file.")
        return

    epochs    = [h["epoch"]      for h in history]
    tr_g      = [h["train_g"]    for h in history]
    tr_g_l1   = [h["train_g_l1"] for h in history]
    tr_d      = [h["train_d"]    for h in history]
    v_g       = [h["val_g"]      for h in history]
    v_g_l1    = [h["val_g_l1"]   for h in history]
    v_d       = [h["val_d"]      for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Generator ablation
    axes[0].plot(epochs, tr_g,    color="royalblue",  label="Train G (GAN+L1)")
    axes[0].plot(epochs, v_g,     color="royalblue",  linestyle="--", label="Val G (GAN+L1)")
    axes[0].plot(epochs, tr_g_l1, color="darkorange", label="Train G (L1 only)")
    axes[0].plot(epochs, v_g_l1,  color="darkorange", linestyle="--", label="Val G (L1 only)")
    axes[0].set_title("Generator Loss (Ablation: GAN+L1 vs L1-only)")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    # Panel 2: Discriminator
    axes[1].plot(epochs, tr_d, color="tomato", label="Train D")
    axes[1].plot(epochs, v_d,  color="tomato", linestyle="--", label="Val D")
    axes[1].set_title("Discriminator Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    # Panel 3: Decompose GAN component from total generator loss
    tr_g_gan = [g - l for g, l in zip(tr_g, tr_g_l1)]
    axes[2].plot(epochs, tr_g_gan, color="purple", label="GAN component")
    axes[2].plot(epochs, tr_g_l1,  color="green",  label="L1 component (λ×L1)")
    axes[2].set_title("Generator Loss Decomposition")
    axes[2].set_xlabel("Epoch")
    axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3)

    plt.suptitle("Pix2Pix SAR→EO  |  Training Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    plt.savefig(args.output_path, dpi=150)
    plt.close()
    print(f"Success: Loss curve plotted and saved to: {args.output_path}")

if __name__ == "__main__":
    main()

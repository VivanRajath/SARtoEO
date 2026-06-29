

import os
import sys

REPO_ID    = "VivanRajath/SAR2EO"
FILENAME   = "checkpoint_latest.pth"
DEST_DIR   = "checkpoints"
DEST_PATH  = os.path.join(DEST_DIR, FILENAME)
HF_URL     = f"https://huggingface.co/{REPO_ID}/resolve/main/{FILENAME}"


def download_via_huggingface_hub():
    """Download using huggingface_hub (preferred — resumable, progress bar)."""
    from huggingface_hub import hf_hub_download
    import shutil

    print(f"[huggingface_hub] Downloading {FILENAME} from {REPO_ID} ...")
    cached = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    os.makedirs(DEST_DIR, exist_ok=True)
    shutil.copy2(cached, DEST_PATH)
    print(f"[huggingface_hub] Saved to: {DEST_PATH}")


def download_via_urllib():
    """Download using stdlib urllib — no extra dependencies."""
    import urllib.request

    os.makedirs(DEST_DIR, exist_ok=True)

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded * 100 / total_size, 100)
            bar = int(pct // 2)
            sys.stdout.write(
                f"\r  [{('#' * bar):<50}] {pct:5.1f}%"
                f"  {downloaded/1024/1024:.1f} / {total_size/1024/1024:.1f} MB"
            )
        else:
            sys.stdout.write(f"\r  Downloaded {downloaded/1024/1024:.1f} MB")
        sys.stdout.flush()

    print(f"[urllib] Downloading from:\n  {HF_URL}")
    print(f"[urllib] Saving to: {DEST_PATH}\n")
    urllib.request.urlretrieve(HF_URL, DEST_PATH, _progress)
    print("\n[urllib] Download complete!")


def main():
    print("=" * 60)
    print("  SAR2EO — Model Weight Downloader")
    print("=" * 60)

    if os.path.exists(DEST_PATH):
        size_mb = os.path.getsize(DEST_PATH) / 1024 / 1024
        print(f"[Info] Weights already exist: {DEST_PATH}  ({size_mb:.0f} MB)")
        ans = input("  Re-download? [y/N]: ").strip().lower()
        if ans != "y":
            print("[Skip] Using existing weights.")
            return

    # Try huggingface_hub first; fall back to urllib
    try:
        import huggingface_hub  # noqa: F401
        download_via_huggingface_hub()
    except ImportError:
        print("[Info] huggingface_hub not installed — using urllib fallback.")
        print("[Tip]  For faster downloads: pip install huggingface_hub\n")
        download_via_urllib()
    except Exception as e:
        print(f"[Warning] huggingface_hub failed ({e}), retrying with urllib ...")
        download_via_urllib()

    # Sanity check
    if os.path.exists(DEST_PATH):
        size_mb = os.path.getsize(DEST_PATH) / 1024 / 1024
        print(f"\n[OK] checkpoint_latest.pth saved  ({size_mb:.0f} MB)")
        print(f"[OK] Path: {os.path.abspath(DEST_PATH)}")
        print("\nNext step — run inference:")
        print("  python infer.py --input_dir sample/ --output_dir outputs/generated_eo/ --weights checkpoints/checkpoint_latest.pth")
    else:
        print("\n[Error] Download may have failed. Try manually:")
        print(f"  {HF_URL}")


if __name__ == "__main__":
    main()

import os
import urllib.request
import sys

def download_weights():
    url = "https://huggingface.co/VivanRajath/SAR2EO/resolve/main/checkpoint_latest.pth"
    dest_dir = "checkpoints"
    dest_path = os.path.join(dest_dir, "checkpoint_latest.pth")
    
    os.makedirs(dest_dir, exist_ok=True)
    
    print(f"Downloading model weights from: {url}")
    print(f"Saving to: {dest_path}")
    
    try:
        # Simple progress report callback
        def reporthook(block_num, block_size, total_size):
            read_so_far = block_num * block_size
            if total_size > 0:
                percent = read_so_far * 1e2 / total_size
                s = f"\rProgress: {percent:5.1f}% [{read_so_far / 1024 / 1024:.1f} MB / {total_size / 1024 / 1024:.1f} MB]"
                sys.stdout.write(s)
                sys.stdout.flush()
            else:
                sys.stdout.write(f"\rDownloaded {read_so_far / 1024 / 1024:.1f} MB")
                sys.stdout.flush()
        
        urllib.request.urlretrieve(url, dest_path, reporthook)
        print("\nDownload completed successfully!")
    except Exception as e:
        print(f"\nError downloading weights: {e}")
        print("Please download the weights manually from: https://huggingface.co/VivanRajath/SAR2EO")

if __name__ == "__main__":
    download_weights()

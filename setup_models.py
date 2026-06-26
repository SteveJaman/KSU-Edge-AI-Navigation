from huggingface_hub import snapshot_download
from pathlib import Path
import sys

# Root of the repo
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# All models to download
MODEL_SOURCES = {
    "fastvlm-0.5b":         "apple/FastVLM-0.5B",
    "LFM2-450m":            "LiquidAI/LFM2-VL-450M",
    "minicpm-v-4.6-1.3b":   "openbmb/MiniCPM-V-4_6",
    "moondream2-0.5b":      "whistleroosh/moondream-0.5B",
    "qwen3.5-0.8b":         "Qwen/Qwen2.5-0.5B-Instruct",
    "smolvlm-500m":         "HuggingFaceTB/SmolVLM-500M-Instruct",
}

print("\n=== Downloading all VLM models into ./models/ ===\n")

for key, repo_id in MODEL_SOURCES.items():
    target_dir = MODELS_DIR / key
    target_dir.mkdir(exist_ok=True)

    print(f"→ Downloading {key}  ({repo_id})")
    try:
        # Added trust_remote_code=True for models that require custom architecture code
        snapshot_download(
            repo_id=repo_id,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            trust_remote_code=True 
        )
        print(f"   ✓ Saved to: {target_dir}\n")
    except Exception as e:
        print(f"   ❌ Failed to download {key}: {e}\n")
        continue

print("=== Download process complete. Check logs above for any errors. ===")
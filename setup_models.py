from huggingface_hub import snapshot_download
from pathlib import Path

# Root of the repo
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------
# Correct model sources for this benchmark
# ---------------------------------------------------------
MODEL_SOURCES = {
    # SmolVLM (500M)
    "smolvlm-500m": "HuggingFaceTB/SmolVLM-500M-Instruct",

    # FastVLM (0.5B)
    "fastvlm-0.5b": "apple/FastVLM-0.5B",

    # MiniCPM-V 4.6 (1.3B)
    "minicpm-v-4.6-1.3b": "openbmb/MiniCPM-V-4_6",

    # Qwen2.5 0.5B Instruct (text-only)
    "qwen3.5-0.8b": "Qwen/Qwen2.5-0.5B-Instruct",

    # LFM2 (450M)
    "lfm2-450m": "LiquidAI/LFM2-VL-450M",

    # Moondream2 (0.5B)
    "moondream2-0.5b": "vikhyatk/moondream2",
}

print("\n=== Downloading all VLM models into ./models/ ===\n")

for key, repo_id in MODEL_SOURCES.items():
    target_dir = MODELS_DIR / key
    target_dir.mkdir(exist_ok=True)

    print(f"→ Downloading {key}  ({repo_id})")
    try:
        # Modern snapshot_download no longer accepts trust_remote_code
        snapshot_download(
            repo_id=repo_id,
            local_dir=target_dir,
            local_dir_use_symlinks=False
        )
        print(f"   ✓ Saved to: {target_dir}\n")
    except Exception as e:
        print(f"   ❌ Failed to download {key}: {e}\n")
        continue

print("=== Download process complete. Check logs above for any errors. ===")

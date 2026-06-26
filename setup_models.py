from huggingface_hub import snapshot_download
from pathlib import Path

# Root of the repo
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# All models to download
MODEL_SOURCES = {
    "smolvlm":      "HuggingFaceTB/SmolVLM-500M-Instruct",
    "fastvlm":      "apple/FastVLM-0.5B",
    "minicpm":      "openbmb/MiniCPM-V-4_6",
    "qwen3.5":      "Qwen/Qwen2.5-0.5B-Instruct",   # text-only 0.8B local version
    "moondream":    "vikhyatk/moondream2",
    "lfm2":         "llava-hf/LFM-2-450M",
}

print("\n=== Downloading all VLM models into ./models/ ===\n")

for key, repo_id in MODEL_SOURCES.items():
    target_dir = MODELS_DIR / key
    target_dir.mkdir(exist_ok=True)

    print(f"→ Downloading {key}  ({repo_id})")
    snapshot_download(
        repo_id=repo_id,
        local_dir=target_dir,
        local_dir_use_symlinks=False,
    )
    print(f"   ✓ Saved to: {target_dir}\n")

print("=== All models downloaded successfully! ===")
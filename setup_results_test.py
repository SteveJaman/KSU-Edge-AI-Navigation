from pathlib import Path

# Set the base directory to your project root
PROJECT_ROOT = Path(__file__).resolve().parent

# Define the folders to create (must match model_key names in benchmark)
RESULTS_SUBFOLDERS = [
    "Combined_CSV",
    "fastvlm-0.5b",
    "gemini_flash",
    "lfm2-450m",
    "minicpm-v-4.6-1.3b",
    "moondream2-0.5b",
    "qwen3.5-0.8b",
    "smolvlm-500m",
]

def setup_folders():
    # 1. Setup Results Structure
    results_dir = PROJECT_ROOT / "results"
    print(f"--- Setting up Results directory: {results_dir} ---")
    
    for subfolder in RESULTS_SUBFOLDERS:
        folder_path = results_dir / subfolder
        folder_path.mkdir(parents=True, exist_ok=True)
        print(f" [OK] Created: {folder_path.name}")

    # 2. Setup Empty Test Folder
    test_dir = PROJECT_ROOT / "test"
    test_dir.mkdir(exist_ok=True)
    print(f"\n--- Setting up Test directory: {test_dir} ---")
    print(" [INFO] Ready for Google Drive data drop.")
    
    print("\n=== Setup complete! ===")

if __name__ == "__main__":
    setup_folders()

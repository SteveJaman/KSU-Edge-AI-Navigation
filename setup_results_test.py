from pathlib import Path

# Set the base directory to your project root
PROJECT_ROOT = Path(__file__).resolve().parent

# Define the folders to create
RESULTS_SUBFOLDERS = [
    "Combined_CSV",
    "fastvlm",
    "gemini_flash",
    "LFM2",
    "minicpm",
    "moondream",
    "qwen3.5",
    "smolvlm"
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
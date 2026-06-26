# 📘 **Indoor Navigation VLM Benchmarking Suite**  
*A fully reproducible research framework for evaluating lightweight Vision‑Language Models (VLMs) on indoor navigation tasks.*

---

## 🚀 Overview

This repository provides a **complete, end‑to‑end benchmarking pipeline** for evaluating multiple **local Vision‑Language Models (VLMs)** on indoor navigation datasets. It is designed for research reproducibility: **anyone can clone the repo and run the full benchmark**, including:

- Automatic model setup  
- Automatic dataset setup  
- Unified benchmarking engine  
- TTFT (Time‑to‑First‑Token)  
- Encode time  
- Decode throughput (TPS)  
- End‑to‑end latency  
- Resume‑safe CSV logging  
- FP16 and INT4 quantization  
- Desktop GPU + Jetson support  

This suite is used in ongoing research at **Kennesaw State University** on assistive indoor navigation for low‑vision individuals.

---

## 📂 Repository Structure

```
project/
│
├── benchmark.py                 # Main benchmarking entry point
├── engines/                     # All model engine implementations
│   ├── smolvlm_engine.py
│   ├── fastvlm_engine.py
│   ├── minicpm_engine.py
│   ├── qwen35_engine.py
│   ├── moondream_engine.py      # Includes Moondream2 limitations
│   ├── lfm2_engine.py
│   └── base_engine.py
│
├── models/                      # Auto-populated by setup_models.py
│   ├── smolvlm/
│   ├── fastvlm/
│   ├── minicpm/
│   ├── qwen3.5/
│   ├── moondream2-0.5b/         # Requires custom loader
│   └── lfm2/
│
├── test/                        # Auto-populated by setup_dataset.py
│   ├── images/
│   ├── videos/
│   └── vlm-nav-images/
│
├── setup_models.py              # Downloads all required models
├── setup_dataset.py             # Downloads the indoor navigation dataset
├── requirements.txt
└── README.md
```

---

## 🧩 Supported Models

All models run **locally** with no internet required after setup.

| Key        | Model Name                | Size   | Type | Notes |
|------------|---------------------------|--------|------|-------|
| `smolvlm`  | SmolVLM‑500M              | 0.5B   | VLM  | Very fast |
| `fastvlm`  | FastVLM‑0.5B              | 0.5B   | VLM  | Heavy vision tower |
| `minicpm`  | MiniCPM‑V 4.6             | 1.3B   | VLM  | High accuracy |
| `qwen3.5`  | Qwen3.5‑0.8B              | 0.8B   | Text | Uses base64 images |
| `moondream`| Moondream2‑0.5B           | 0.5B   | VLM  | **VQA‑only; cannot perform spatial reasoning** |
| `lfm2`     | LFM2‑450M                 | 0.45B  | VLM  | Lightweight LLaVA‑style |
| `gemini`   | Gemini 2.5 Flash          | Cloud  | Gold‑standard reference |

### ⚠️ Important Note About Moondream2‑0.5B  
Moondream2 is a **VQA‑style model**, not a captioning or spatial reasoning model.  
It **cannot** produce scene‑level or navigation‑level descriptions.  
In this benchmark, Moondream2 will:

- Load successfully  
- Produce valid latency metrics  
- But return **empty or trivial outputs** for spatial prompts  

This is expected and documented in the results.

---

## 🛠️ Installation

### 1. Clone the repository

```
git clone https://github.com/<your-repo>/indoor-vlm-benchmark.git
cd indoor-vlm-benchmark
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

Supports:

- Windows  
- Linux  
- Jetson Orin (Nano/AGX)  
- CUDA 11+  

---

## 📥 Download All Models

Run:

```
python setup_models.py
```

This script automatically downloads:

- SmolVLM‑500M  
- FastVLM‑0.5B  
- MiniCPM‑V‑4.6  
- Qwen3.5‑0.8B  
- Moondream2‑0.5B  
- LFM2‑450M  

Models are stored in:

```
models/<model-key>/
```

### ⚠️ Moondream2 Special Handling  
Moondream2 requires a **custom loader** (`hf_moondream.py`) and cannot be loaded using `AutoModelForCausalLM`.  
The engine handles this automatically.

---

## 🛠 Data Setup

Download the dataset:

**Google Drive:**  
[https://drive.google.com/drive/folders/1_4paEVv-lHZO6c0Bk9POEvoVDqvEq_Mg?usp=sharing](https://drive.google.com/drive/folders/1_4paEVv-lHZO6c0Bk9POEvoVDqvEq_Mg?usp=sharing)

Extract into:

```
test/
    images/
    videos/
```

---

## 🔐 Environment Variables (`.env` Setup)

Some models (e.g., **Gemini Flash**) require API keys.  
To enable cloud‑based evaluation, create a `.env` file in the project root:

```
# .env
GEMINI_API_KEY=your_api_key_here
```

Then load it automatically by installing:

```
python-dotenv
```

Your engines will automatically read environment variables using:

```python
from dotenv import load_dotenv
load_dotenv()
```

### ✔ Offline Models  
All local VLMs (SmolVLM, FastVLM, MiniCPM, Qwen, LFM2, Moondream2) **do not require any API keys** and run fully offline.

### ✔ Cloud Models  
Only Gemini Flash requires:

```
GEMINI_API_KEY
```

If the key is missing, the benchmark will:

- Skip cloud models  
- Continue running all offline models normally

---

## ▶️ Running the Benchmark

```
python benchmark.py
```

You will be prompted to select:

1. Model  
2. Precision (FP16 or INT4)  
3. Dataset  
4. Resume mode  

The benchmark automatically:

- Loads the model  
- Processes each image  
- Measures TTFT, encode time, decode TPS, E2E latency  
- Saves results to CSV  
- Supports resume mode  

---

## 📊 Output Format (CSV)

Each row contains:

| Column | Description |
|--------|-------------|
| `image_path` | Input image |
| `model_key` | Model used |
| `precision` | FP16 or INT4 |
| `encode_time_s` | Vision encoder time |
| `ttft_s` | Time to first token |
| `decode_tps` | Tokens per second |
| `e2e_latency_s` | Encode + full generation |
| `token_count` | Output token count |
| `output` | Model response |

### ⚠️ Moondream2 CSV Behavior  
Moondream2 will produce:

- Valid latency numbers  
- `token_count = 0`  
- `output = "[Moondream2 returned no answer]"`  

This is expected.

---

## ⚙️ Reproducibility Guarantees

This repo is designed so **anyone** can reproduce your results:

- All models downloaded automatically  
- Dataset downloaded automatically  
- No absolute paths  
- Deterministic benchmarking  
- Resume‑safe CSV logging  
- All engines share the same evaluation pipeline  
- Works offline after setup  

---

## 🧪 Adding New Models

To add a new model:

1. Create a new engine in `engines/`
2. Add the model to `MODEL_REGISTRY`
3. Add a label to `MODEL_LABELS`
4. Add the model to `setup_models.py`

The system will automatically integrate it into the menu and CSV.

---

## 🐛 Troubleshooting

### **Moondream2 returns empty outputs**  
Expected. Moondream2 is a VQA model and cannot perform spatial reasoning.

### **OpenMP crash on Windows**  
Set this at the top of your script:

```
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
```

### **CUDA out of memory**  
Use INT4 or reduce image resolution.

### **Dataset missing**  
Run:

```
python setup_dataset.py
```

---

## 📄 Citation

```
Nguyen, A. (2026). Indoor Navigation VLM Benchmarking Suite.
Kennesaw State University.
```

---

## 🤝 Acknowledgements

This project is part of ongoing research under:

- **Prof. Brooke Zhao**  
- **Prof. Xu Tao**  

Kennesaw State University  
Assistive Indoor Navigation for Low‑Vision Individuals

---

# 📘 **Indoor Navigation VLM Benchmarking Suite**  
*A fully reproducible research framework for evaluating lightweight Vision‑Language Models (VLMs) on indoor navigation tasks.*

---

## 🚀 Overview

This repository provides a **complete, end‑to‑end benchmarking pipeline** for evaluating multiple **local Vision‑Language Models (VLMs)** on indoor navigation image datasets. It was designed for research reproducibility: **anyone can clone the repo and run the full benchmark**, including:

- Automatic model setup  
- Automatic dataset setup  
- Unified benchmarking engine  
- TTFT (Time‑to‑First‑Token)  
- Encode time  
- Decode throughput (TPS)  
- End‑to‑end latency  
- Resume‑safe CSV logging  
- Support for FP16 and INT4 quantization  
- Support for desktop GPUs and Jetson devices  

This project is used in ongoing research at **Kennesaw State University** on assistive indoor navigation for low‑vision individuals.

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
│   ├── moondream_engine.py
│   ├── lfm2_engine.py
│   └── base_engine.py
│
├── models/                      # Auto-populated by setup_models.py
│   ├── smolvlm/
│   ├── fastvlm/
│   ├── minicpm/
│   ├── qwen3.5/
│   ├── moondream/
│   └── lfm2/
│
├── test/                       # Auto-populated by setup_dataset.py
│   ├── images/              
│   ├── videos/
│   └── vlm-nav-images          

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
| `moondream`| Moondream2‑0.5B           | 0.5B   | VLM  | Extremely fast |
| `lfm2`     | LFM2‑450M                 | 0.45B  | VLM  | Lightweight LLaVA‑style |
| `gemini`   | Gemini 2.5 Flash          | Cloud  | Gold standard reference |

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

---

## 🛠 Data Setup

To run the benchmarking scripts, you will need the project's test dataset (containing the evaluation images and videos).

1. **Download the dataset:** [Download Test Dataset from Google Drive](https://drive.google.com/drive/folders/1_4paEVv-lHZO6c0Bk9POEvoVDqvEq_Mg?usp=sharing)
2. **Setup:** Download the folder and extract the contents into the `test/` directory at the root of this project:

```bash
# Your directory structure should look like this:
KSU-Edge-AI-Navigation/
├── test/
│   ├── images/
│   └── videos/
├── vlm_benchmark.py
└── README.md

    Models:
    Note: Model weights are not included in this repository due to size constraints. Please ensure your models/ directory is populated with the necessary files before running the benchmark.

## ▶️ Running the Benchmark

Run:

```
python benchmark.py
```

You will be prompted to select:

1. Model  
2. Precision (FP16 or INT4)  
3. Dataset path  
4. Output CSV path  

The benchmark automatically:

- Loads the model  
- Processes each image  
- Measures TTFT, encode time, decode TPS, E2E latency  
- Saves results to CSV  
- Supports resume mode (skips completed rows)

---

## 📊 Output Format (CSV)

Each row contains:

| Column | Description |
|--------|-------------|
| `image_path` | Input image |
| `model_key` | Model used |
| `precision` | FP16 or INT4 |
| `encode_time_s` | Tokenization + preprocessing |
| `ttft_s` | Time to first token |
| `decode_tps` | Tokens per second |
| `e2e_latency_s` | Encode + full generation |
| `token_count` | Output token count |
| `output` | Model response |

---

## ⚙️ Reproducibility Guarantees

This repo is designed so **anyone** can reproduce your results:

### ✔ All models downloaded automatically  
### ✔ Dataset downloaded automatically  
### ✔ No absolute paths  
### ✔ Deterministic benchmarking  
### ✔ Resume‑safe CSV logging  
### ✔ All engines use the same evaluation pipeline  
### ✔ Works offline after setup  

This meets the reproducibility expectations of academic research supervisors and reviewers.

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

### **Benchmark is extremely slow (e.g., 75s/image)**  
Cause: FP16 on an 8GB GPU with a heavy vision tower (FastVLM).  
Fix: Use INT4.

### **CUDA out of memory**  
Reduce:
- Image resolution  
- max_new_tokens  
- Use INT4  

### **Model not found**  
Run:

```
python setup_models.py
```

### **Dataset missing**  
Run:

```
python setup_dataset.py
```

---

## 📄 Citation

If you use this benchmark in research, please cite:

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
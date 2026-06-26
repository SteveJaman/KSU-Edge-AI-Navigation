# Navigation Benchmarking Metrics

## 1. Metrics Definitions

* **Spatial Accuracy (Scale 0.0 - 1.0):** - *Calculation:* Evaluated by LLM Judge (Gemini 2.5 Flash). Score = Average of [Direction Correctness] and [Distance Estimation Accuracy].
* *Goal:* 1.0 indicates precise navigation instructions (e.g., "Left, 2 meters").


* **Safety Relevance (Binary 0 or 1):**
* *Calculation:* Indicates if the model successfully flagged critical environmental hazards (e.g., stairs, obstacles, uneven surfaces) within the provided frame.


* **Conciseness (Word Count Index):**
* *Calculation:* $1 - |(\text{Actual Words} - 45) / 45|$.
* *Goal:* Measures proximity to the Gold Standard reference length (45 words).


* **Mean Overall Score:**
* *Calculation:* Unweighted average of Spatial Accuracy, Safety Relevance, and Conciseness.


## 2. Methodology

* **Ground Truth:** Derived from Gemini 2.5 Flash outputs.
* **Inference:** Models processed 16 frames per video.
* **Reporting:** Results are reported as $\mu \pm \sigma$ (Mean $\pm$ Standard Deviation) to demonstrate model stability.

## 3. Hardware Tiers (`device_tier`)

The `device_tier` column identifies the specific hardware environment used during data collection:

| Label | Hardware / Environment |
| --- | --- |
| `edge_jetson_orin_nano` | NVIDIA Jetson Orin Nano (Constrained Edge) |
| `consumer_rtx_2080` | NVIDIA RTX 2080 (Mid-tier Consumer) |
| `workstation_rtx_pro_6000` | NVIDIA RTX Pro 6000 (High-end Workstation) |
| `cloud` | Gemini 2.5 Flash API (Cloud-based Baseline) |

## 4. System Latency & Resource Metrics

* **`load_time_s`**: Initialization time for the model and memory allocation (seconds).
* **`mem_gb`**: Peak VRAM allocated during inference (GB).
* **`encode_time_s`**: Duration of vision encoder feature extraction.
* **`ttft_s` (Time to First Token)**: Latency until the first generated character.
* **`decode_tps` (Tokens Per Second)**: Generation speed calculation ($\text{tokens} / \text{E2E latency}$).
* **`e2e_latency_s`**: Median end-to-end inference latency (seconds).
* **`e2e_q1` / `e2e_q3**`: 25th and 75th percentiles of latency to measure jitter.
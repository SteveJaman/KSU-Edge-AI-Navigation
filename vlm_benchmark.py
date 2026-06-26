"""
benchmark_engine.py
====================
VLM Benchmarking Pipeline — "Benchmarking Small VLMs for Spatial Context-Aware
Mixed Reality on Edge Devices" (Nguyen, Tao, Zhao — KSU)

Supports models  : SmolVLM-500M, FastVLM-0.5B, MiniCPM-V-4.6-1.3B, Qwen2.5-VL-3B, Moondream2-0.5B, LFM2-450M,Gemini 2.5 Flash
Supports datasets: test/images, vlm-nav-images
Precision        : FP16 (default) or INT4 (bitsandbytes)
Devices          : T1 Jetson Orin Nano | T2 RTX-2080-class | T3 RTX PRO 6000

Output CSV columns (one row per image):
  ID, dataset, image, model_id, precision, device_tier, prompt_key,
  vlm_output, token_count, load_time_s, mem_gb, encode_time_s,
  ttft_s, decode_tps, e2e_latency_s, is_error, timestamp

Usage:
  python vlm_benchmark.py
"""

import os, sys, time, json, argparse, warnings, gc
from unittest.mock import MagicMock
from datetime import datetime
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from dotenv import load_dotenv
import types

warnings.filterwarnings("ignore", category=UserWarning)
load_dotenv()

flash_attn_stub = types.ModuleType("flash_attn")
flash_attn_stub.__spec__ = types.SimpleNamespace()
sys.modules["flash_attn"] = flash_attn_stub
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ──────────────────────────────────────────────────────────────────────────────
# Prompt Registry  (exact prompt from paper §2.1)
# ──────────────────────────────────────────────────────────────────────────────
PROMPTS = {
    "spatial_mr_v1": (
        "Describe the scene's spatial layout in 1-3 concise sentences. "
        "Include: (a) the key objects and surfaces present; "
        "(b) their positions relative to the viewer (left/right, in-front/behind, above/below, near/far); "
        "and (c) the navigable free space, any obstacles or hazards. "
        "Describe only what is clearly visible; do not guess or invent objects."
    ),
}
DEFAULT_PROMPT_KEY = "spatial_mr_v1"

# ──────────────────────────────────────────────────────────────────────────────
# Model Registry
# Add new models here; engine picks the right loader automatically.
# ──────────────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent
MODELS_DIR = REPO_ROOT / "models"

MODEL_REGISTRY = {
    "smolvlm":   str(MODELS_DIR / "SmolVLM-500M-Instruct"),
    "fastvlm":   str(MODELS_DIR / "llava-interleave-qwen-0.5b"),
    "minicpm":   str(MODELS_DIR / "minicpm-v-4.6-1.3b"),
    "qwen3.5":   str(MODELS_DIR / "qwen3.5-0.8b"),
    "moondream": str(MODELS_DIR / "moondream2-0.5b"),
    "lfm2":      str(MODELS_DIR / "LFM2-450m"),
    "gemini":    "api",
}

WARM_RUNS   = 3   # discard first N runs (GPU kernel warmup)
MEASURE_RUNS = 20  # runs used to compute median latency metrics


# ──────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────────────────────
def peak_vram_gb(device: str) -> float:
    """Returns peak VRAM allocated in GB (CUDA only)."""
    if device == "cuda" and torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / 1e9, 3)
    return 0.0

def reset_vram_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

def count_tokens(text: str) -> int:
    """Whitespace-based token count (language-agnostic proxy)."""
    return len(text.split())

def median_iqr(values: list) -> tuple[float, float, float]:
    """Returns (median, Q1, Q3)."""
    import statistics
    s = sorted(values)
    n = len(s)
    q1 = s[n // 4]
    q3 = s[(3 * n) // 4]
    return statistics.median(s), q1, q3


# ──────────────────────────────────────────────────────────────────────────────
# Base Engine
# ──────────────────────────────────────────────────────────────────────────────
class VLM_Benchmarking_Engine:
    """
    Abstract-style base. Subclasses implement _load_model() and _generate().
    run_benchmark() handles the full loop: warmup → measure → crash-safe CSV.
    """

    def __init__(
        self,
        image_dir:     str,
        results_dir:   str,
        output_filename: str,
        model_key:     str,
        precision:     str  = "fp16",   # "fp16" | "int4"
        device_tier:   str  = "T2",     # "T1" | "T2" | "T3" | "cloud"
        dataset_name:  str  = "indoor", # "indoor" | "dl3dv"
        prompt_key:    str  = DEFAULT_PROMPT_KEY,
        resume:        bool = True,
        max_images:    int  = 100,
    ):
        self.image_dir      = image_dir
        self.results_file   = os.path.join(results_dir, output_filename)
        self.model_key      = model_key
        self.model_id       = MODEL_REGISTRY[model_key]
        self.precision      = precision
        self.device_tier    = device_tier
        self.dataset_name   = dataset_name
        self.prompt_key     = prompt_key
        self.prompt         = PROMPTS[prompt_key]
        self.resume         = resume
        self.max_images     = max_images
        os.makedirs(results_dir, exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()

    def _load_model(self):
        raise NotImplementedError

    def _generate(self, image_path: str) -> str:
        """Returns raw text output for one image."""
        raise NotImplementedError

    # ── Latency measurement helpers ───────────────────────────────────────────

    def _measure_encode_ttft_decode(self, image_path: str) -> dict:
        """
        Returns per-query latency breakdown for one image.
        Subclasses may override for model-specific timing hooks.
        Default: treats full _generate() as E2E; encode/ttft not split.
        """
        t0 = time.perf_counter()
        output = self._generate(image_path)
        e2e = time.perf_counter() - t0
        tokens = count_tokens(output)
        return {
            "output":        output,
            "encode_time_s": None,    # overridden in subclasses that support it
            "ttft_s":        None,
            "decode_tps":    round(tokens / e2e, 2) if e2e > 0 else None,
            "e2e_latency_s": round(e2e, 4),
            "token_count":   tokens,
        }

    def _warm_and_measure(self, image_path: str) -> dict:
        """Runs WARM_RUNS discard passes then MEASURE_RUNS timed passes."""
        for _ in range(WARM_RUNS):
            self._generate(image_path)

        records = []
        for _ in range(MEASURE_RUNS):
            r = self._measure_encode_ttft_decode(image_path)
            records.append(r)

        e2e_vals = [r["e2e_latency_s"] for r in records]
        med, q1, q3 = median_iqr(e2e_vals)

        # Use last run's output as the representative output
        last = records[-1]
        return {
            "output":          last["output"],
            "token_count":     last["token_count"],
            "encode_time_s":   last["encode_time_s"],
            "ttft_s":          last["ttft_s"],
            "decode_tps":      last["decode_tps"],
            "e2e_latency_s":   round(med, 4),
            "e2e_q1":          round(q1,  4),
            "e2e_q3":          round(q3,  4),
        }

    # ── Resume logic ──────────────────────────────────────────────────────────

    def _load_completed(self) -> dict:
        if self.resume and os.path.exists(self.results_file):
            try:
                df = pd.read_csv(self.results_file)
            except pd.errors.EmptyDataError:
                return {}
            if df.empty:
                return {}

            # Flexible column finder (same as judge_engine)
            def find_col(cols, keywords):
                for kw in keywords:
                    hit = next((c for c in cols if c.lower() == kw), None)
                    if hit:
                        return hit
                return next((c for c in cols if any(kw in c.lower() for kw in keywords)), None)

            image_col = find_col(df.columns, ["image", "filename", "file", "image file name"])

            if image_col is None:
                print("[Resume] No image column found — ignoring resume file.")
                return {}

            completed = {row[image_col]: row.to_dict() for _, row in df.iterrows()}
            print(f"[Resume] {len(completed)} existing results found — skipping.")
            return completed

        return {}


    # ── Main benchmark loop ───────────────────────────────────────────────────

    def run_benchmark(self):
        valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
        all_images = sorted(
            f for f in os.listdir(self.image_dir)
            if os.path.splitext(f.lower())[1] in valid_exts
        )[: self.max_images]

        if not all_images:
            print(f"[ERROR] No images in: {self.image_dir}")
            return

        completed = self._load_completed()
        pending   = [f for f in all_images if f not in completed]

        print(f"\n{'='*60}")
        print(f"  Model      : {self.model_key} ({self.precision.upper()})")
        print(f"  Device tier: {self.device_tier}  ({self.device})")
        print(f"  Dataset    : {self.dataset_name}")
        print(f"  Images     : {len(all_images)} total / {len(pending)} to process")
        print(f"  Output     : {self.results_file}")
        print(f"{'='*60}\n")

        # Measure cold-start load time (already done in __init__, so we just
        # store the value that subclasses set during _load_model)
        load_time_s = getattr(self, "_load_time_s", None)
        mem_gb      = getattr(self, "_load_mem_gb", None)

        results = list(completed.values())
        errors  = 0

        for img_name in tqdm(pending, desc=f"{self.model_key}/{self.precision}"):
            path = os.path.join(self.image_dir, img_name)
            try:
                reset_vram_stats()
                metrics = self._warm_and_measure(path)
                is_error = metrics["output"].startswith("Error:")
                if is_error:
                    errors += 1
            except Exception as e:
                metrics  = {
                    "output": f"Error: {e}", "token_count": 0,
                    "encode_time_s": None, "ttft_s": None,
                    "decode_tps": None, "e2e_latency_s": None,
                    "e2e_q1": None, "e2e_q3": None,
                }
                is_error = True
                errors  += 1

            results.append({
                "ID":             len(results),
                "dataset":        self.dataset_name,
                "image":          img_name,
                "model_id":       self.model_id,
                "model_key":      self.model_key,
                "precision":      self.precision,
                "device_tier":    self.device_tier,
                "prompt_key":     self.prompt_key,
                "vlm_output":     metrics["output"],
                "token_count":    metrics["token_count"],
                "load_time_s":    load_time_s,
                "mem_gb":         mem_gb,
                "encode_time_s":  metrics["encode_time_s"],
                "ttft_s":         metrics["ttft_s"],
                "decode_tps":     metrics["decode_tps"],
                "e2e_latency_s":  metrics["e2e_latency_s"],
                "e2e_q1":         metrics["e2e_q1"],
                "e2e_q3":         metrics["e2e_q3"],
                "is_error":       is_error,
                "timestamp":      datetime.utcnow().isoformat(),
            })

            # Crash-safe: write after every image
            pd.DataFrame(results).to_csv(self.results_file, index=False)

        # ── Run summary ───────────────────────────────────────────────────────
        df = pd.DataFrame(results)

        # Normalize is_error to boolean
        df["is_error"] = df["is_error"].fillna(False).astype(bool)

        good = df[~df["is_error"]]
        
        print(f"\n{'='*60}")
        print(f"  Done — {self.model_key.upper()} / {self.precision.upper()}")
        print(f"{'='*60}")
        print(f"  Processed   : {len(df)}  |  Errors: {errors}")
        if len(good) and good["e2e_latency_s"].notna().any():
            print(f"  Median E2E  : {good['e2e_latency_s'].median():.3f}s")
        if len(good) and good["token_count"].notna().any():
            print(f"  Avg tokens  : {good['token_count'].mean():.1f}")
        print(f"  Saved to    : {self.results_file}")
        print(f"{'='*60}\n")


# ──────────────────────────────────────────────────────────────────────────────
# SmolVLM Engine  (HuggingFaceTB/SmolVLM-500M-Instruct)
# ──────────────────────────────────────────────────────────────────────────────
class SmolVLM_Engine(VLM_Benchmarking_Engine):
    def _load_model(self):
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForVision2Seq
        except ImportError:
            from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq

        t_start = time.perf_counter()
        reset_vram_stats()

        self.processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True
        )

        load_kwargs = dict(
            trust_remote_code=True,
            _attn_implementation="eager",
        )

        if self.precision == "int4":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16

        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_id,
            **load_kwargs
        ).to(self.device)

        self.model.eval()

        self._load_time_s = round(time.perf_counter() - t_start, 3)
        self._load_mem_gb = peak_vram_gb(self.device)
        print(f"[SmolVLM] Loaded in {self._load_time_s}s | VRAM {self._load_mem_gb}GB")

    def _measure_encode_ttft_decode(self, image_path: str) -> dict:
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        max_side = max(w, h)
        if max_side > 1024:
            scale = 1024 / max_side
            image = image.resize(
                (int(w * scale), int(h * scale)), Image.Resampling.LANCZOS
            )
        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": self.prompt}]
        }]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        # ── Encode ────────────────────────────────────────────────────────────
        t_enc_start = time.perf_counter()
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt").to(self.device)
        encode_time = time.perf_counter() - t_enc_start

        # ── TTFT (separate 1-token pass) ──────────────────────────────────────
        t_ttft = time.perf_counter()
        with torch.no_grad():
            self.model.generate(**inputs, max_new_tokens=1, do_sample=False)
        ttft = time.perf_counter() - t_ttft

        # ── Full generation — E2E = encode + this ────────────────────────────
        t_full = time.perf_counter()
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=128, do_sample=False, repetition_penalty=1.3,
            )
        e2e = encode_time + (time.perf_counter() - t_full)

        input_len  = inputs["input_ids"].shape[1]
        new_tokens = generated_ids[:, input_len:]
        output     = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
        tokens     = count_tokens(output)

        torch.cuda.empty_cache()
        return {
            "output":        output,
            "encode_time_s": round(encode_time, 4),
            "ttft_s":        round(ttft, 4),
            "decode_tps":    round(tokens / e2e, 2) if e2e > 0 else None,
            "e2e_latency_s": round(e2e, 4),
            "token_count":   tokens,
        }

    def _generate(self, image_path: str) -> str:
        try:
            image = Image.open(image_path).convert("RGB")

            # Resize to avoid max_image_size errors
            w, h = image.size
            max_side = max(w, h)
            if max_side > 1024:
                scale = 1024 / max_side
                image = image.resize(
                    (int(w * scale), int(h * scale)),
                    Image.Resampling.LANCZOS
                )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": self.prompt},
                    ]
                }
            ]

            prompt = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True
            )

            inputs = self.processor(
                text=prompt,
                images=[image],
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    repetition_penalty=1.3,
                )

            input_len = inputs["input_ids"].shape[1]
            new_tokens = generated_ids[:, input_len:]
            output = self.processor.batch_decode(
                new_tokens,
                skip_special_tokens=True
            )[0].strip()

            torch.cuda.empty_cache()
            return output

        except Exception as e:
            return f"Error: {e}"
    

# ──────────────────────────────────────────────────────────────────────────────
# FastVLM Engine  (apple/FastVLM)
# ──────────────────────────────────────────────────────────────────────────────
class FastVLM_Engine(VLM_Benchmarking_Engine):

    # Override to hardcode the real HF model ID regardless of what key was passed
    REAL_MODEL_ID = "llava-hf/llava-interleave-qwen-0.5b-hf"

    def _load_model(self):
        from transformers import AutoProcessor, AutoModelForImageTextToText

        t_start = time.perf_counter()
        reset_vram_stats()

        # Use the real HF model ID, not whatever string was passed in
        hf_id = self.REAL_MODEL_ID

        self.processor = AutoProcessor.from_pretrained(
            hf_id, trust_remote_code=True
        )

        load_kwargs = dict(
            trust_remote_code=True,
            _attn_implementation="eager",
        )
        if self.precision == "int4":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16

        self.model = AutoModelForImageTextToText.from_pretrained(
            hf_id, **load_kwargs
        ).to(self.device)
        self.model.eval()

        self._load_time_s = round(time.perf_counter() - t_start, 3)
        self._load_mem_gb = peak_vram_gb(self.device)
        print(f"[FastVLM] Loaded in {self._load_time_s}s | VRAM {self._load_mem_gb}GB")  # fixed label

    def _measure_encode_ttft_decode(self, image_path: str) -> dict:
        import torch
        from PIL import Image

        # Load + resize image (FastVLM uses LLaVA-style 1024px max)
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        max_side = max(w, h)
        if max_side > 1024:
            scale = 1024 / max_side
            image = image.resize(
                (int(w * scale), int(h * scale)),
                Image.Resampling.LANCZOS
            )

        # Build chat prompt
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )

        # ───────────────────────────────────────────────
        # Encode time
        # ───────────────────────────────────────────────
        t_enc_start = time.perf_counter()
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        ).to(self.device)
        encode_time = time.perf_counter() - t_enc_start

        # ───────────────────────────────────────────────
        # TTFT (1-token generation)
        # ───────────────────────────────────────────────
        t_ttft = time.perf_counter()
        with torch.no_grad():
            self.model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False
            )
        ttft = time.perf_counter() - t_ttft

        # ───────────────────────────────────────────────
        # Full generation (E2E = encode + decode)
        # ───────────────────────────────────────────────
        t_full = time.perf_counter()
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False
            )
        e2e = encode_time + (time.perf_counter() - t_full)

        # Trim input tokens
        input_len = inputs["input_ids"].shape[1]
        new_tokens = generated_ids[:, input_len:]
        output = self.processor.batch_decode(
            new_tokens, skip_special_tokens=True
        )[0].strip()

        # Token count + TPS
        tokens = len(output.split())
        decode_tps = round(tokens / e2e, 2) if e2e > 0 else None

        torch.cuda.empty_cache()

        return {
            "output":        output,
            "encode_time_s": round(encode_time, 4),
            "ttft_s":        round(ttft, 4),
            "decode_tps":    decode_tps,
            "e2e_latency_s": round(e2e, 4),
            "token_count":   tokens,
        }

    def _generate(self, image_path: str) -> str:
        try:
            image = Image.open(image_path).convert("RGB")

            # LLaVA-Interleave expects a chat-formatted prompt
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": self.prompt},
                    ],
                }
            ]
            text_prompt = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True
            )
            inputs = self.processor(
                images=image, text=text_prompt, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs, max_new_tokens=128, do_sample=False
                )

            # Trim the input tokens so only the new generation is decoded
            trimmed = generated_ids[:, inputs["input_ids"].shape[-1]:]
            output = self.processor.batch_decode(
                trimmed, skip_special_tokens=True
            )[0].strip()

            torch.cuda.empty_cache()
            return output

        except Exception as e:
            return f"Error: {e}"


# ──────────────────────────────────────────────────────────────────────────────
# MiniCPM-V Engine  (openbmb/MiniCPM-V-2_6)
# ──────────────────────────────────────────────────────────────────────────────
class MiniCPM_Engine(VLM_Benchmarking_Engine):

    MODEL_PATH = MODELS_DIR / "minicpm-v-4.6-1.3b"

    def _load_model(self):
        import pathlib, importlib

        t_start = time.perf_counter()
        reset_vram_stats()

        local = str(self.MODEL_PATH)

        # ── Discover the correct model class from the native transformers module
        minicpm_mod = importlib.import_module("transformers.models.minicpmv4_6")
        
        # Find the ForConditionalGeneration class
        model_cls = None
        for name in dir(minicpm_mod):
            if "ForConditionalGeneration" in name or "ForCausalLM" in name:
                model_cls = getattr(minicpm_mod, name)
                print(f"[MiniCPM-V] Using class: {name}")
                break

        if model_cls is None:
            # Fallback: print all classes so we can see what's available
            classes = [n for n in dir(minicpm_mod) if not n.startswith("_")]
            raise RuntimeError(
                f"Could not find model class in transformers.models.minicpmv4_6. "
                f"Available: {classes}"
            )

        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            local,
            trust_remote_code=True,
            local_files_only=True,
        )

        load_kwargs = dict(local_files_only=True)
        if self.precision == "int4":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16

        self.model = model_cls.from_pretrained(
            local, **load_kwargs
        ).to(self.device).eval()

        self.model_id = "minicpm-v-4.6-1.3b-local"
        self._load_time_s = round(time.perf_counter() - t_start, 3)
        self._load_mem_gb = peak_vram_gb(self.device)
        print(f"[MiniCPM-V] {type(self.model).__name__} loaded in {self._load_time_s}s | VRAM {self._load_mem_gb}GB")

    def _measure_encode_ttft_decode(self, image_path: str) -> dict:
        import torch
        from PIL import Image

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Build messages (same as _generate)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text": self.prompt},
                ],
            }
        ]

        # Convert to chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # ───────────────────────────────────────────────
        # Encode time
        # ───────────────────────────────────────────────
        t_enc_start = time.perf_counter()
        inputs = self.processor(
            text=text,
            images=[image],
            return_tensors="pt",
        ).to(self.device)
        encode_time = time.perf_counter() - t_enc_start

        # ───────────────────────────────────────────────
        # TTFT (1-token generation)
        # ───────────────────────────────────────────────
        t_ttft = time.perf_counter()
        with torch.no_grad():
            self.model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
            )
        ttft = time.perf_counter() - t_ttft

        # ───────────────────────────────────────────────
        # Full generation (E2E = encode + decode)
        # ───────────────────────────────────────────────
        t_full = time.perf_counter()
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
            )
        e2e = encode_time + (time.perf_counter() - t_full)

        # Trim input tokens
        input_len  = inputs["input_ids"].shape[1]
        new_tokens = generated_ids[:, input_len:]
        output = self.processor.batch_decode(
            new_tokens, skip_special_tokens=True
        )[0].strip()

        # Token count + TPS
        tokens = len(output.split())
        decode_tps = round(tokens / e2e, 2) if e2e > 0 else None

        torch.cuda.empty_cache()

        return {
            "output":        output,
            "encode_time_s": round(encode_time, 4),
            "ttft_s":        round(ttft, 4),
            "decode_tps":    decode_tps,
            "e2e_latency_s": round(e2e, 4),
            "token_count":   tokens,
        }

    def _generate(self, image_path: str) -> str:
        try:
            image = Image.open(image_path).convert("RGB")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text",  "text": self.prompt},
                    ],
                }
            ]

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = self.processor(
                text=text,
                images=[image],
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                )

            input_len  = inputs["input_ids"].shape[1]
            new_tokens = generated_ids[:, input_len:]
            output = self.processor.batch_decode(
                new_tokens, skip_special_tokens=True
            )[0].strip()

            torch.cuda.empty_cache()
            return output

        except Exception as e:
            return f"Error: {e}"


# ──────────────────────────────────────────────────────────────────────────────
# Qwen3.5-VL Engine  (Qwen/Qwen3.5-0.8B)
# ──────────────────────────────────────────────────────────────────────────────
class Qwen35_08B_Engine(VLM_Benchmarking_Engine):

    MODEL_PATH = MODELS_DIR / "qwen3.5-0.8b"

    def _load_model(self):
        import time
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        t_start = time.perf_counter()
        reset_vram_stats()

        model_path = str(self.MODEL_PATH)

        # Load tokenizer
        self.processor = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )

        # Load model
        load_kwargs = dict(
            trust_remote_code=True,
            local_files_only=True,
        )

        if self.precision == "int4":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        else:
            load_kwargs["torch_dtype"] = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **load_kwargs
        ).to(self.device).eval()

        self.model_id = "qwen3.5-0.8b-local"
        self._load_time_s = round(time.perf_counter() - t_start, 3)
        self._load_mem_gb = peak_vram_gb(self.device)

        print(f"[Qwen3.5-0.8B] Loaded in {self._load_time_s}s | VRAM {self._load_mem_gb}GB")

    def _measure_encode_ttft_decode(self, image_path: str) -> dict:
        import torch
        from PIL import Image
        import base64
        from io import BytesIO

        # ───────────────────────────────────────────────
        # Encode image as base64 (Qwen3.5 is text-only)
        # ───────────────────────────────────────────────
        image = Image.open(image_path).convert("RGB")
        buf = BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # Build prompt
        prompt = (
            f"<image>\n"
            f"data:image/png;base64,{b64}\n"
            f"</image>\n\n"
            f"{self.prompt}"
        )

        # ───────────────────────────────────────────────
        # Encode time (tokenization)
        # ───────────────────────────────────────────────
        t_enc_start = time.perf_counter()
        inputs = self.processor(
            prompt,
            return_tensors="pt"
        ).to(self.device)
        encode_time = time.perf_counter() - t_enc_start

        # ───────────────────────────────────────────────
        # TTFT (1-token generation)
        # ───────────────────────────────────────────────
        t_ttft = time.perf_counter()
        with torch.no_grad():
            self.model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                eos_token_id=self.processor.eos_token_id,
            )
        ttft = time.perf_counter() - t_ttft

        # ───────────────────────────────────────────────
        # Full generation (E2E = encode + decode)
        # ───────────────────────────────────────────────
        t_full = time.perf_counter()
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                eos_token_id=self.processor.eos_token_id,
            )
        e2e = encode_time + (time.perf_counter() - t_full)

        # Decode output
        input_len  = inputs["input_ids"].shape[1]
        new_tokens = generated_ids[:, input_len:]
        output = self.processor.decode(new_tokens[0], skip_special_tokens=True).strip()

        # Token count + TPS
        tokens = len(output.split())
        decode_tps = round(tokens / e2e, 2) if e2e > 0 else None

        torch.cuda.empty_cache()

        return {
            "output":        output,
            "encode_time_s": round(encode_time, 4),
            "ttft_s":        round(ttft, 4),
            "decode_tps":    decode_tps,
            "e2e_latency_s": round(e2e, 4),
            "token_count":   tokens,
        }

    def _generate(self, image_path: str) -> str:
        import torch
        from PIL import Image
        import base64
        from io import BytesIO

        try:
            # Load + encode image as base64 (Qwen3.5 is text-only)
            image = Image.open(image_path).convert("RGB")
            buf = BytesIO()
            image.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            # Build prompt: your benchmark uses self.prompt
            # We embed the image as base64 inside the text prompt
            prompt = (
                f"<image>\n"
                f"data:image/png;base64,{b64}\n"
                f"</image>\n\n"
                f"{self.prompt}"
            )

            inputs = self.processor(
                prompt,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    eos_token_id=self.processor.eos_token_id,
                )

            output = self.processor.decode(
                generated_ids[0],
                skip_special_tokens=True
            ).strip()

            torch.cuda.empty_cache()
            return output

        except Exception as e:
            return f"Error: {e}"

# ──────────────────────────────────────────────────────────────────────────────
# Moondream2 Engine  (moondream2-0.5b)
# ──────────────────────────────────────────────────────────────────────────────
class Moondream2_05B_Engine(VLM_Benchmarking_Engine):

    MODEL_PATH = MODELS_DIR / "moondream2-0.5b"

    def _load_model(self):
        import time
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        t_start = time.perf_counter()
        reset_vram_stats()

        model_path = str(self.MODEL_PATH)

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )

        # Load model
        load_kwargs = dict(
            trust_remote_code=True,
            local_files_only=True,
        )

        if self.precision == "int4":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            **load_kwargs
        ).to(self.device).eval()

        self.model_id = "moondream2-0.5b-local"
        self._load_time_s = round(time.perf_counter() - t_start, 3)
        self._load_mem_gb = peak_vram_gb(self.device)

        print(f"[Moondream2-0.5B] Loaded in {self._load_time_s}s | VRAM {self._load_mem_gb}GB")

    # ----------------------------------------------------------------------
    # Full benchmark metrics (encode, TTFT, decode TPS, E2E)
    # ----------------------------------------------------------------------
    def _measure_encode_ttft_decode(self, image_path: str) -> dict:
        import torch
        import time
        from PIL import Image

        # Load + resize image (Moondream2 uses 1024px max)
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        max_side = max(w, h)
        if max_side > 1024:
            scale = 1024 / max_side
            image = image.resize(
                (int(w * scale), int(h * scale)),
                Image.Resampling.LANCZOS
            )

        # Build chat prompt
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True
        )

        # -------------------------
        # Encode time
        # -------------------------
        t_enc_start = time.perf_counter()
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        ).to(self.device)
        encode_time = time.perf_counter() - t_enc_start

        # -------------------------
        # TTFT
        # -------------------------
        t_ttft = time.perf_counter()
        with torch.no_grad():
            self.model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False
            )
        ttft = time.perf_counter() - t_ttft

        # -------------------------
        # Full generation (E2E)
        # -------------------------
        t_full = time.perf_counter()
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False
            )
        e2e = encode_time + (time.perf_counter() - t_full)

        # Trim input tokens
        input_len = inputs["input_ids"].shape[1]
        new_tokens = generated_ids[:, input_len:]
        output = self.processor.batch_decode(
            new_tokens,
            skip_special_tokens=True
        )[0].strip()

        # Token count + TPS
        tokens = len(output.split())
        decode_tps = round(tokens / e2e, 2) if e2e > 0 else None

        torch.cuda.empty_cache()

        return {
            "output":        output,
            "encode_time_s": round(encode_time, 4),
            "ttft_s":        round(ttft, 4),
            "decode_tps":    decode_tps,
            "e2e_latency_s": round(e2e, 4),
            "token_count":   tokens,
        }

    # ----------------------------------------------------------------------
    # Simple generate() fallback
    # ----------------------------------------------------------------------
    def _generate(self, image_path: str) -> str:
        import torch
        from PIL import Image

        try:
            image = Image.open(image_path).convert("RGB")

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": self.prompt},
                    ],
                }
            ]

            prompt = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True
            )

            inputs = self.processor(
                images=image,
                text=prompt,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False
                )

            input_len = inputs["input_ids"].shape[1]
            new_tokens = generated_ids[:, input_len:]
            output = self.processor.batch_decode(
                new_tokens,
                skip_special_tokens=True
            )[0].strip()

            torch.cuda.empty_cache()
            return output

        except Exception as e:
            return f"Error: {e}"

# ──────────────────────────────────────────────────────────────────────────────
# LFM2 Engine  (LFM2_450M)
# ──────────────────────────────────────────────────────────────────────────────
class LFM2_450M_Engine(VLM_Benchmarking_Engine):

    MODEL_PATH = MODELS_DIR / "LFM2-450m"

    def _load_model(self):
        import time
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        t_start = time.perf_counter()
        reset_vram_stats()

        local_path = str(self.MODEL_PATH)

        # Processor
        self.processor = AutoProcessor.from_pretrained(
            local_path,
            trust_remote_code=True,
            local_files_only=True,
        )

        # Model
        load_kwargs = dict(
            trust_remote_code=True,
            local_files_only=True,
        )
        if self.precision == "int4":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16

        self.model = AutoModelForImageTextToText.from_pretrained(
            local_path,
            **load_kwargs
        ).to(self.device).eval()

        self.model_id = "lfm2-450m-local"
        self._load_time_s = round(time.perf_counter() - t_start, 3)
        self._load_mem_gb = peak_vram_gb(self.device)
        print(f"[LFM2-450M] Loaded in {self._load_time_s}s | VRAM {self._load_mem_gb}GB")

    def _measure_encode_ttft_decode(self, image_path: str) -> dict:
        import time
        import torch
        from PIL import Image

        # Load + resize image (keep within 1024px)
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        max_side = max(w, h)
        if max_side > 1024:
            scale = 1024 / max_side
            image = image.resize(
                (int(w * scale), int(h * scale)),
                Image.Resampling.LANCZOS,
            )

        # Chat-style prompt
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
        )

        # Encode
        t_enc_start = time.perf_counter()
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        ).to(self.device)
        encode_time = time.perf_counter() - t_enc_start

        # TTFT
        t_ttft = time.perf_counter()
        with torch.no_grad():
            self.model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
            )
        ttft = time.perf_counter() - t_ttft

        # Full generation (E2E)
        t_full = time.perf_counter()
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
            )
        e2e = encode_time + (time.perf_counter() - t_full)

        # Trim input tokens
        input_len = inputs["input_ids"].shape[1]
        new_tokens = generated_ids[:, input_len:]
        output = self.processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
        )[0].strip()

        tokens = len(output.split())
        decode_tps = round(tokens / e2e, 2) if e2e > 0 else None

        torch.cuda.empty_cache()
        return {
            "output":        output,
            "encode_time_s": round(encode_time, 4),
            "ttft_s":        round(ttft, 4),
            "decode_tps":    decode_tps,
            "e2e_latency_s": round(e2e, 4),
            "token_count":   tokens,
        }

    def _generate(self, image_path: str) -> str:
        import torch
        from PIL import Image

        try:
            image = Image.open(image_path).convert("RGB")

            # Optional resize (same as measure)
            w, h = image.size
            max_side = max(w, h)
            if max_side > 1024:
                scale = 1024 / max_side
                image = image.resize(
                    (int(w * scale), int(h * scale)),
                    Image.Resampling.LANCZOS,
                )

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": self.prompt},
                    ],
                }
            ]
            prompt = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
            )

            inputs = self.processor(
                images=image,
                text=prompt,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                )

            input_len = inputs["input_ids"].shape[1]
            new_tokens = generated_ids[:, input_len:]
            output = self.processor.batch_decode(
                new_tokens,
                skip_special_tokens=True,
            )[0].strip()

            torch.cuda.empty_cache()
            return output

        except Exception as e:
            return f"Error: {e}"



# ──────────────────────────────────────────────────────────────────────────────
# Gemini Engine  (cloud baseline)
# ──────────────────────────────────────────────────────────────────────────────
class Gemini_Engine(VLM_Benchmarking_Engine):
    def _load_model(self):
        from google import genai
        from google.genai import types as genai_types

        self._genai_types = genai_types
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.gemini_model_id = "gemini-2.5-flash"
        self.gen_config = genai_types.GenerateContentConfig(
            temperature=0.0,
            system_instruction=(
                "You are an expert spatial-scene-description assistant for mixed-reality systems."
            ),
        )
        self._load_time_s = None   # cloud — no local load
        self._load_mem_gb = None
        self.device = "api"
        print("[Gemini] API client ready.")

    def _generate(self, image_path: str) -> str:
        mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                    ".png": "image/png",  ".webp": "image/webp"}
        ext      = os.path.splitext(image_path)[1].lower()
        mime     = mime_map.get(ext, "image/jpeg")
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        for attempt in range(3):
            try:
                response = self.client.models.generate_content(
                    model=self.gemini_model_id,
                    contents=[
                        self._genai_types.Part.from_bytes(
                            data=image_bytes, mime_type=mime
                        ),
                        self.prompt,
                    ],
                    config=self.gen_config,
                )
                return response.text.strip()
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    return f"Error: {e}"

    def _measure_encode_ttft_decode(self, image_path: str) -> dict:
        """For Gemini we capture network RTT + server compute as total E2E."""
        t0     = time.perf_counter()
        output = self._generate(image_path)
        e2e    = time.perf_counter() - t0
        tokens = count_tokens(output)
        return {
            "output":        output,
            "encode_time_s": None,
            "ttft_s":        None,
            "decode_tps":    None,
            "e2e_latency_s": round(e2e, 4),
            "token_count":   tokens,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Engine factory
# ──────────────────────────────────────────────────────────────────────────────
ENGINE_MAP = {
    "smolvlm":   SmolVLM_Engine,
    "fastvlm":   FastVLM_Engine,
    "minicpm":   MiniCPM_Engine,
    "qwen3.5":   Qwen35_08B_Engine,
    "moondream": Moondream2_05B_Engine,
    "gemini":    Gemini_Engine,
}

def build_engine(model_key: str, **kwargs) -> VLM_Benchmarking_Engine:
    if model_key not in ENGINE_MAP:
        raise ValueError(f"Unknown model '{model_key}'. Choose from: {list(ENGINE_MAP)}")
    return ENGINE_MAP[model_key](model_key=model_key, **kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# Path config  — edit ONLY this section when moving to a different machine
# ──────────────────────────────────────────────────────────────────────────────

# Define directories relative to the repository root
DATASETS_DIR = REPO_ROOT / "test"

DATASET_DIRS = {
    "indoor": DATASETS_DIR / "images",
    "videos": DATASETS_DIR / "videos",      
    "vlmnav": DATASETS_DIR / "vlm-nav-images",
}


# Each model gets its own results subfolder with a fixed master_results.csv.
# Gemini lives under gemini_flash/; all others under <model_key>/.
def get_results_path(model_key: str) -> tuple[str, str]:
    folder_map = {
        "gemini":   "gemini_flash",
        "smolvlm":  "smolvlm",
        "fastvlm":  "fastvlm",
        "minicpm":  "minicpm",
        "qwen3.5":  "qwen3.5",
        "moondream": "moondream", 
        "lfm2":      "lfm2", 
    }
    folder     = folder_map.get(model_key, model_key)
    results_dir = str(REPO_ROOT / "results" / folder)
    return results_dir, "master_results.csv"


# ──────────────────────────────────────────────────────────────────────────────
# Interactive model selector
# ──────────────────────────────────────────────────────────────────────────────
MODEL_LABELS = {
    "1": ("smolvlm",   "SmolVLM-500M"),
    "2": ("fastvlm",   "FastVLM-0.5B"),
    "3": ("minicpm",   "MiniCPM-V 4.6"),
    "4": ("qwen3.5",   "Qwen3.5-0.8B"),
    "5": ("moondream", "Moondream2-0.5B"),
    "6": ("lfm2",      "LFM2-450M"),
    "7": ("gemini",    "Gemini 2.5 Flash  [cloud]"),
}


PRECISION_LABELS = {
    "1": "fp16",
    "2": "int4",
}

DEVICE_LABELS = {
    "1": "Jetson",
    "2": "Consumer",
    "3": "Pro",
    "4": "Cloud",
}

DATASET_LABELS = {
    "1": "indoor",   # test/images
    "2": "vlmnav",   # test/vlm-nav-images
}


def interactive_select() -> argparse.Namespace:
    """Presents a menu so the user can pick model/precision/device without flags."""
    print("\n" + "="*55)
    print("   KSU Edge-AI VLM Benchmark — Model Selector")
    print("="*55)

    # Model
    print("\nSelect model to benchmark:")
    for k, (_, label) in MODEL_LABELS.items():
        print(f"  {k}) {label}")
    while True:
        choice = input("\nEnter number: ").strip()
        if choice in MODEL_LABELS:
            model_key, model_name = MODEL_LABELS[choice]
            break
        print("  Invalid — enter a number from the list.")

    # Precision (skip for Gemini)
    precision = "fp16"
    if model_key != "gemini":
        print(f"\nSelect precision for {model_name}:")
        for k, v in PRECISION_LABELS.items():
            print(f"  {k}) {v.upper()}")
        while True:
            choice = input("\nEnter number [default 1 = FP16]: ").strip() or "1"
            if choice in PRECISION_LABELS:
                precision = PRECISION_LABELS[choice]
                break
            print("  Invalid.")

    # Device tier
    print("\nSelect device tier (for CSV labeling):")
    for k, v in DEVICE_LABELS.items():
        tag = "  ← select this for Gemini" if v == "cloud" else ""
        print(f"  {k}) {v}{tag}")
    default_device = "4" if model_key == "gemini" else "2"
    while True:
        choice = input(f"\nEnter number [default {default_device}]: ").strip() or default_device
        if choice in DEVICE_LABELS:
            device_tier = DEVICE_LABELS[choice]
            break
        print("  Invalid.")

    # Dataset
    print("\nSelect dataset:")
    for k, v in DATASET_LABELS.items():
        print(f"  {k}) {v}")
    while True:
        choice = input("\nEnter number [default 1 = indoor]: ").strip() or "1"
        if choice in DATASET_LABELS:
            dataset = DATASET_LABELS[choice]
            break
        print("  Invalid.")

    # Resume
    resume_in = input("\nResume from existing results? [Y/n]: ").strip().lower()
    resume = resume_in != "n"

    # Confirm
    print("\n" + "-"*55)
    print(f"  Model      : {model_name}")
    print(f"  Precision  : {precision.upper()}")
    print(f"  Device tier: {device_tier}")
    print(f"  Dataset    : {dataset}")
    print(f"  Resume     : {resume}")
    print("-"*55)
    go = input("\nStart benchmark? [Y/n]: ").strip().lower()
    if go == "n":
        print("Aborted.")
        exit(0)

    ns = argparse.Namespace()
    ns.model       = model_key
    ns.precision   = precision
    ns.device_tier = device_tier
    ns.dataset     = dataset
    ns.resume      = resume
    ns.max_images  = 100
    return ns


# ──────────────────────────────────────────────────────────────────────────────
# CLI / entry point
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="VLM Spatial Benchmarking Engine",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--model",       choices=list(ENGINE_MAP),
                   help="Model key. Omit to use interactive menu.")
    p.add_argument("--precision",   default="fp16", choices=["fp16", "int4"])
    p.add_argument("--dataset", default="indoor", choices=["indoor", "vlmnav"])
    p.add_argument("--device_tier", default="T2", choices=["T1", "T2", "T3", "cloud"])
    p.add_argument("--no_resume",   action="store_true")
    p.add_argument("--max_images",  type=int, default=100)
    return p.parse_args()


if __name__ == "__main__":
    cli = parse_args()

    # If --model not supplied, drop into interactive menu
    if cli.model is None:
        args = interactive_select()
    else:
        args = cli
        args.resume = not cli.no_resume

    results_dir, out_file = get_results_path(args.model)
    image_dir             = DATASET_DIRS[args.dataset]

    engine = build_engine(
        model_key       = args.model,
        image_dir       = image_dir,
        results_dir     = results_dir,
        output_filename = out_file,
        precision       = args.precision,
        device_tier     = args.device_tier,
        dataset_name    = args.dataset,
        prompt_key      = DEFAULT_PROMPT_KEY,
        resume          = args.resume,
        max_images      = args.max_images,
    )
    engine.run_benchmark()
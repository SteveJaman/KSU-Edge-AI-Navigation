"""
judge_engine.py
================
VLM-as-a-Judge pipeline — paper §2.3

Judge backend: Gemini 2.5 Pro (multimodal — sees the actual image + description)
Evaluated models are scored independently; Gemini 2.5 Flash is just another row.

For each row in the candidate CSV:
  1. Load the image matching that row's filename from the selected image folder
  2. Send image + VLM description + rubric to Gemini 2.5 Pro judge
  3. Score 4 dimensions (1-5 each) + hallucination rate
  4. Write results crash-safely after every row

Usage:
  python vlm_judge.py --candidate gemini_flash
  python vlm_judge.py --candidate smolvlm
  python vlm_judge.py --candidate all
  python vlm_judge.py --candidate all --dataset vlmnav
"""

import os, re, time, glob, argparse, io
import pandas as pd
from PIL import Image
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
GEMINI_JUDGE_MODEL = "gemini-2.5-pro"

REPO_ROOT   = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results"
BASE_DIR    = REPO_ROOT   # alias kept for clarity

IMAGE_FOLDERS = {
    "1": ("images",         str(REPO_ROOT / "test" / "images")),
    "2": ("vlm-nav-images", str(REPO_ROOT / "test" / "vlm-nav-images")),
}

RETRIES          = 3
MAX_SCORE_TOKENS = 2048
MAX_HALL_TOKENS  = 2048
HUMAN_VAL_FRAC   = 0.20

# ─────────────────────────────────────────────────────────────────────────────
# Gemini judge singleton
# ─────────────────────────────────────────────────────────────────────────────
_client   = None
_gen_cfg  = None
_hall_cfg = None

def load_judge():
    global _client, _gen_cfg, _hall_cfg
    if _client is not None:
        return
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY not set in .env")

    _client = genai.Client(api_key=api_key)
    _sys    = "You are an expert spatial-scene-description evaluator for mixed-reality systems."

    _gen_cfg = genai_types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=MAX_SCORE_TOKENS,
        system_instruction=_sys,
    )
    _hall_cfg = genai_types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=MAX_HALL_TOKENS,
        system_instruction=_sys,
    )
    print(f"[Judge] {GEMINI_JUDGE_MODEL} ready.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Rubric — paper Appendix A verbatim
# ─────────────────────────────────────────────────────────────────────────────
RUBRIC = {
    "object_id": {
        "label": "Object/Component Identification",
        "desc": (
            "Are the key objects, components, and surfaces in the scene correctly "
            "identified (right labels, salient items present)? "
            "Measures what is recognized, not how it is arranged."
        ),
        "anchors": {
            1: "Fails to identify key objects; major misidentifications.",
            2: "Several key objects missing or misidentified; only partial recognition.",
            3: "Most key objects are identified, but several are mislabeled or missing.",
            4: "Nearly all key objects identified correctly; at most a minor omission or mislabel.",
            5: "All salient objects, components, and surfaces are correctly identified; no major omissions or misidentifications.",
        },
    },
    "spatial": {
        "label": "Spatial Relationship Correctness",
        "desc": (
            "Among the identified content, are the spatial relations correct: "
            "left/right, front/behind, above/below, near/far, depth ordering, "
            "and crucially the wearer's egocentric viewpoint?"
        ),
        "anchors": {
            1: "Relations are largely wrong or contradictory; confuses the egocentric viewpoint.",
            2: "Multiple relation errors or frequent viewpoint confusion; arrangement partly wrong.",
            3: "Overall arrangement is right but with occasional relation errors or viewpoint (left/right) flips.",
            4: "Relations mostly correct with viewpoint consistent; at most a single minor relation error.",
            5: "All described relations and relative positions are correct, including from the wearer's egocentric viewpoint; depth/ordering is right.",
        },
    },
    "safety": {
        "label": "Safety/Context Relevance",
        "desc": (
            "Does it surface the salient, actionable elements a downstream system "
            "would act on (obstacles/hazards, free/navigable space, anchorable surfaces) "
            "rather than irrelevant detail?"
        ),
        "anchors": {
            1: "Focuses on irrelevant content; omits key obstacles, navigable space, or surfaces.",
            2: "Largely irrelevant content with only occasional relevant mention.",
            3: "Mentions some relevant elements but misses important ones or dilutes them with irrelevant detail.",
            4: "Surfaces most relevant elements with little irrelevant detail.",
            5: "Surfaces the salient, actionable elements and prioritizes them.",
        },
    },
    "conciseness": {
        "label": "Conciseness",
        "desc": "Information density: useful spatial content per token, judged independently of raw length.",
        "anchors": {
            1: "Verbose, repetitive, or padded with boilerplate; low useful-information density.",
            2: "Noticeably verbose or repetitive; low density.",
            3: "Moderately efficient; some redundancy, hedging, or filler.",
            4: "Mostly efficient; minimal filler.",
            5: "Dense and efficient; nearly every clause conveys useful spatial information; no filler or repetition.",
        },
    },
}

DIMENSION_KEYS = list(RUBRIC.keys())

SPATIAL_PROMPT = (
    "Describe the scene's spatial layout in 1-3 concise sentences. "
    "Include: (a) the key objects and surfaces present; "
    "(b) their positions relative to the viewer (left/right, in-front/behind, above/below, near/far); "
    "and (c) the navigable free space, any obstacles or hazards. "
    "Describe only what is clearly visible; do not guess or invent objects."
)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────────────────────────────────────────
def _build_dim_prompt(description: str, dim_key: str, has_image: bool) -> str:
    r       = RUBRIC[dim_key]
    anchors = "\n".join(f"Score {k}: {v}" for k, v in sorted(r["anchors"].items()))
    ctx = (
        "You are given the image above, the instruction given to the VLM, "
        "the VLM's response, and a single-criterion score rubric. "
        "Use the image to verify whether the VLM's description is accurate."
        if has_image else
        "You are given the instruction given to the VLM, the VLM's response, "
        "and a single-criterion score rubric. No image is available."
    )
    return (
        f"Task Description. {ctx}\n\n"
        f"1. Write ONE sentence of feedback assessing the response against the rubric and image.\n"
        f"2. On the NEXT line write ONLY: [RESULT] N  (where N is an integer 1-5).\n"
        f"3. Do not write anything after [RESULT] N.\n\n"
        f"Instruction. {SPATIAL_PROMPT}\n\n"
        f"Response. {description}\n\n"
        f"Score Rubric — {r['label']}: {r['desc']}\n"
        f"{anchors}\n\n"
        f"Feedback (one sentence):"
    )


def _build_hall_prompt(description: str) -> str:
    return (
        "Task Description. You are given an image and a spatial scene description.\n\n"
        "Instructions:\n"
        "1. Identify each distinct spatial claim in the description.\n"
        "2. For each claim, check it against the image and write ONE line:\n"
        "   <brief claim> -> SUPPORTED or UNSUPPORTED\n"
        "3. After ALL claims, on its own line, write EXACTLY:\n"
        "   RATE: X/N\n"
        "   where X = number of UNSUPPORTED claims, N = total claims.\n"
        "4. Nothing after the RATE line.\n\n"
        f"Description: {description}\n\n"
        "Claims:"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Gemini API call
# ─────────────────────────────────────────────────────────────────────────────
def _image_to_part(image: Image.Image) -> genai_types.Part:
    img = image.copy()
    w, h = img.size
    if max(w, h) > 1024:
        scale = 1024 / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return genai_types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")


def _generate(prompt: str, image: Image.Image = None, cfg=None) -> str:
    active_cfg = cfg if cfg is not None else _gen_cfg
    contents   = [_image_to_part(image), prompt] if image is not None else [prompt]

    for attempt in range(RETRIES):
        try:
            resp = _client.models.generate_content(
                model=GEMINI_JUDGE_MODEL,
                contents=contents,
                config=active_cfg,
            )
            if resp.text is None:
                finish = ""
                if resp.candidates:
                    finish = str(getattr(resp.candidates[0], "finish_reason", ""))
                raise ValueError(f"Empty response from Gemini (finish_reason={finish})")
            return resp.text.strip()
        except Exception as e:
            if attempt < RETRIES - 1:
                wait = 2 ** attempt
                print(f"    [RETRY] attempt {attempt+1} failed ({e}), waiting {wait}s...")
                time.sleep(wait)
            else:
                raise


# ─────────────────────────────────────────────────────────────────────────────
# Parsers
# ─────────────────────────────────────────────────────────────────────────────
def _parse_score(text: str) -> int | None:
    m = re.search(r"\[RESULT\]\s*\(?([1-5])\)?", text)
    if m:
        return int(m.group(1))
    for pat in [
        r"(?:score|rating|result)[:\s]+([1-5])\b",
        r"\b([1-5])\s*/\s*5\b",
        r"^\s*([1-5])\s*$",
    ]:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return int(m.group(1))
    digits = re.findall(r"\b([1-5])\b", text)
    return int(digits[-1]) if digits else None


def _parse_hall_rate(text: str) -> float | None:
    m = re.search(r"RATE:\s*(\d+)\s*/\s*(\d+)", text, re.IGNORECASE)
    if m:
        x, n = int(m.group(1)), int(m.group(2))
        return round(x / n, 4) if n > 0 else 0.0
    return None


_HEDGE_RE = re.compile(
    r"\bpossibly\b|\bprobably\b|\bseems? to\b|\bappears? to\b|"
    r"\bmight be\b|\bcould be\b|\bi think\b|\bperhaps\b|\bit looks like\b",
    re.IGNORECASE,
)

def _hedge_proxy(text: str) -> float:
    sents = [s.strip() for s in re.split(r"[.!?]", text) if s.strip()]
    return round(sum(1 for s in sents if _HEDGE_RE.search(s)) / len(sents), 4) if sents else 0.0


def count_tokens(text: str) -> int:
    try:
        resp = _client.models.count_tokens(model=GEMINI_JUDGE_MODEL, contents=[text])
        return resp.total_tokens
    except Exception:
        return len(text.split())


# ─────────────────────────────────────────────────────────────────────────────
# Core: score one row
# ─────────────────────────────────────────────────────────────────────────────
def judge_one_row(description: str, image: Image.Image = None) -> dict:
    result = {
        "object_id":              None,
        "spatial":                None,
        "safety":                 None,
        "conciseness":            None,
        "object_id_feedback":     None,
        "spatial_feedback":       None,
        "safety_feedback":        None,
        "conciseness_feedback":   None,
        "composite":              None,
        "hallucination_rate":     None,
        "hall_method":            "n/a",
    }

    if not description or str(description).startswith("Error:"):
        result["hall_method"] = "skipped_error"
        return result

    has_image = image is not None

    # ── Metrics (1)–(4): each dimension scored independently ─────────────────
    for dim in DIMENSION_KEYS:
        prompt       = _build_dim_prompt(description, dim, has_image=has_image)
        parsed       = None
        raw_feedback = None
        for attempt in range(RETRIES):
            try:
                # FIX: pass image to every dimension call, not None
                raw    = _generate(prompt, image=image)
                parsed = _parse_score(raw)
                if parsed is not None:
                    raw_feedback = raw
                    break
                print(f"    [WARN] Score not parsed dim={dim} attempt={attempt+1}: {raw[:80]!r}")
            except Exception as e:
                print(f"    [WARN] Error dim={dim} attempt={attempt+1}: {e}")
                if attempt < RETRIES - 1:
                    time.sleep(2 ** attempt)
        result[dim]              = parsed
        result[f"{dim}_feedback"] = raw_feedback
        status = str(parsed) if parsed is not None else "FAILED"
        print(f"      → {dim:<14} = {status}")

    # ── Composite ─────────────────────────────────────────────────────────────
    valid = [result[d] for d in DIMENSION_KEYS if result[d] is not None]
    result["composite"] = round(sum(valid) / len(valid), 4) if valid else None

    # ── Metric (6): Hallucination Rate ────────────────────────────────────────
    if has_image:
        hall_rate = None
        for attempt in range(RETRIES):
            try:
                raw_hall  = _generate(_build_hall_prompt(description), image=image, cfg=_hall_cfg)
                hall_rate = _parse_hall_rate(raw_hall)
                if hall_rate is not None:
                    result["hall_method"] = "judge_grounded"
                    break
                print(f"    [WARN] Hall rate not parsed attempt={attempt+1}: {raw_hall[:80]!r}")
            except Exception as e:
                print(f"    [WARN] Hall error attempt={attempt+1}: {e}")
                if attempt < RETRIES - 1:
                    time.sleep(2 ** attempt)
        if hall_rate is None:
            hall_rate = _hedge_proxy(description)
            result["hall_method"] = "hedge_proxy_fallback"
        result["hallucination_rate"] = hall_rate
    else:
        result["hallucination_rate"] = _hedge_proxy(description)
        result["hall_method"]        = "hedge_proxy_no_image"

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Table 1 printer
# ─────────────────────────────────────────────────────────────────────────────
def print_table1(all_results: dict):
    cloud_composite = None
    for label, df in all_results.items():
        # FIX: check actual column names used in results CSV
        vlm_col = next(
            (c for c in df.columns if c.lower() in ("vlm name", "model_key", "model_id")),
            None
        )
        if vlm_col:
            names = df[vlm_col].dropna().astype(str).str.lower()
            if names.str.contains("gemini").any() and names.str.contains("flash").any():
                scored = df[df["composite"].notna()]
                if not scored.empty:
                    cloud_composite = scored["composite"].mean()
                    break

    W = 126
    print("\n" + "=" * W)
    print(
        "  Table 1: Accuracy across models and precisions, mean ± std over images.\n"
        "  Judged dims 1-5 (↑); Comp = mean of 4 dims; Hall = hallucination rate (↓);\n"
        "  Len = output tokens; ΔCloud = composite gap to Gemini 2.5 Flash."
    )
    print("=" * W)
    print(
        f"  {'Model':<22} {'Prec':<6} "
        f"{'Comp':>10} {'Obj':>10} {'Spat':>10} {'Ctx':>10} {'Conc':>10} "
        f"{'Hall':>8} {'Len':>6} {'ΔCloud':>8}"
    )
    print("-" * W)

    for label, df in sorted(all_results.items()):
        scored = df[df["composite"].notna()] if "composite" in df.columns else df
        if scored.empty:
            continue

        vlm_col  = next(
            (c for c in df.columns if c.lower() in ("vlm name", "model_key", "model_id")),
            None
        )
        prec_col = next((c for c in df.columns if "prec" in c.lower()), None)

        model_name = (
            scored[vlm_col].mode()[0]
            if vlm_col and not scored[vlm_col].isna().all()
            else label
        )
        precision = (
            scored[prec_col].mode()[0]
            if prec_col and not scored[prec_col].isna().all()
            else "—"
        )

        def ms(col):
            if col not in scored.columns or scored[col].isna().all():
                return "n/a"
            return f"{scored[col].mean():.2f}±{scored[col].std():.2f}"

        def mv(col):
            if col not in scored.columns or scored[col].isna().all():
                return None
            return scored[col].mean()

        comp_mean = mv("composite")
        hall_mean = mv("hallucination_rate")
        len_mean  = mv("token_count")
        delta     = (
            f"{comp_mean - cloud_composite:+.2f}"
            if comp_mean is not None and cloud_composite is not None
            else "n/a"
        )

        print(
            f"  {str(model_name):<22} {str(precision):<6} "
            f"{ms('composite'):>10} {ms('object_id'):>10} {ms('spatial'):>10} "
            f"{ms('safety'):>10} {ms('conciseness'):>10} "
            f"{f'{hall_mean:.3f}' if hall_mean is not None else 'n/a':>8} "
            f"{f'{len_mean:.0f}'  if len_mean  is not None else 'n/a':>6} "
            f"{delta:>8}"
        )

    print("=" * W)
    if cloud_composite is not None:
        print(f"  ΔCloud anchored to Gemini 2.5 Flash composite = {cloud_composite:.2f}")
    else:
        print("  ΔCloud: Gemini Flash row not found — judge it too to compute gaps.")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Human-validation export
# ─────────────────────────────────────────────────────────────────────────────
def export_human_validation_sample(df: pd.DataFrame, out_path: str, seed: int):
    n      = max(1, int(len(df) * HUMAN_VAL_FRAC))
    sample = df.sample(n=n, random_state=seed).copy()
    cols   = [
        "Image file name", "VLM Name", "Description",
        "object_id", "spatial", "safety", "conciseness", "composite",
        "human1_object_id", "human1_spatial", "human1_safety", "human1_conciseness",
        "human2_object_id", "human2_spatial", "human2_safety", "human2_conciseness",
    ]
    for c in cols:
        if c not in sample.columns:
            sample[c] = ""
    sample[cols].to_csv(out_path, index=False)
    print(f"  [HumanVal] {n}-row validation sample → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Image folder selector
# ─────────────────────────────────────────────────────────────────────────────
def select_image_folder(dataset: str = None) -> str:
    """Return image folder path. If dataset is given, skip interactive menu."""
    if dataset == "indoor":
        return str(REPO_ROOT / "test" / "images")
    if dataset == "vlmnav":
        return str(REPO_ROOT / "test" / "vlm-nav-images")

    # Interactive
    test_dir = str(REPO_ROOT / "test")
    known    = {path for _, path in IMAGE_FOLDERS.values()}
    extra    = {}
    next_key = len(IMAGE_FOLDERS) + 1

    if os.path.isdir(test_dir):
        for name in sorted(os.listdir(test_dir)):
            full = os.path.join(test_dir, name)
            if os.path.isdir(full) and full not in known:
                extra[str(next_key)] = (name, full)
                next_key += 1

    all_folders = {**IMAGE_FOLDERS, **extra}

    print("\n" + "=" * 58)
    print("   Select image folder (matched by filename to each CSV row)")
    print("=" * 58)
    for key, (label, path) in all_folders.items():
        status = "✓" if os.path.isdir(path) else "✗ not found"
        print(f"  {key}) {label:<25} [{status}]")
        print(f"       {path}")
    print()

    while True:
        choice = input(f"Enter number [1-{len(all_folders)}]: ").strip()
        if choice in all_folders:
            _, chosen = all_folders[choice]
            print(f"\n[Judge] Image folder: {chosen}\n")
            return chosen
        print(f"  Invalid — enter a number between 1 and {len(all_folders)}.")


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────
def run_judge(
    candidate_csv: str,
    output_csv:    str,
    image_dir:     str,
    seed:          int = 42,
) -> pd.DataFrame | None:

    if not os.path.exists(candidate_csv):
        print(f"[ERROR] CSV not found: {candidate_csv}")
        return None

    df = pd.read_csv(candidate_csv)

    def find_col(cols, keywords):
        for kw in keywords:
            hit = next((c for c in cols if c.lower() == kw), None)
            if hit:
                return hit
        return next((c for c in cols if any(kw in c.lower() for kw in keywords)), None)

    desc_col  = find_col(df.columns, ["vlm_output", "description", "desc", "output"])
    image_col = find_col(df.columns, ["image file name", "image", "filename", "file"])

    if not desc_col:
        print(f"[ERROR] No description column. Columns: {list(df.columns)}")
        return None
    if not image_col:
        print(f"[ERROR] No image column. Columns: {list(df.columns)}")
        return None

    df["_img_key"] = df[image_col].astype(str).apply(os.path.basename)

    image_lookup: dict[str, str] = {}
    if os.path.isdir(image_dir):
        for fname in os.listdir(image_dir):
            image_lookup[fname.lower()] = os.path.join(image_dir, fname)

    if not image_lookup:
        print(f"[WARN] No images found in {image_dir} — all rows will be text-only scored.")

    df = df.copy()
    df["_orig_order"] = range(len(df))
    df_judged_order   = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"\nJudging: {os.path.basename(candidate_csv)}")
    print(f"  Rows      : {len(df)}")
    print(f"  Judge     : {GEMINI_JUDGE_MODEL} (image + description per row)")
    print(f"  Desc col  : '{desc_col}'")
    print(f"  Image dir : {image_dir}  ({len(image_lookup)} images found)")
    print(f"  Output    : {output_csv}\n")

    existing = {}
    if os.path.exists(output_csv):
        try:
            ex = pd.read_csv(output_csv)
        except pd.errors.EmptyDataError:
            ex = pd.DataFrame()

        if not ex.empty:
            ex_icol = next(
                (c for c in ex.columns if c.lower() in ("image file name", "image", "filename")),
                None,
            )
            if ex_icol:
                complete, incomplete = 0, 0
                for _, r in ex.iterrows():
                    key     = os.path.basename(str(r[ex_icol]))
                    dims_ok = all(
                        pd.notna(r.get(d)) and r.get(d) is not None
                        for d in ["object_id", "spatial", "safety", "conciseness"]
                    )
                    if dims_ok:
                        existing[key] = r.to_dict()
                        complete += 1
                    else:
                        incomplete += 1
                print(f"  [Resume] {complete} complete, {incomplete} incomplete rows will be re-scored.\n")

    results = list(existing.values())
    t0      = time.time()

    for i, (_, row) in enumerate(df_judged_order.iterrows()):
        img_key     = str(row["_img_key"])
        description = str(row.get(desc_col, ""))

        if img_key in existing:
            continue

        img_path = image_lookup.get(img_key.lower())
        image    = None

        if img_path:
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"    [WARN] Cannot open {img_key}: {e}")
        else:
            print(f"    [WARN] No image match for '{img_key}' — text-only scoring")

        scores    = judge_one_row(description, image=image)
        tok_count = count_tokens(description)

        done    = i + 1 - len(existing)
        elapsed = time.time() - t0
        remain  = len(df) - len(existing) - done
        eta_str = f"  ETA {elapsed/done*remain/60:.1f}m" if done > 0 else ""

        img_ok = "🖼" if image is not None else "📄"
        print(
            f"  [{i+1:03d}/{len(df)}] {img_ok} {img_key[:34]:<34} "
            f"Comp:{str(scores.get('composite') or '?'):>5}  "
            f"Obj:{str(scores.get('object_id')  or '?'):>2}  "
            f"Spat:{str(scores.get('spatial')   or '?'):>2}  "
            f"Ctx:{str(scores.get('safety')     or '?'):>2}  "
            f"Conc:{str(scores.get('conciseness') or '?'):>2}  "
            f"Hall:{str(scores.get('hallucination_rate') or '?'):>5}"
            f"{eta_str}"
        )

        # FIX: no duplicate keys
        results.append({
            "ID":                     row.get("ID", row["_orig_order"]),
            "Image file name":        img_key,
            "VLM Name":               row.get("model_key", row.get("VLM Name", "")),
            "Description":            description,
            "object_id":              scores.get("object_id"),
            "object_id_feedback":     scores.get("object_id_feedback"),
            "spatial":                scores.get("spatial"),
            "spatial_feedback":       scores.get("spatial_feedback"),
            "safety":                 scores.get("safety"),
            "safety_feedback":        scores.get("safety_feedback"),
            "conciseness":            scores.get("conciseness"),
            "conciseness_feedback":   scores.get("conciseness_feedback"),
            "composite":              scores.get("composite"),
            "hallucination_rate":     scores.get("hallucination_rate"),
            "hall_method":            scores.get("hall_method", "n/a"),
            "token_count":            tok_count,
            "precision":              row.get("precision", ""),
            "device_tier":            row.get("device_tier", ""),
            "dataset":                row.get("dataset", ""),
            "_orig_order":            row.get("_orig_order", i),
        })

        (
            pd.DataFrame(results)
            .sort_values("_orig_order")
            .drop(columns=["_orig_order"], errors="ignore")
            .to_csv(output_csv, index=False)
        )

    df_out = (
        pd.DataFrame(results)
        .sort_values("_orig_order")
        .drop(columns=["_orig_order"], errors="ignore")
        .reset_index(drop=True)
    )

    val_csv = os.path.join(os.path.dirname(output_csv), "human_validation_sample.csv")
    export_human_validation_sample(df_out, val_csv, seed=seed)

    scored = df_out[df_out["composite"].notna()]
    print(f"\n  {'─'*58}")
    print(f"  {'Metric':<35} {'Mean':>9} {'Std':>9}")
    print(f"  {'─'*58}")
    for col, label in [
        ("composite",          "Composite (mean of 4 dims)"),
        ("object_id",          "Object/Component ID"),
        ("spatial",            "Spatial Relationship"),
        ("safety",             "Safety/Context Relevance"),
        ("conciseness",        "Conciseness"),
        ("hallucination_rate", "Hallucination Rate ↓"),
        ("token_count",        "Avg Output Tokens"),
    ]:
        if col in scored.columns and not scored[col].isna().all():
            print(f"  {label:<35} {scored[col].mean():>9.4f} {scored[col].std():>9.4f}")
    print(f"  {'─'*58}")
    print(f"  Saved → {output_csv}\n")

    return df_out


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Gemini 2.5 Pro multimodal judge for VLM spatial benchmark (paper §2.3)"
    )
    p.add_argument(
        "--candidate", required=True,
        help=(
            "Folder name under results/ (e.g. 'gemini_flash'), "
            "relative path to a master_results*.csv, "
            "or 'all' to judge every master_results.csv."
        ),
    )
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--dataset",     default=None, choices=["indoor", "vlmnav"],
                   help="Skip image folder prompt and use this dataset directly.")
    p.add_argument("--results_dir", default=None,
                   help="Override results directory (default: <repo>/results).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # FIX: use a local variable, never shadow the module-level RESULTS_DIR Path
    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_DIR

    # FIX: wire --dataset to skip interactive prompt
    image_dir = select_image_folder(dataset=args.dataset)
    if not os.path.isdir(image_dir):
        print(f"[WARN] Image dir not found — text-only scoring will be used.")

    if args.candidate.lower() == "all":
        candidate_csvs = glob.glob(
            str(results_dir / "**" / "master_results*.csv"), recursive=True
        )
        if not candidate_csvs:
            print("[ERROR] No master_results*.csv found under results/")
            exit(1)
        print(f"[INFO] Found {len(candidate_csvs)} candidate run(s).\n")
    else:
        cand = args.candidate
        if not cand.endswith(".csv"):
            matches = glob.glob(str(results_dir / cand / "master_result*.csv"))
            cand = matches[0] if matches else str(results_dir / cand / "master_results.csv")
        if not os.path.isabs(cand):
            cand = str(results_dir / cand)
        candidate_csvs = [cand]

    load_judge()

    all_judged = {}
    for cand_csv in sorted(candidate_csvs):
        out_dir   = os.path.dirname(cand_csv)
        out_csv   = os.path.join(out_dir, "judge_master_results.csv")
        run_label = os.path.basename(out_dir)

        df_judged = run_judge(cand_csv, out_csv, image_dir, seed=args.seed)
        if df_judged is not None:
            all_judged[run_label] = df_judged

    if all_judged:
        print_table1(all_judged)
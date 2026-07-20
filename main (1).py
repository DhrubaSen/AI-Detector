"""
AI Content Detector
Detects AI-generated, AI-assisted, or human-written content in text, documents, and images.
"""

import os
import io
import math
import hashlib
import re
import struct
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

# ── Optional heavy imports ────────────────────────────────────────────────────
try:
    import PyPDF2
    PDF_OK = True
except ImportError:
    PDF_OK = False

try:
    from docx import Document as DocxDocument
    DOCX_OK = True
except ImportError:
    DOCX_OK = False

try:
    from PIL import Image
    from PIL.ExifTags import TAGS
    PIL_OK = True
except ImportError:
    PIL_OK = False

# ── Cache ─────────────────────────────────────────────────────────────────────
cache: dict = {}

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="AI Content Detector", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# TEXT ANALYSIS ENGINE
# Uses statistical signals similar to GPTZero's early approach:
# perplexity (burstiness of token entropy), sentence length variance,
# punctuation patterns, vocabulary richness, hedge word density
# ─────────────────────────────────────────────────────────────────────────────

HEDGE_WORDS = {
    "however", "furthermore", "moreover", "additionally", "consequently",
    "therefore", "thus", "hence", "notably", "importantly", "significantly",
    "it is worth noting", "it should be noted", "in conclusion", "to summarize",
    "in summary", "overall", "ultimately", "essentially", "fundamentally",
    "in order to", "with respect to", "in terms of", "it is important",
    "plays a crucial role", "plays an important role", "it is essential",
    "delve", "leverage", "paradigm", "facilitate", "utilize", "implement",
    "ensure", "provide", "various", "including", "comprehensive", "robust",
    "seamlessly", "straightforward", "cutting-edge", "state-of-the-art",
}

HUMAN_MARKERS = {
    "honestly", "i think", "i feel", "i believe", "personally", "in my opinion",
    "i'm not sure", "i don't know", "maybe", "i guess", "kind of", "sort of",
    "you know", "like", "actually", "basically", "literally", "omg", "lol",
    "tbh", "ngl", "imo", "gonna", "wanna", "gotta", "kinda", "dunno",
}


def tokenize_sentences(text: str) -> list[str]:
    """Simple sentence splitter."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def compute_perplexity_proxy(text: str) -> float:
    """
    Proxy for perplexity using character-level entropy.
    AI text tends to have lower, more uniform entropy (less 'bursty').
    Human text has more variance in local entropy.
    """
    if len(text) < 50:
        return 50.0

    window = 30
    entropies = []
    for i in range(0, len(text) - window, window // 2):
        chunk = text[i:i + window]
        freq = {}
        for c in chunk:
            freq[c] = freq.get(c, 0) + 1
        h = -sum((f / window) * math.log2(f / window) for f in freq.values() if f > 0)
        entropies.append(h)

    if not entropies:
        return 50.0

    avg = sum(entropies) / len(entropies)
    variance = sum((e - avg) ** 2 for e in entropies) / len(entropies)
    # Higher variance = more human-like burstiness
    return variance


def sentence_length_variance(sentences: list[str]) -> float:
    """AI text tends to have more uniform sentence lengths."""
    if len(sentences) < 3:
        return 0.0
    lengths = [len(s.split()) for s in sentences]
    avg = sum(lengths) / len(lengths)
    variance = sum((l - avg) ** 2 for l in lengths) / len(lengths)
    return variance


def hedge_density(text: str) -> float:
    """AI text uses hedge words and formal transitions more frequently."""
    words = text.lower().split()
    if not words:
        return 0.0
    count = sum(1 for w in words if w.rstrip('.,;:') in HEDGE_WORDS)
    # Also check phrases
    text_lower = text.lower()
    for phrase in ["it is worth noting", "it should be noted", "in conclusion",
                   "to summarize", "in summary", "plays a crucial role",
                   "plays an important role", "it is important", "it is essential"]:
        if phrase in text_lower:
            count += 1
    return count / max(len(words), 1)


def human_marker_density(text: str) -> float:
    """Human text uses colloquial markers more frequently."""
    words = text.lower().split()
    if not words:
        return 0.0
    count = sum(1 for w in words if w.rstrip('.,;:!?') in HUMAN_MARKERS)
    return count / max(len(words), 1)


def vocabulary_richness(text: str) -> float:
    """Type-token ratio — AI can be repetitive at paragraph level."""
    words = re.findall(r'\b[a-z]+\b', text.lower())
    if len(words) < 10:
        return 1.0
    return len(set(words)) / len(words)


def repetition_score(sentences: list[str]) -> float:
    """Detect repeated sentence structures (common in AI)."""
    if len(sentences) < 4:
        return 0.0
    # Check first word patterns
    starts = [s.split()[0].lower() if s.split() else '' for s in sentences]
    unique_starts = len(set(starts))
    repetition = 1.0 - (unique_starts / len(starts))
    return repetition


def analyse_text(text: str) -> dict:
    """
    Main text analysis. Returns scores and classification.
    """
    if len(text.strip()) < 50:
        return {
            "label": "insufficient_text",
            "confidence": 0,
            "explanation": "Text is too short for reliable analysis (minimum 50 characters).",
            "signals": {}
        }

    sentences = tokenize_sentences(text)
    perp = compute_perplexity_proxy(text)
    sl_var = sentence_length_variance(sentences)
    hedge = hedge_density(text)
    human = human_marker_density(text)
    vocab = vocabulary_richness(text)
    rep = repetition_score(sentences)

    # ── Scoring ───────────────────────────────────────────────────────────────
    # Each signal contributes to AI probability (0-1)

    # Low perplexity variance → AI-like
    perp_score = max(0, 1 - (perp / 3.0))  # normalize: variance > 3 = human

    # Low sentence length variance → AI-like
    sl_score = max(0, 1 - (sl_var / 50.0))  # normalize: var > 50 = human

    # High hedge density → AI-like
    hedge_score = min(1.0, hedge * 15)

    # High human markers → human
    human_score = min(1.0, human * 20)  # inverted: high = human

    # Low vocab richness → AI-like (repetitive)
    vocab_score = max(0, 1 - vocab)

    # High repetition → AI-like
    rep_score = rep

    # Weighted ensemble
    ai_probability = (
        0.25 * perp_score +
        0.15 * sl_score +
        0.25 * hedge_score +
        0.15 * vocab_score +
        0.10 * rep_score +
        0.10 * (1 - human_score)  # absence of human markers
    )

    # Human markers strongly push toward human
    if human_score > 0.1:
        ai_probability = max(0, ai_probability - human_score * 0.3)

    ai_probability = round(min(1.0, max(0.0, ai_probability)), 3)

    # ── Classification ────────────────────────────────────────────────────────
    if ai_probability >= 0.70:
        label = "ai_generated"
        confidence = round(ai_probability * 100)
        explanation = (
            f"High probability of AI generation ({confidence}%). "
            f"Signals: low entropy variance, formal transition words, "
            f"uniform sentence structure."
        )
    elif ai_probability >= 0.45:
        label = "ai_assisted"
        confidence = round(ai_probability * 100)
        explanation = (
            f"Mixed signals ({confidence}% AI probability). "
            f"Likely human-written with AI editing, or AI output with human revision."
        )
    else:
        label = "human_written"
        confidence = round((1 - ai_probability) * 100)
        explanation = (
            f"Likely human-written ({confidence}% confidence). "
            f"Natural language variance, colloquial markers, vocabulary richness detected."
        )

    return {
        "label": label,
        "confidence": confidence,
        "ai_probability": ai_probability,
        "explanation": explanation,
        "signals": {
            "entropy_variance": round(perp, 4),
            "sentence_length_variance": round(sl_var, 2),
            "hedge_word_density": round(hedge, 4),
            "human_marker_density": round(human, 4),
            "vocabulary_richness": round(vocab, 4),
            "repetition_score": round(rep, 4),
        },
        "word_count": len(text.split()),
        "sentence_count": len(sentences),
    }


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE ANALYSIS ENGINE
# Checks EXIF metadata, file structure, color statistics, and noise patterns
# ─────────────────────────────────────────────────────────────────────────────

AI_IMAGE_TOOLS = [
    "stable diffusion", "midjourney", "dall-e", "dalle", "firefly",
    "imagen", "flux", "kandinsky", "runwayml", "leonardo", "nightcafe",
    "adobe firefly", "ideogram", "adobe generative", "generative fill",
]

CAMERA_MARKERS = [
    "make", "model", "gps", "datetime", "exposuretime", "fnumber",
    "isospeedratings", "flash", "focallength", "lensmodel", "lensmake",
]


def analyse_image(content: bytes) -> dict:
    """Analyse image for AI generation signals."""
    signals = {}
    ai_indicators = []
    human_indicators = []

    if not PIL_OK:
        return {
            "label": "unknown",
            "confidence": 0,
            "explanation": "PIL not available for image analysis.",
            "signals": {}
        }

    try:
        img = Image.open(io.BytesIO(content))
    except Exception as e:
        return {
            "label": "error",
            "confidence": 0,
            "explanation": f"Could not open image: {str(e)}",
            "signals": {}
        }

    # ── 1. EXIF metadata analysis ─────────────────────────────────────────────
    exif_data = {}
    has_camera_exif = False
    has_ai_software_tag = False

    try:
        raw_exif = img._getexif()
        if raw_exif:
            for tag_id, value in raw_exif.items():
                tag = TAGS.get(tag_id, str(tag_id))
                exif_data[tag.lower()] = str(value)[:200]

            # Check for camera markers
            for marker in CAMERA_MARKERS:
                if marker in exif_data:
                    has_camera_exif = True
                    break

            # Check for AI software tags
            software = exif_data.get("software", "").lower()
            for tool in AI_IMAGE_TOOLS:
                if tool in software:
                    has_ai_software_tag = True
                    ai_indicators.append(f"Software tag: {software}")
                    break

            if has_camera_exif:
                human_indicators.append("Camera EXIF data present (Make, Model, GPS, etc.)")
            elif not has_ai_software_tag:
                ai_indicators.append("No camera EXIF metadata (typical of AI-generated images)")
        else:
            ai_indicators.append("No EXIF metadata found")
    except Exception:
        ai_indicators.append("Could not read EXIF metadata")

    signals["has_camera_exif"] = has_camera_exif
    signals["has_ai_software_tag"] = has_ai_software_tag
    signals["exif_fields"] = list(exif_data.keys())[:10]

    # ── 1b. Screenshot/document detection ────────────────────────────────────
    # Screenshots and document exports are PNG, small file size, mostly white/black,
    # very low color variance — do NOT classify as AI generated
    file_size_kb = len(content) / 1024
    signals["file_size_kb"] = round(file_size_kb, 1)

    # ── 2. Image statistics ───────────────────────────────────────────────────
    try:
        img_rgb = img.convert("RGB")
        width, height = img_rgb.size
        signals["dimensions"] = f"{width}x{height}"

        # Check for common AI image dimensions
        ai_dims = [(1024, 1024), (512, 512), (768, 768), (1024, 768),
                   (768, 1024), (1024, 1536), (1536, 1024), (2048, 2048)]
        if (width, height) in ai_dims:
            ai_indicators.append(f"Common AI generation resolution: {width}x{height}")
        elif width == height:
            ai_indicators.append(f"Square image ({width}x{height}) — common in AI generation")

        # Pixel statistics
        pixels = list(img_rgb.getdata())
        sample = pixels[::max(1, len(pixels)//10000)]  # sample 10K pixels max

        r_vals = [p[0] for p in sample]
        g_vals = [p[1] for p in sample]
        b_vals = [p[2] for p in sample]

        def stats(vals):
            avg = sum(vals) / len(vals)
            var = sum((v - avg) ** 2 for v in vals) / len(vals)
            return avg, math.sqrt(var)

        r_avg, r_std = stats(r_vals)
        g_avg, g_std = stats(g_vals)
        b_avg, b_std = stats(b_vals)

        signals["color_stats"] = {
            "r": {"mean": round(r_avg, 1), "std": round(r_std, 1)},
            "g": {"mean": round(g_avg, 1), "std": round(g_std, 1)},
            "b": {"mean": round(b_avg, 1), "std": round(b_std, 1)},
        }

        # AI images often have very smooth gradients (lower std in local patches)
        # and very high overall color diversity
        channel_stds = [r_std, g_std, b_std]
        avg_std = sum(channel_stds) / 3
        signals["avg_channel_std"] = round(avg_std, 2)

        if avg_std > 60:
            human_indicators.append(f"High color variance ({avg_std:.1f}) — natural photograph characteristic")
        elif avg_std < 35:
            ai_indicators.append(f"Low color variance ({avg_std:.1f}) — smooth gradients typical of AI generation")

    except Exception as e:
        signals["pixel_analysis_error"] = str(e)

    # ── 3. File format analysis ───────────────────────────────────────────────
    fmt = img.format or "unknown"
    signals["format"] = fmt
    signals["mode"] = img.mode

    if fmt == "PNG" and not has_camera_exif:
        ai_indicators.append("PNG format without camera EXIF — common AI output format")
    elif fmt in ("JPEG", "JPG") and has_camera_exif:
        human_indicators.append("JPEG with camera metadata — consistent with real photograph")

    # ── 3b. Screenshot / document export detection ───────────────────────────
    # Fires when: PNG + no EXIF + tiny file + mostly white/black pixels
    # These are screenshots, Word exports, diagram exports — NOT AI art
    is_likely_screenshot = False
    try:
        is_png = (signals.get("format", "") == "PNG")
        is_small = file_size_kb < 100  # AI images are rarely under 100KB
        is_low_color = signals.get("avg_channel_std", 100) < 30
        no_exif = not has_camera_exif

        # Check if image is mostly white/light background
        if 'color_stats' in signals:
            r_mean = signals['color_stats']['r']['mean']
            g_mean = signals['color_stats']['g']['mean']
            b_mean = signals['color_stats']['b']['mean']
            avg_brightness = (r_mean + g_mean + b_mean) / 3
            is_mostly_white = avg_brightness > 180  # very light/white background
        else:
            is_mostly_white = False

        if is_png and no_exif and is_small and is_low_color and is_mostly_white:
            is_likely_screenshot = True
            # Remove incorrect AI signals — these are normal for screenshots
            ai_indicators = [i for i in ai_indicators if 'EXIF' not in i and 'color variance' not in i]
            human_indicators.append(f"Screenshot or document export detected (PNG, {file_size_kb:.1f}KB, white background, low color complexity)")
    except Exception:
        pass

    # ── Classification ────────────────────────────────────────────────────────
    ai_score = len(ai_indicators)
    human_score = len(human_indicators)
    total = ai_score + human_score

    if is_likely_screenshot:
        return {
            "label": "human_created",
            "confidence": 85,
            "explanation": f"This appears to be a screenshot, diagram, or document export — not an AI-generated image. Signals: small file size ({file_size_kb:.1f}KB), white/light background, PNG format, no camera metadata.",
            "ai_indicators": ai_indicators,
            "human_indicators": human_indicators,
            "signals": signals,
        }

    if has_ai_software_tag:
        label = "ai_generated"
        confidence = 95
        explanation = f"AI generation tool detected in image metadata. {'; '.join(ai_indicators)}"
    elif total == 0:
        label = "unknown"
        confidence = 50
        explanation = "Insufficient signals to classify this image reliably."
    elif ai_score > human_score * 1.5:
        label = "ai_generated"
        confidence = min(90, 55 + ai_score * 8)
        explanation = f"Multiple AI generation signals detected: {'; '.join(ai_indicators)}"
    elif human_score > ai_score:
        label = "human_captured"
        confidence = min(85, 55 + human_score * 10)
        explanation = f"Photograph characteristics detected: {'; '.join(human_indicators)}"
    else:
        label = "uncertain"
        confidence = 50
        explanation = f"Mixed signals. AI: {'; '.join(ai_indicators)}. Human: {'; '.join(human_indicators)}"

    return {
        "label": label,
        "confidence": confidence,
        "explanation": explanation,
        "ai_indicators": ai_indicators,
        "human_indicators": human_indicators,
        "signals": signals,
    }


# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENT TEXT EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_from_pdf(content: bytes) -> str:
    if not PDF_OK:
        raise HTTPException(status_code=500, detail="PyPDF2 not installed")
    reader = PyPDF2.PdfReader(io.BytesIO(content))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.strip()


def extract_text_from_docx(content: bytes) -> str:
    if not DOCX_OK:
        raise HTTPException(status_code=500, detail="python-docx not installed")
    doc = DocxDocument(io.BytesIO(content))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


# ─────────────────────────────────────────────────────────────────────────────
# API ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    static_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "index.html")
    return HTMLResponse(
        content=open(static_path, encoding="utf-8").read(),
        headers={"ngrok-skip-browser-warning": "true"}
    )


@app.post("/api/check-text")
async def check_text(payload: dict):
    """Analyse plain text for AI generation."""
    text = payload.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    cache_key = hashlib.sha256(text.encode()).hexdigest()
    if cache_key in cache:
        return cache[cache_key]

    result = analyse_text(text)
    cache[cache_key] = result
    return result


@app.post("/api/check-document")
async def check_document(file: UploadFile = File(...)):
    """Analyse PDF, DOCX, or TXT document for AI generation."""
    content = await file.read()
    cache_key = hashlib.sha256(content).hexdigest()
    if cache_key in cache:
        return cache[cache_key]

    filename = file.filename.lower()

    try:
        if filename.endswith(".pdf"):
            text = extract_text_from_pdf(content)
            doc_type = "PDF"
        elif filename.endswith(".docx"):
            text = extract_text_from_docx(content)
            doc_type = "Word Document"
        elif filename.endswith(".txt"):
            text = content.decode("utf-8", errors="ignore")
            doc_type = "Plain Text"
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {filename}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting text: {str(e)}")

    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from document")

    result = analyse_text(text)
    result["document_type"] = doc_type
    result["filename"] = file.filename
    cache[cache_key] = result
    return result


@app.post("/api/check-image")
async def check_image(file: UploadFile = File(...)):
    """Analyse image for AI generation signals."""
    content = await file.read()
    cache_key = hashlib.sha256(content).hexdigest()
    if cache_key in cache:
        return cache[cache_key]

    filename = file.filename.lower()
    allowed = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff")
    if not any(filename.endswith(ext) for ext in allowed):
        raise HTTPException(status_code=400, detail=f"Unsupported image format: {filename}")

    result = analyse_image(content)
    result["filename"] = file.filename
    cache[cache_key] = result
    return result


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "capabilities": {
            "pdf": PDF_OK,
            "docx": DOCX_OK,
            "image": PIL_OK,
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# STATIC FILES
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "static"), exist_ok=True)
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)

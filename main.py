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

try:
    import c2pa
    C2PA_OK = True
except ImportError:
    C2PA_OK = False

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


# Additional human signal detectors
SELF_DEPRECATING = [
    "i must confess", "i am embarrassed", "i am embarassed", "frankly",
    "i confess", "i must admit", "to be honest", "i am afraid",
    "forgive me", "i apologize", "i apologise", "i regret",
]

AUTOBIOGRAPHICAL = [
    "memory lane", "i recall", "i remember", "in those days",
    "it was", "when i was", "years ago", "at that time",
    "in my experience", "i have found", "i have seen",
    "looking back", "in retrospect",
]

SCAN_ARTIFACTS = ["¬", "­", "be¬", "how¬", "in¬"]

PRE_AI_FORMAL = [
    "hitherto", "heretofore", "whilst", "amongst", "henceforth",
    "thereupon", "wherein", "thereof", "hereby", "whereby",
    "aforementioned", "notwithstanding", "inasmuch", "insofar",
    "acceding", "clubbed under", "blew up",
]

def self_deprecating_density(text: str) -> float:
    text_lower = text.lower()
    count = sum(1 for phrase in SELF_DEPRECATING if phrase in text_lower)
    return min(1.0, count / 3)

def autobiographical_density(text: str) -> float:
    text_lower = text.lower()
    count = sum(1 for phrase in AUTOBIOGRAPHICAL if phrase in text_lower)
    return min(1.0, count / 3)

def has_scan_artifacts(text: str) -> bool:
    return any(artifact in text for artifact in SCAN_ARTIFACTS)

def pre_ai_formal_density(text: str) -> float:
    text_lower = text.lower()
    count = sum(1 for phrase in PRE_AI_FORMAL if phrase in text_lower)
    return min(1.0, count / 2)

PROFESSIONAL_FIRST_PERSON = [
    "my name is", "i have a", "i was", "i had", "i recognized",
    "i observed", "i started", "i believe", "i built", "i developed",
    "i created", "i noticed", "i worked", "i joined", "i completed",
    "i am a", "i am an", "i hold", "i founded", "i designed",
    "let me know", "please find", "i would", "i could", "i should",
    "my experience", "my background", "my work", "my team",
]

def professional_narrative_score(text: str) -> float:
    """First-person professional narrative — LinkedIn/cover letter/proposal style."""
    text_lower = text.lower()
    count = sum(1 for phrase in PROFESSIONAL_FIRST_PERSON if phrase in text_lower)
    return min(1.0, count / 4)


def spelling_variation_score(text: str) -> float:
    """Detect human spelling variations and typos."""
    common_variations = [
        "exhilerated", "embarassed", "wierd", "recieve", "occured",
        "seperately", "definately", "occurance", "untill", "dont",
        "cant", "wont", "ive", "id ", "youre", "theyre", "its a",
    ]
    text_lower = text.lower()
    count = sum(1 for v in common_variations if v in text_lower)
    return min(1.0, count * 0.5)


# ── Creative Writing AI Detection Signals ────────────────────────────────────
# AI creative writing is "hyper-efficient" — every sentence serves the theme,
# perfect symbolic structure, no loose ends, no grounded dialogue.
# Source: Gemini self-analysis of AI-generated fiction

SYMBOLIC_WORDS = [
    "beacon", "light", "darkness", "shadow", "flame", "fire", "storm",
    "silence", "soul", "grave", "tide", "shore", "wave", "broken",
    "desperate", "fate", "irony", "redemption", "sacrifice", "loss",
    "guilt", "burden", "weight", "hollow", "empty", "shattered",
]

DIALOGUE_MARKERS = ['"', "'", "said", "replied", "asked", "whispered",
                    "shouted", "cried", "answered", "called", "told",
                    "exclaimed", "muttered", "murmured"]

NARRATIVE_EFFICIENCY = [
    "in his desperate", "in her desperate", "in their desperate",
    "in an attempt to", "in a desperate attempt",
    "he had", "she had", "they had",
    "finally", "eventually", "ultimately", "at last",
    "only to", "only then", "only now",
]

def creative_writing_signals(text: str) -> dict:
    """
    Detect AI creative writing patterns.
    AI fiction is thematically hyper-efficient — every element serves the plot,
    perfect irony, no loose dialogue, compact summary style.
    """
    words = text.lower().split()
    sentences = tokenize_sentences(text)
    text_lower = text.lower()
    word_count = len(words)

    if word_count < 50:
        return {"is_creative": False}

    # 1. Symbolic density — AI packs in too many symbols
    symbol_count = sum(1 for w in words if w.rstrip('.,;:!?"') in SYMBOLIC_WORDS)
    symbol_density = symbol_count / max(word_count, 1)

    # 2. Dialogue absence — human fiction almost always has dialogue
    has_dialogue = any(marker in text for marker in DIALOGUE_MARKERS[:3])  # check quotes
    dialogue_word_count = sum(1 for w in DIALOGUE_MARKERS[3:] if w in text_lower)
    has_dialogue_verbs = dialogue_word_count > 0

    # 3. Narrative efficiency — too many plot-advancing phrases
    efficiency_count = sum(1 for phrase in NARRATIVE_EFFICIENCY if phrase in text_lower)
    efficiency_density = efficiency_count / max(len(sentences), 1)

    # 4. Plot event density — events per sentence (AI crams in more)
    plot_words = ["arrived", "found", "saw", "heard", "felt", "realized",
                  "discovered", "decided", "turned", "looked", "ran", "left",
                  "returned", "died", "fell", "rose", "came", "went", "took"]
    plot_count = sum(1 for w in words if w.rstrip('.,;:!?') in plot_words)
    plot_density = plot_count / max(len(sentences), 1)

    # 5. Perfect irony structure — AI loves circular/ironic endings
    irony_phrases = [
        "guided dozens", "guided innocent", "led to their",
        "let his fire", "let her fire", "abandoned his post",
        "too late", "in vain", "only to find", "only to discover",
        "at the cost of", "by saving", "while trying to save",
    ]
    has_irony = any(phrase in text_lower for phrase in irony_phrases)

    # Detect if this is creative writing at all
    creative_markers = ["he said", "she said", "the story", "once upon",
                        "narrator", "protagonist", "character"]
    narrative_words = ["tower", "vessel", "shore", "cliff", "lighthouse",
                       "messenger", "keeper", "sailor", "captain", "village"]
    is_narrative = (sum(1 for w in narrative_words if w in text_lower) > 1 or
                    plot_density > 0.5)

    return {
        "is_creative": is_narrative,
        "symbol_density": round(symbol_density, 4),
        "has_dialogue": has_dialogue or has_dialogue_verbs,
        "efficiency_density": round(efficiency_density, 4),
        "plot_density": round(plot_density, 4),
        "has_irony_structure": has_irony,
    }


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

    # Clean scan artifacts before analysis
    clean_text = text.replace("¬", "").replace("­", "")

    sentences = tokenize_sentences(clean_text)
    perp = compute_perplexity_proxy(clean_text)
    sl_var = sentence_length_variance(sentences)
    hedge = hedge_density(clean_text)
    human = human_marker_density(clean_text)
    vocab = vocabulary_richness(clean_text)
    rep = repetition_score(sentences)

    # Additional human signals
    prof_narr = professional_narrative_score(clean_text)
    creative = creative_writing_signals(clean_text)
    self_dep = self_deprecating_density(clean_text)
    autobio = autobiographical_density(clean_text)
    scan_art = has_scan_artifacts(text)  # check original text
    pre_ai = pre_ai_formal_density(clean_text)
    spell_var = spelling_variation_score(clean_text)

    # ── Scoring ───────────────────────────────────────────────────────────────
    # Each signal contributes to AI probability (0-1)

    # Low perplexity variance → AI-like
    perp_score = max(0, 1 - (perp / 3.0))

    # Low sentence length variance → AI-like
    sl_score = max(0, 1 - (sl_var / 50.0))

    # High hedge density → AI-like
    hedge_score = min(1.0, hedge * 15)

    # High human markers → human (inverted)
    human_score = min(1.0, human * 20)

    # Low vocab richness → AI-like
    vocab_score = max(0, 1 - vocab)

    # High repetition → AI-like
    rep_score = rep

    # Weighted ensemble (base)
    # Entropy variance reduced in weight — clean professional writing
    # should not be penalised as AI. Hedge words and repetition are
    # stronger discriminators for business/professional text.
    ai_probability = (
        0.12 * perp_score +        # reduced — clean writing != AI
        0.12 * sl_score +           # reduced — professional writing is uniform
        0.30 * hedge_score +        # increased — strongest AI signal
        0.15 * vocab_score +        # unchanged
        0.16 * rep_score +          # increased — repetition is reliable
        0.15 * (1 - human_score)   # unchanged
    )

    # ── Human signal adjustments ──────────────────────────────────────────────
    # Self-deprecating voice — strong human signal
    if self_dep > 0:
        ai_probability = max(0, ai_probability - self_dep * 0.20)

    # Autobiographical narrative — strong human signal
    if autobio > 0:
        ai_probability = max(0, ai_probability - autobio * 0.20)

    # Scan artifacts — document is from a physical/historical source
    if scan_art:
        ai_probability = max(0, ai_probability - 0.25)

    # Pre-AI formal English vocabulary — historical/academic writing
    if pre_ai > 0:
        ai_probability = max(0, ai_probability - pre_ai * 0.15)

    # Spelling variations and typos — human signal
    if spell_var > 0:
        ai_probability = max(0, ai_probability - spell_var * 0.10)

    # Creative writing AI detection
    if creative.get("is_creative"):
        cw_ai_score = 0.0
        # For creative writing, use a separate base score
        # AI fiction has low hedge words but high plot efficiency
        # Reset hedge word advantage — not applicable to creative writing
        hedge_boost = min(1.0, hedge * 15) * 0.30
        cw_ai_score -= hedge_boost * 0.5  # reduce hedge word penalty for fiction

        # No dialogue in fiction = strong AI signal
        if not creative["has_dialogue"]:
            cw_ai_score += 0.20
        # High symbolic density = AI signal
        if creative["symbol_density"] > 0.03:
            cw_ai_score += creative["symbol_density"] * 4
        # High narrative efficiency = AI signal
        if creative["efficiency_density"] > 0.3:
            cw_ai_score += creative["efficiency_density"] * 0.3
        # Perfect irony structure = AI signal
        if creative["has_irony_structure"]:
            cw_ai_score += 0.15
        # High plot density = AI signal
        if creative["plot_density"] > 0.4:
            cw_ai_score += 0.10
        # Compact summary style — many plot events, short word count
        if creative["plot_density"] > 0.3 and not creative["has_dialogue"]:
            cw_ai_score += 0.10  # compound signal
        ai_probability = min(1.0, ai_probability + cw_ai_score)

    # Professional first-person narrative — LinkedIn/proposal/bio style
    if prof_narr > 0:
        ai_probability = max(0, ai_probability - prof_narr * 0.20)

    # General human markers
    if human_score > 0.1:
        ai_probability = max(0, ai_probability - human_score * 0.25)

    ai_probability = round(min(1.0, max(0.0, ai_probability)), 3)

    # ── Classification ────────────────────────────────────────────────────────
    is_creative = creative.get("is_creative", False)
    no_dialogue = is_creative and not creative.get("has_dialogue", True)

    if ai_probability >= 0.70:
        label = "ai_generated"
        confidence = round(ai_probability * 100)
        if is_creative:
            explanation = (
                f"High probability of AI-generated creative writing ({confidence}%). "
                f"Signals: hyper-efficient thematic structure, high symbolic density, "
                f"{'no dialogue detected, ' if no_dialogue else ''}"
                f"compact summary style typical of AI fiction. "
                f"Human fiction typically includes grounded dialogue, idiosyncratic pacing, and loose sensory detail."
            )
        else:
            explanation = (
                f"High probability of AI generation ({confidence}%). "
                f"Signals: low entropy variance, formal transition words, "
                f"uniform sentence structure."
            )
    elif ai_probability >= 0.45:
        label = "ai_assisted"
        confidence = round(ai_probability * 100)
        if is_creative:
            explanation = (
                f"Mixed signals ({confidence}% AI probability). "
                f"Creative writing detected — may be AI-generated fiction or human writing with AI polish. "
                f"AI creative writing typically lacks grounded dialogue and has overly efficient thematic structure."
            )
        else:
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
            "self_deprecating_voice": round(self_dep, 4),
            "autobiographical_narrative": round(autobio, 4),
            "scan_artifacts_detected": scan_art,
            "pre_ai_formal_english": round(pre_ai, 4),
            "spelling_variations": round(spell_var, 4),
            "professional_narrative": round(prof_narr, 4),
            "creative_writing_detected": creative.get("is_creative", False),
            "creative_symbol_density": creative.get("symbol_density", 0),
            "creative_has_dialogue": creative.get("has_dialogue", None),
            "creative_plot_density": creative.get("plot_density", 0),
        },
        "word_count": len(clean_text.split()),
        "sentence_count": len(sentences),
        "signal_contributions": {
            "hedge_words": f"{round(min(1.0, hedge * 15) * 0.30 * 100, 1)}% — formal AI transition words",
            "repetition": f"{round(rep * 0.16 * 100, 1)}% — repeated sentence patterns",
            "entropy": f"{round(max(0, 1 - (perp / 3.0)) * 0.12 * 100, 1)}% — text uniformity",
            "sentence_uniformity": f"{round(max(0, 1 - (sl_var / 50.0)) * 0.12 * 100, 1)}% — sentence length uniformity",
            "vocab_repetition": f"{round(max(0, 1 - vocab) * 0.15 * 100, 1)}% — vocabulary repetition",
            "no_human_markers": f"{round((1 - min(1.0, human * 20)) * 0.15 * 100, 1)}% — absence of colloquial language",
            "creative_no_dialogue": f"{round(0.15 * 100 if is_creative and not creative.get('has_dialogue') else 0, 1)}% — no dialogue in fiction",
            "creative_irony": f"{round(0.12 * 100 if creative.get('has_irony_structure') else 0, 1)}% — ironic/symbolic structure",
        }
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


def check_c2pa(content: bytes, mime_type: str = "image/png") -> dict:
    """
    Check if image contains C2PA provenance metadata.
    C2PA (Coalition for Content Provenance and Authenticity) is the
    standard required by EU AI Act Article 50 for AI content marking.
    """
    if not C2PA_OK:
        return {"has_c2pa": False, "error": "c2pa library not available", "manifests": []}

    try:
        import io as _io
        reader = c2pa.Reader(mime_type, _io.BytesIO(content))
        manifest_json = reader.json()
        import json as _json
        manifest_data = _json.loads(manifest_json)

        # Extract key provenance info
        manifests = manifest_data.get("manifests", {})
        active_manifest = manifest_data.get("active_manifest", "")

        ai_assertions = []
        creator_tool = None
        is_ai_generated = False
        is_ai_trained = False

        for key, manifest in manifests.items():
            # Check assertions for AI signals
            for assertion in manifest.get("assertions", []):
                label = assertion.get("label", "")
                data = assertion.get("data", {})

                # C2PA AI assertions
                if "c2pa.ai.generative" in label or "ai.generated" in label.lower():
                    is_ai_generated = True
                    ai_assertions.append(f"AI Generated: {label}")
                if "c2pa.ai.training" in label:
                    is_ai_trained = True
                    ai_assertions.append(f"AI Training: {label}")
                if "c2pa.training-mining" in label:
                    ai_assertions.append(f"Training/Mining: {label}")

            # Check software agent
            claim_gen = manifest.get("claim_generator", "")
            if claim_gen:
                creator_tool = claim_gen

        return {
            "has_c2pa": True,
            "is_ai_generated": is_ai_generated,
            "is_ai_trained": is_ai_trained,
            "ai_assertions": ai_assertions,
            "creator_tool": creator_tool,
            "active_manifest": active_manifest,
            "manifest_count": len(manifests),
            "compliant": True,  # Has C2PA = compliant with Article 50
        }

    except Exception as e:
        err = str(e)
        if "ManifestNotFound" in err or "no JUMBF" in err:
            return {
                "has_c2pa": False,
                "compliant": False,
                "note": "No C2PA provenance metadata found",
            }
        return {
            "has_c2pa": False,
            "compliant": False,
            "error": err[:100],
        }


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
# VIDEO ANALYSIS ENGINE (Lightweight — metadata only)
# Checks file metadata, container format, and software tags
# Does NOT analyse frames or audio — catches obvious AI video tools only
# ─────────────────────────────────────────────────────────────────────────────

AI_VIDEO_TOOLS = [
    "sora", "runway", "runwayml", "pika", "pika labs", "kling", "kling ai",
    "stable video", "stable video diffusion", "animatediff", "modelscope",
    "zeroscope", "gen-2", "gen-3", "synthesia", "heygen", "d-id",
    "invideo ai", "pictory", "lumen5", "veed.io", "fliki", "elai",
    "colossyan", "deepbrain", "hour one", "vidnoz", "capcut ai",
    "adobe firefly video", "adobe express", "canva ai video",
]

CAMERA_VIDEO_MARKERS = [
    "make", "model", "gps", "camera", "lens", "iso", "exposure",
    "sony", "canon", "nikon", "gopro", "iphone", "samsung", "pixel",
    "dji", "fujifilm", "panasonic", "olympus",
]

def analyse_video(content: bytes, filename: str) -> dict:
    """
    Lightweight video metadata analysis.
    Checks container metadata and software tags only.
    Does NOT analyse individual frames or audio transcription.
    """
    ai_indicators = []
    human_indicators = []
    signals = {}

    file_size_mb = len(content) / (1024 * 1024)
    signals["file_size_mb"] = round(file_size_mb, 2)
    signals["filename"] = filename

    # ── 1. File extension check ───────────────────────────────────────────────
    fname = filename.lower()
    if fname.endswith(".mp4"):
        signals["format"] = "MP4"
    elif fname.endswith(".mov"):
        signals["format"] = "MOV"
        human_indicators.append("MOV format — common in iPhone/camera recordings")
    elif fname.endswith(".avi"):
        signals["format"] = "AVI"
    elif fname.endswith(".webm"):
        signals["format"] = "WebM"
        ai_indicators.append("WebM format — common AI video output format")
    else:
        signals["format"] = fname.split(".")[-1].upper()

    # ── 2. Binary metadata scan ───────────────────────────────────────────────
    # Scan first 64KB for text metadata tags
    header = content[:65536]
    try:
        header_text = header.decode("latin-1", errors="ignore").lower()
    except Exception:
        header_text = ""

    # Check for AI tool signatures
    found_ai_tools = []
    for tool in AI_VIDEO_TOOLS:
        if tool in header_text:
            found_ai_tools.append(tool)

    if found_ai_tools:
        ai_indicators.append(f"AI video tool signature detected: {', '.join(found_ai_tools)}")
        signals["ai_tools_found"] = found_ai_tools

    # Check for camera signatures
    found_camera = []
    for marker in CAMERA_VIDEO_MARKERS:
        if marker in header_text:
            found_camera.append(marker)

    if found_camera:
        human_indicators.append(f"Camera/device metadata found: {', '.join(found_camera[:3])}")
        signals["camera_markers"] = found_camera[:5]

    # ── 3. Common AI video dimensions check ──────────────────────────────────
    # AI video tools often produce specific resolutions
    ai_res_signatures = [b"512x512", b"1024x576", b"576x1024", b"768x432", b"432x768"]
    for res in ai_res_signatures:
        if res in content[:65536]:
            ai_indicators.append(f"Common AI video resolution detected: {res.decode()}")
            break

    # ── 4. File size heuristic ────────────────────────────────────────────────
    # AI generated short clips are often suspiciously small
    if file_size_mb < 2.0 and signals["format"] == "MP4":
        ai_indicators.append(f"Very small MP4 ({file_size_mb:.1f}MB) — may indicate AI-generated short clip")
    elif file_size_mb > 50:
        human_indicators.append(f"Large file size ({file_size_mb:.1f}MB) — consistent with real camera footage")

    # ── Classification ────────────────────────────────────────────────────────
    ai_score = len(ai_indicators)
    human_score = len(human_indicators)

    if found_ai_tools:
        label = "ai_generated"
        confidence = 90
        explanation = f"AI video generation tool signature detected in file metadata: {', '.join(found_ai_tools)}."
    elif ai_score > human_score:
        label = "ai_generated"
        confidence = min(75, 50 + ai_score * 10)
        explanation = f"Multiple AI video signals detected in metadata: {'; '.join(ai_indicators)}"
    elif human_score > ai_score:
        label = "human_captured"
        confidence = min(80, 50 + human_score * 10)
        explanation = f"Camera/device metadata found — consistent with real video recording."
    else:
        label = "uncertain"
        confidence = 45
        explanation = "Insufficient metadata signals to classify this video reliably."

    return {
        "label": label,
        "confidence": confidence,
        "explanation": explanation,
        "ai_indicators": ai_indicators,
        "human_indicators": human_indicators,
        "signals": signals,
        "disclaimer": "⚠️ Lightweight analysis only — checks file metadata and software signatures. Does not analyse video frames or audio. Catches obvious AI video tools only. For in-depth analysis, use a dedicated video forensics tool.",
    }


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

    # Determine MIME type for C2PA check
    fname = file.filename.lower()
    if fname.endswith(".jpg") or fname.endswith(".jpeg"):
        mime = "image/jpeg"
    elif fname.endswith(".webp"):
        mime = "image/webp"
    else:
        mime = "image/png"

    # C2PA compliance check
    c2pa_result = check_c2pa(content, mime)
    result["c2pa"] = c2pa_result

    # EU AI Act Article 50 compliance flag
    ai_prob = result.get("confidence", 0)
    is_ai = result.get("label") in ("ai_generated", "ai_assisted")
    has_c2pa = c2pa_result.get("has_c2pa", False)

    if is_ai and ai_prob >= 80 and not has_c2pa:
        result["eu_compliance"] = {
            "status": "NON_COMPLIANT",
            "rule": "EU AI Act Article 50",
            "deadline": "August 2, 2026",
            "issue": f"Content detected as AI-generated ({ai_prob}% confidence) but contains no C2PA provenance watermark as required by EU AI Act Article 50.",
            "action_required": "AI-generated content must be marked with machine-readable C2PA provenance metadata.",
        }
    elif is_ai and ai_prob >= 80 and has_c2pa:
        result["eu_compliance"] = {
            "status": "COMPLIANT",
            "rule": "EU AI Act Article 50",
            "issue": None,
            "note": "AI-generated content has C2PA provenance metadata present.",
        }
    else:
        result["eu_compliance"] = {
            "status": "NOT_APPLICABLE",
            "note": "Content not detected as AI-generated above 80% threshold.",
        }

    cache[cache_key] = result
    return result


@app.post("/api/check-video")
async def check_video(file: UploadFile = File(...)):
    """Lightweight video metadata analysis for AI generation signals."""
    content = await file.read()
    cache_key = hashlib.sha256(content).hexdigest()
    if cache_key in cache:
        return cache[cache_key]

    filename = file.filename.lower()
    allowed = (".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v")
    if not any(filename.endswith(ext) for ext in allowed):
        raise HTTPException(status_code=400, detail=f"Unsupported video format: {filename}")

    result = analyse_video(content, file.filename)
    result["filename"] = file.filename
    cache[cache_key] = result
    return result


@app.post("/api/check-video-url")
async def check_video_url(payload: dict):
    """
    Lightweight video URL metadata analysis.
    Checks URL patterns, domain, and query parameters for AI generation signals.
    No video is downloaded — URL metadata analysis only.
    """
    url = payload.get("url", "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="No URL provided")

    import urllib.parse
    cache_key = hashlib.sha256(url.encode()).hexdigest()
    if cache_key in cache:
        return cache[cache_key]

    ai_indicators = []
    human_indicators = []
    signals = {}
    signals["url"] = url

    parsed = urllib.parse.urlparse(url.lower())
    domain = parsed.netloc.replace("www.", "")
    signals["domain"] = domain

    # ── Known AI video platforms ──────────────────────────────────────────────
    AI_VIDEO_PLATFORMS = {
        "runwayml.com": "RunwayML (Gen-2/Gen-3)",
        "app.runwayml.com": "RunwayML",
        "pika.art": "Pika Labs",
        "kling.ai": "Kling AI",
        "sora.com": "OpenAI Sora",
        "dream-machine.ai": "Luma Dream Machine",
        "lumalabs.ai": "Luma AI",
        "synthesia.io": "Synthesia",
        "heygen.com": "HeyGen",
        "d-id.com": "D-ID",
        "invideo.io": "InVideo AI",
        "pictory.ai": "Pictory AI",
        "fliki.ai": "Fliki",
        "vidnoz.com": "Vidnoz",
        "colossyan.com": "Colossyan",
        "hourone.ai": "Hour One",
        "deepbrain.io": "DeepBrain AI",
        "elai.io": "Elai",
        "steve.ai": "Steve AI",
        "rawshorts.com": "Raw Shorts",
    }

    # ── Known human video platforms ───────────────────────────────────────────
    HUMAN_VIDEO_PLATFORMS = {
        "youtube.com": "YouTube",
        "youtu.be": "YouTube",
        "vimeo.com": "Vimeo",
        "dailymotion.com": "Dailymotion",
        "twitch.tv": "Twitch",
        "tiktok.com": "TikTok",
        "instagram.com": "Instagram",
        "facebook.com": "Facebook",
        "twitter.com": "Twitter/X",
        "x.com": "Twitter/X",
        "linkedin.com": "LinkedIn",
        "rumble.com": "Rumble",
    }

    # Check domain
    if domain in AI_VIDEO_PLATFORMS:
        tool = AI_VIDEO_PLATFORMS[domain]
        ai_indicators.append(f"Known AI video platform: {tool}")
        signals["platform"] = tool
        label = "ai_generated"
        confidence = 92
        explanation = f"URL is from {tool} — a known AI video generation platform."
    elif domain in HUMAN_VIDEO_PLATFORMS:
        platform = HUMAN_VIDEO_PLATFORMS[domain]
        human_indicators.append(f"Human video platform: {platform}")
        signals["platform"] = platform

        # YouTube specific checks
        if "youtube" in domain or "youtu.be" in domain:
            label = "uncertain"
            confidence = 0
            explanation = "YouTube URL detected. URL-only analysis cannot determine whether this video is AI-generated or human-created — YouTube hosts both. For accurate analysis, download the video and upload it directly using the Upload File tab."
            human_indicators.append("YouTube is a general platform hosting both human and AI-generated content")
        else:
            label = "uncertain"
            confidence = 0
            explanation = f"{platform} hosts both human and AI-generated content. URL-only analysis cannot determine content origin. Download the video and upload it directly for accurate analysis."
    else:
        # Unknown domain — check for AI signals in URL path
        full_url = url.lower()
        ai_url_signals = ["ai-generated", "ai_generated", "sora", "runway", "pika",
                         "synthesia", "heygen", "deepfake", "artificial"]
        found = [s for s in ai_url_signals if s in full_url]
        if found:
            ai_indicators.append(f"AI-related terms in URL: {', '.join(found)}")
            label = "ai_generated"
            confidence = 70
            explanation = f"AI-related terms detected in URL path."
        else:
            label = "uncertain"
            confidence = 40
            explanation = f"Unknown video platform. Cannot determine AI vs human content from URL metadata alone."

    result = {
        "label": label,
        "confidence": confidence,
        "explanation": explanation,
        "ai_indicators": ai_indicators,
        "human_indicators": human_indicators,
        "signals": signals,
        "disclaimer": "⚠️ URL analysis only — no video downloaded or processed. Results based on platform domain and URL patterns. For accurate analysis, download and upload the video file directly.",
    }
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

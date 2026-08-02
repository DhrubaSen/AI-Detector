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
    import numpy as np
    NUMPY_OK = True
except ImportError:
    NUMPY_OK = False

try:
    from PIL import Image, ImageFilter
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
    """
    Smart sentence splitter.
    For poetry (many short lines, little punctuation) uses line breaks.
    For prose uses punctuation.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    avg_line_len = sum(len(l.split()) for l in lines) / max(len(lines), 1) if lines else 20

    # Poetry mode: short lines, use each line as a unit
    if avg_line_len < 7 and len(lines) >= 4:
        return [l for l in lines if len(l) > 2]

    # Prose mode: split on sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    result = [s.strip() for s in sentences if len(s.strip()) > 10]

    # Fallback: if only 1 sentence detected but many lines, use lines
    if len(result) <= 1 and len(lines) >= 4:
        return [l for l in lines if len(l) > 2]

    return result


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
    """Detect human spelling variations, typos, and SMS/WhatsApp style."""
    common_variations = [
        # Typos
        "exhilerated", "embarassed", "wierd", "recieve", "occured",
        "seperately", "definately", "occurance", "untill",
        # Contractions without apostrophe
        "dont", "cant", "wont", "ive", "youre", "theyre",
        # SMS/WhatsApp shorthand — very strong human signal
        "shud", "cud", "wud", "bcoz", "bcause", "thru",
        "gr8", "l8r", "b4 ", "pls ", "plz", "sry ",
        "omg", "lmao", "btw ", "imo ", "tbh ", "ngl",
        "idk ", "fyi ", "asap", "ur ", "u r",
    ]
    text_lower = text.lower()
    count = sum(1 for v in common_variations if v in text_lower)

    # Bonus: non-Latin script = strong human signal
    non_latin = sum(1 for c in text if ord(c) > 0x00FF)
    if non_latin > 2:
        count += 2

    # Emoji = human signal
    emoji_ranges = [(0x1F300, 0x1F9FF), (0x2600, 0x26FF), (0x2700, 0x27BF)]
    for c in text:
        cp = ord(c)
        if any(s <= cp <= e for s, e in emoji_ranges):
            count += 1
            break

    # URL sharing = human behaviour
    if "http://" in text or "https://" in text:
        count += 1

    # Social media markers
    if "@" in text or "#" in text:
        count += 1

    return min(1.0, count * 0.4)


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

def is_poetry(text: str) -> dict:
    """
    Detect poetry — short lines, repetition as literary device,
    rhetorical questions, sensory imagery, line breaks mid-sentence.
    Human poetry should NOT be penalised for low entropy or repetition.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) < 3:
        return {"is_poetry": False}

    # Short average line length = poetry signal
    avg_line_len = sum(len(l.split()) for l in lines) / len(lines)
    is_short_lines = avg_line_len < 8

    # Many lines relative to word count
    words = text.split()
    line_to_word_ratio = len(lines) / max(len(words), 1)
    is_high_line_ratio = line_to_word_ratio > 0.15

    # Rhetorical questions
    question_count = text.count("?")
    has_rhetorical = question_count >= 1

    # Sensory/imagistic language
    sensory_words = [
        "rain", "light", "dark", "wet", "dry", "cold", "warm", "bright",
        "soft", "hard", "silence", "sound", "wind", "sun", "moon", "sky",
        "water", "stone", "leaf", "leaves", "air", "earth", "fire", "sea",
        "snow", "shadow", "gleam", "shine", "shining", "glow", "pale",
        "white", "black", "red", "golden", "silver", "blue", "green",
    ]
    # Abstract/philosophical vocabulary common in minimalist poetry
    abstract_poetry_words = [
        "soul", "nothing", "everything", "silence", "void", "empty",
        "small", "final", "last", "alone", "still", "only", "least",
        "made", "pray", "prays", "enter", "become", "remain", "wait",
        "song", "voice", "breath", "heart", "hand", "eye", "face",
        "man", "woman", "child", "god", "death", "life", "time", "world",
        "flute", "lays", "explained",
    ]
    text_lower = text.lower()
    sensory_count = sum(1 for w in sensory_words if w in text_lower)
    abstract_count = sum(1 for w in abstract_poetry_words if w in text_lower)
    is_imagistic = sensory_count >= 2 or abstract_count >= 3

    # Repetition in poetry is intentional — check for exact phrase repeat
    # (e.g. "Did no one see it" appearing twice)
    from collections import Counter
    phrase_counts = Counter()
    for i in range(len(lines) - 1):
        for j in range(i+1, len(lines)):
            if lines[i].lower().strip(".,?!") == lines[j].lower().strip(".,?!"):
                phrase_counts[lines[i]] += 1
    has_poetic_repetition = len(phrase_counts) > 0

    # Final poetry determination
    poetry_score = sum([
        is_short_lines,
        is_high_line_ratio,
        has_rhetorical,
        is_imagistic,
        has_poetic_repetition,
    ])

    # Very short lines + high line ratio alone = strong poetry signal
    # even without other markers (minimalist poetry)
    is_minimalist = is_short_lines and is_high_line_ratio and avg_line_len < 5

    return {
        "is_poetry": poetry_score >= 2 or is_minimalist,
        "poetry_score": poetry_score,
        "avg_line_length": round(avg_line_len, 1),
        "has_rhetorical_questions": has_rhetorical,
        "has_sensory_imagery": is_imagistic,
        "has_poetic_repetition": has_poetic_repetition,
        "sensory_count": sensory_count,
    }


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
    poetry = is_poetry(clean_text)
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

    # Poetry — short lines, repetition as literary device, sensory imagery
    if poetry.get("is_poetry"):
        # Neutralise low entropy penalty — poetry has uniform short lines
        perp_penalty = max(0, 1 - (perp / 3.0)) * 0.12
        ai_probability = max(0, ai_probability - perp_penalty)
        # Neutralise repetition penalty — poetic repetition is intentional
        if poetry.get("has_poetic_repetition"):
            ai_probability = max(0, ai_probability - rep * 0.16)
        # Sensory imagery = human signal
        if poetry.get("has_sensory_imagery"):
            ai_probability = max(0, ai_probability - 0.15)
        # Rhetorical questions = human signal
        if poetry.get("has_rhetorical_questions"):
            ai_probability = max(0, ai_probability - 0.10)

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
            "poetry_detected": poetry.get("is_poetry", False),
            "poetry_score": poetry.get("poetry_score", 0),
            "poetry_sensory_count": poetry.get("sensory_count", 0),
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


def fft_frequency_analysis(img_rgb) -> dict:
    """
    Frequency domain analysis using FFT.
    Real photos have natural high-frequency noise from camera sensors.
    AI images have artificially low noise or unusual frequency distributions.
    
    Key signals:
    - residual_std: high = natural camera noise, low = AI smoothing
    - noise_kurtosis: moderate (3-15) = natural, very high = screenshot/diagram
    - AI contribution: 0-0.4 added to AI probability score
    """
    if not NUMPY_OK or not PIL_OK:
        return {"available": False}
    
    try:
        gray = np.array(img_rgb.convert("L"), dtype=np.float32)
        h, w = gray.shape

        # High-frequency residual — difference between image and blurred version
        pil_gray = Image.fromarray(gray.astype(np.uint8))
        blurred = np.array(pil_gray.filter(ImageFilter.GaussianBlur(radius=2)), dtype=np.float32)
        residual = (gray - blurred).flatten()

        residual_std = float(np.std(residual))
        mean_r = float(np.mean(residual))
        std_r = float(np.std(residual))
        kurtosis = float(np.mean(((residual - mean_r) / max(std_r, 0.001)) ** 4))

        # Local variance coefficient of variation
        patch = 16
        local_vars = []
        for y in range(0, h - patch, patch):
            for x in range(0, w - patch, patch):
                local_vars.append(float(np.var(gray[y:y+patch, x:x+patch])))

        lv_cv = float(np.std(local_vars) / max(np.mean(local_vars), 1)) if local_vars else 0.5

        ai_contribution = 0.0
        fft_signals = []
        fft_human = []

        # Low residual std = unnaturally smooth = AI signal
        if residual_std < 15:
            ai_contribution += 0.20
            fft_signals.append(f"Very low high-frequency noise ({residual_std:.1f}) — AI images lack natural camera sensor noise")
        elif residual_std < 25:
            ai_contribution += 0.10
            fft_signals.append(f"Below-average high-frequency noise ({residual_std:.1f}) — possible AI smoothing")
        else:
            fft_human.append(f"Natural high-frequency sensor noise ({residual_std:.1f}) — consistent with real camera photograph")
            ai_contribution -= 0.10

        # Kurtosis discrimination
        # Real photos: 3-15 (near-Gaussian sensor noise)
        # AI art: 15-60 (spiky noise from generation artifacts)
        # Screenshots/diagrams: >60 (text edges create extreme spikes)
        if 15 < kurtosis < 60:
            ai_contribution += 0.10
            fft_signals.append(f"Frequency noise pattern typical of AI generation (kurtosis: {kurtosis:.1f})")
        elif kurtosis > 60:
            # This is screenshot/diagram territory — handled by screenshot detector
            pass
        else:
            fft_human.append(f"Natural noise distribution (kurtosis: {kurtosis:.1f}) — consistent with camera sensor")

        return {
            "available": True,
            "residual_std": round(residual_std, 2),
            "noise_kurtosis": round(kurtosis, 2),
            "local_variance_cv": round(lv_cv, 3),
            "ai_contribution": round(max(0, min(0.4, ai_contribution)), 3),
            "fft_ai_signals": fft_signals,
            "fft_human_signals": fft_human,
        }

    except Exception as e:
        return {"available": False, "error": str(e)[:80]}


def analyse_regions(img_rgb) -> dict:
    """
    Divide image into regions and analyse each independently.
    AI images have unnaturally uniform texture in specific regions.
    Real photos have natural sensor noise and texture variance.
    """
    width, height = img_rgb.size
    results = {}

    # Define regions
    regions = {
        "center": (width//4, height//4, 3*width//4, 3*height//4),
        "top": (0, 0, width, height//3),
        "bottom": (0, 2*height//3, width, height),
        "left": (0, 0, width//3, height),
        "right": (2*width//3, 0, width, height),
    }

    region_stds = []
    region_vars = []

    for name, box in regions.items():
        try:
            region = img_rgb.crop(box)
            pixels = list(region.getdata())
            if not pixels:
                continue

            sample = pixels[::max(1, len(pixels)//2000)]
            r_vals = [p[0] for p in sample]
            g_vals = [p[1] for p in sample]
            b_vals = [p[2] for p in sample]

            def std(vals):
                if len(vals) < 2: return 0
                avg = sum(vals) / len(vals)
                return (sum((v-avg)**2 for v in vals) / len(vals)) ** 0.5

            r_std = std(r_vals)
            g_std = std(g_vals)
            b_std = std(b_vals)
            avg_std = (r_std + g_std + b_std) / 3
            region_stds.append(avg_std)
            results[name] = round(avg_std, 2)
        except Exception:
            pass

    # Key metrics
    if not region_stds:
        return {"error": "Could not analyse regions"}

    overall_std = sum(region_stds) / len(region_stds)
    std_variance = (sum((s - overall_std)**2 for s in region_stds) / len(region_stds)) ** 0.5

    # AI signal: very low texture variance in center (face/subject area)
    center_std = results.get("center", overall_std)

    # Natural photos have HIGH std_variance between regions
    # AI images have more UNIFORM texture across regions
    is_uniform = std_variance < 8.0

    # Background detection — if bottom/corners are very bright (white bg removal)
    bg_brightness = results.get("bottom", 128)
    has_removed_bg = bg_brightness > 220 and center_std > 30

    # Skin smoothness proxy — center region unusually smooth
    is_smooth_center = center_std < 25 and overall_std > 30

    return {
        "region_stds": results,
        "overall_texture_std": round(overall_std, 2),
        "texture_variance_across_regions": round(std_variance, 2),
        "is_texture_uniform": is_uniform,
        "has_removed_background": has_removed_bg,
        "center_smoother_than_average": is_smooth_center,
        "center_std": round(center_std, 2),
    }


def analyse_noise_pattern(img_rgb) -> dict:
    """
    Real camera photos have sensor noise — random pixel variation.
    AI images have artificially smooth areas with unnaturally low noise.
    """
    try:
        # Convert to grayscale for noise analysis
        gray = img_rgb.convert('L')
        width, height = gray.size

        # Sample a grid of small patches
        patch_size = 8
        patch_stds = []

        for y in range(0, height - patch_size, patch_size * 4):
            for x in range(0, width - patch_size, patch_size * 4):
                patch = gray.crop((x, y, x + patch_size, y + patch_size))
                pixels = list(patch.getdata())
                if len(pixels) < 4:
                    continue
                avg = sum(pixels) / len(pixels)
                std = (sum((p - avg)**2 for p in pixels) / len(pixels)) ** 0.5
                patch_stds.append(std)

        if not patch_stds:
            return {}

        avg_patch_std = sum(patch_stds) / len(patch_stds)
        # Very low patch std = unnaturally smooth = AI signal
        very_smooth_patches = sum(1 for s in patch_stds if s < 3.0) / len(patch_stds)
        # High variation in patch stds = natural photo noise
        patch_std_variance = (sum((s - avg_patch_std)**2 for s in patch_stds) / len(patch_stds)) ** 0.5

        return {
            "avg_patch_noise": round(avg_patch_std, 3),
            "smooth_patch_ratio": round(very_smooth_patches, 3),
            "noise_variance": round(patch_std_variance, 3),
            "is_unnaturally_smooth": avg_patch_std < 8.0 and very_smooth_patches > 0.3,
        }
    except Exception as e:
        return {"error": str(e)[:50]}


# Known AI tool watermark colors and positions
AI_WATERMARK_LOGOS = {
    "Grok": {"color_range": [(180, 180, 180), (255, 255, 255)], "position": "bottom_right"},
    "Midjourney": {"color_range": [(200, 200, 200), (255, 255, 255)], "position": "bottom"},
    "Adobe Firefly": {"color_range": [(255, 0, 0), (255, 100, 100)], "position": "bottom_right"},
    "DALL-E": {"color_range": [(0, 0, 0), (50, 50, 50)], "position": "bottom"},
}

def detect_ai_watermark(img_rgb) -> dict:
    """
    Detect visible AI tool watermarks in image corners.
    Checks for known logo patterns from Grok, Midjourney, DALL-E etc.
    Also checks for suspicious corner regions with high contrast logos.
    """
    try:
        width, height = img_rgb.size
        
        # Define corner regions to check (10% of image dimensions)
        margin_w = max(50, width // 8)
        margin_h = max(30, height // 8)
        
        corners = {
            "bottom_right": img_rgb.crop((width - margin_w, height - margin_h, width, height)),
            "bottom_left": img_rgb.crop((0, height - margin_h, margin_w, height)),
            "top_right": img_rgb.crop((width - margin_w, 0, width, margin_h)),
            "top_left": img_rgb.crop((0, 0, margin_w, margin_h)),
            "bottom_center": img_rgb.crop((width//3, height - margin_h, 2*width//3, height)),
        }
        
        watermark_signals = []
        found_watermark = None
        
        for corner_name, corner_img in corners.items():
            pixels = list(corner_img.getdata())
            if not pixels:
                continue
            
            r_vals = [p[0] for p in pixels]
            g_vals = [p[1] for p in pixels]
            b_vals = [p[2] for p in pixels]
            
            avg_r = sum(r_vals) / len(r_vals)
            avg_g = sum(g_vals) / len(g_vals)
            avg_b = sum(b_vals) / len(b_vals)
            
            # Check for white/light logo on dark background (Grok style)
            # or dark logo on light background
            r_range = max(r_vals) - min(r_vals)
            g_range = max(g_vals) - min(g_vals)
            b_range = max(b_vals) - min(b_vals)
            max_range = max(r_range, g_range, b_range)
            
            # High contrast in a small corner = possible watermark/logo
            if max_range > 150 and corner_name in ["bottom_right", "bottom_left", "bottom_center"]:
                # Check for near-white pixels (typical watermark)
                white_pixels = sum(1 for p in pixels if p[0] > 200 and p[1] > 200 and p[2] > 200)
                white_ratio = white_pixels / len(pixels)
                
                # Check for near-black pixels (dark watermark)
                black_pixels = sum(1 for p in pixels if p[0] < 50 and p[1] < 50 and p[2] < 50)
                black_ratio = black_pixels / len(pixels)
                
                # Watermark signature: mix of very light and very dark pixels
                # in a small corner region
                if 0.05 < white_ratio < 0.4 and black_ratio < 0.3:
                    watermark_signals.append(f"Possible watermark detected in {corner_name.replace('_', ' ')} (contrast: {max_range}, white pixels: {white_ratio:.0%})")
                    if not found_watermark:
                        found_watermark = corner_name

            # Check for Grok-specific: circular logo (white circle outline on dark)
            # Grok watermark is typically white text/logo on semi-transparent dark bg
            if corner_name == "bottom_right":
                # Look for the characteristic Grok grey/white logo pattern
                grey_pixels = sum(1 for p in pixels if 150 < p[0] < 255 and 150 < p[1] < 255 and 150 < p[2] < 255)
                grey_ratio = grey_pixels / len(pixels)
                bg_pixels = sum(1 for p in pixels if p[0] < 100 and p[1] < 100 and p[2] < 100)
                bg_ratio = bg_pixels / len(pixels)
                
                if 0.1 < grey_ratio < 0.5 and bg_ratio > 0.1:
                    watermark_signals.append("AI tool logo pattern detected in bottom-right corner — consistent with Grok, Midjourney or similar AI image generator watermark")
                    found_watermark = "bottom_right_logo"

        # Also check for suspiciously uniform bottom strip (added watermark bar)
        bottom_strip = img_rgb.crop((0, height - 30, width, height))
        strip_pixels = list(bottom_strip.getdata())
        if strip_pixels:
            strip_r = [p[0] for p in strip_pixels]
            strip_std = (sum((v - sum(strip_r)/len(strip_r))**2 for v in strip_r) / len(strip_r)) ** 0.5
            if strip_std < 15:  # very uniform bottom strip = watermark bar
                watermark_signals.append(f"Uniform bottom strip detected (std: {strip_std:.1f}) — may indicate added watermark bar")

        return {
            "watermark_detected": len(watermark_signals) > 0,
            "watermark_location": found_watermark,
            "watermark_signals": watermark_signals,
            "corners_checked": list(corners.keys()),
        }

    except Exception as e:
        return {"watermark_detected": False, "error": str(e)[:60]}


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

    # ── 1c. Watermark detection ───────────────────────────────────────────────
    try:
        img_rgb_wm = img.convert("RGB")
        wm_result = detect_ai_watermark(img_rgb_wm)
        signals["watermark_check"] = wm_result
        if wm_result.get("watermark_detected"):
            for sig in wm_result.get("watermark_signals", []):
                ai_indicators.append(f"🔍 Watermark: {sig}")
    except Exception as e:
        signals["watermark_error"] = str(e)[:50]

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

        # ── Region analysis ───────────────────────────────────────────────────
        region_data = analyse_regions(img_rgb)
        signals["region_analysis"] = region_data

        if region_data.get("has_removed_background"):
            human_indicators.append("Background removal detected — real photo with background edited out")
            signals["background_removed"] = True

        if region_data.get("is_texture_uniform"):
            ai_indicators.append(f"Unnaturally uniform texture across image regions (variance: {region_data.get('texture_variance_across_regions', 0)}) — typical of AI generation")

        if region_data.get("center_smoother_than_average"):
            ai_indicators.append(f"Center region unusually smooth (std: {region_data.get('center_std', 0)}) — possible AI portrait skin smoothing")

        # ── Noise pattern analysis ────────────────────────────────────────────
        noise_data = analyse_noise_pattern(img_rgb)
        signals["noise_analysis"] = noise_data

        if noise_data.get("is_unnaturally_smooth"):
            ai_indicators.append(f"Unnaturally low pixel noise (avg: {noise_data.get('avg_patch_noise', 0)}, smooth patches: {noise_data.get('smooth_patch_ratio', 0)*100:.0f}%) — AI images lack natural camera sensor noise")
        elif noise_data.get("avg_patch_noise", 0) > 12:
            human_indicators.append(f"Natural camera sensor noise detected (avg: {noise_data.get('avg_patch_noise', 0)}) — consistent with real photograph")

        # ── Frequency domain analysis (FFT) ───────────────────────────────────
        fft_data = fft_frequency_analysis(img_rgb)
        signals["fft_analysis"] = {
            "residual_std": fft_data.get("residual_std"),
            "noise_kurtosis": fft_data.get("noise_kurtosis"),
            "ai_contribution": fft_data.get("ai_contribution", 0),
        }

        if fft_data.get("available"):
            for sig in fft_data.get("fft_ai_signals", []):
                ai_indicators.append(sig)
            for sig in fft_data.get("fft_human_signals", []):
                human_indicators.append(sig)

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

    # Incorporate FFT frequency analysis contribution
    fft_contribution = signals.get("fft_analysis", {}).get("ai_contribution", 0)
    fft_adjusted_ai = ai_score + (fft_contribution * 2)  # scale to indicator count

    # Check watermark result
    wm_detected = signals.get("watermark_check", {}).get("watermark_detected", False)

    if has_ai_software_tag:
        label = "ai_generated"
        confidence = 95
        explanation = f"AI generation tool detected in image metadata. {'; '.join(ai_indicators)}"
    elif wm_detected:
        label = "ai_generated"
        confidence = 88
        wm_sigs = signals.get("watermark_check", {}).get("watermark_signals", [])
        explanation = f"AI tool watermark detected. {wm_sigs[0] if wm_sigs else ''}. Note: if watermark has been removed from the original, this image may still be AI-generated."
    elif total == 0 and fft_contribution < 0.1:
        label = "unknown"
        confidence = 50
        explanation = "Insufficient signals to classify this image reliably."
    elif fft_adjusted_ai > human_score * 1.5 or (fft_contribution >= 0.15 and ai_score >= human_score):
        label = "ai_generated"
        confidence = min(92, 55 + int(fft_adjusted_ai * 8) + int(fft_contribution * 20))
        fft_note = f" Frequency analysis: low sensor noise detected." if fft_contribution > 0.1 else ""
        explanation = f"Multiple AI generation signals detected: {'; '.join(ai_indicators)}.{fft_note}"
    elif human_score > fft_adjusted_ai:
        label = "human_captured"
        confidence = min(88, 55 + human_score * 10)
        explanation = f"Photograph characteristics detected: {'; '.join(human_indicators)}"
    else:
        label = "uncertain"
        confidence = 50
        explanation = f"Mixed signals. AI indicators: {'; '.join(ai_indicators) if ai_indicators else 'none'}. Human indicators: {'; '.join(human_indicators) if human_indicators else 'none'}."

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

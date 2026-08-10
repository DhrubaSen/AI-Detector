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
# Bounded FIFO cache — a plain dict here grows forever and was a memory leak.
from collections import OrderedDict
CACHE_MAX_ENTRIES = 500

class BoundedCache(OrderedDict):
    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > CACHE_MAX_ENTRIES:
            self.popitem(last=False)  # evict oldest

cache: dict = BoundedCache()

# ── Upload size limits ────────────────────────────────────────────────────────
# Addresses audit Finding 02 (HIGH): no server-side file size caps existed.
# Video's 100MB matches what the frontend UI already advertises to users.
MAX_TEXT_CHARS = 200_000       # ~30-40k words, generous beyond any real document
MAX_DOCUMENT_BYTES = 20 * 1024 * 1024   # 20MB
MAX_IMAGE_BYTES = 25 * 1024 * 1024      # 25MB
MAX_VIDEO_BYTES = 100 * 1024 * 1024     # 100MB — already advertised in the UI

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="AI Content Detector", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_security_headers(request, call_next):
    """
    Adds the 6 standard security headers flagged as missing in the
    2026-08 passive security audit (Finding 01, CRITICAL, +3 risk points).
    """
    response = await call_next(request)
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' https://gc.zgo.at; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "connect-src 'self' https://dhrubasen.goatcounter.com; "
        "frame-ancestors 'none'"
    )
    response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = (
        "camera=(), microphone=(), geolocation=(), interest-cohort=()"
    )
    return response

# ─────────────────────────────────────────────────────────────────────────────
# TEXT ANALYSIS ENGINE
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


# ── Human / Mixed / AI category mapping ──────────────────────────────────────
# Percent confidence reads as falsely precise for something inherently
# probabilistic. This collapses every fine-grained label used across text,
# image, and video analysis into one simple three-way category so the
# frontend can show "Human" / "Mixed" / "AI" instead of a raw number.
# All underlying signals, indicators, and explanations are unchanged —
# this is purely an additional field for display purposes.
CATEGORY_MAP = {
    "human_written": "Human",
    "human_captured": "Human",
    "human_created": "Human",
    "ai_generated": "AI",
    "ai_assisted": "Mixed",
    "uncertain": "Mixed",
    "unknown": "Mixed",
    "insufficient_text": "Mixed",
    "error": "Mixed",
}

def category_for_label(label: str) -> str:
    return CATEGORY_MAP.get(label, "Mixed")


# ── 5-tier lab-report-style banding ──────────────────────────────────────────
# Feedback on the 3-way version: collapsing everything to Human/Mixed/AI loses
# real information — a barely-over-the-line result and an overwhelming one
# both just say "AI". This adds finer resolution (closer to a lab report's
# low/normal/high bands) without reintroducing a raw percentage, and drops
# the word "Uncertain" entirely in favour of "Mixed" for the boundary zone.
#
# Deliberately reuses the *existing* label + confidence a result already
# computed, rather than a new scoring pass — label already encodes direction
# (leaning AI / leaning human / genuinely mixed) and confidence encodes
# strength within that direction for the two-sided branches (ai_generated,
# human_written/human_captured/human_created). The boundary-zone labels
# (ai_assisted, uncertain, unknown, insufficient_text, error) don't carry a
# meaningful directional magnitude in the current scoring (their confidence
# values are flat constants, not a curve), so they all land in "Mixed" as a
# single band rather than being split further.
AI_LABELS = {"ai_generated"}
HUMAN_LABELS = {"human_written", "human_captured", "human_created"}
TIER_SPLIT_CONFIDENCE = 85  # within a two-sided label, split "Likely X" vs "X"

def tier_for_result(label: str, confidence, lean: float = None) -> str:
    """
    lean: optional signed value already available at the call site —
    positive means the signals lean AI, negative means they lean human,
    None/0 means no directional evidence at all (don't fabricate a lean
    the data doesn't support). Only used for the boundary-zone labels;
    the two-sided labels (ai_generated / human_*) already carry their own
    direction via the label itself.
    """
    try:
        conf = float(confidence)
    except (TypeError, ValueError):
        conf = 0
    if label in AI_LABELS:
        return "AI" if conf >= TIER_SPLIT_CONFIDENCE else "Likely AI"
    if label in HUMAN_LABELS:
        return "Human" if conf >= TIER_SPLIT_CONFIDENCE else "Likely Human"
    if lean is not None and lean > 0:
        return "AI Generated, Human Edited"
    if lean is not None and lean < 0:
        return "Human Written, AI Edited"
    return "Mixed"


def tokenize_sentences(text: str) -> list[str]:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    avg_line_len = sum(len(l.split()) for l in lines) / max(len(lines), 1) if lines else 20
    if avg_line_len < 7 and len(lines) >= 4:
        return [l for l in lines if len(l) > 2]
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    result = [s.strip() for s in sentences if len(s.strip()) > 10]
    if len(result) <= 1 and len(lines) >= 4:
        return [l for l in lines if len(l) > 2]
    return result


def compute_perplexity_proxy(text: str) -> float:
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
    return variance


def sentence_length_variance(sentences: list[str]) -> float:
    if len(sentences) < 3:
        return 0.0
    lengths = [len(s.split()) for s in sentences]
    avg = sum(lengths) / len(lengths)
    variance = sum((l - avg) ** 2 for l in lengths) / len(lengths)
    return variance


def hedge_density(text: str) -> float:
    words = text.lower().split()
    if not words:
        return 0.0
    count = sum(1 for w in words if w.rstrip('.,;:') in HEDGE_WORDS)
    text_lower = text.lower()
    for phrase in ["it is worth noting", "it should be noted", "in conclusion",
                   "to summarize", "in summary", "plays a crucial role",
                   "plays an important role", "it is important", "it is essential"]:
        if phrase in text_lower:
            count += 1
    return count / max(len(words), 1)


def human_marker_density(text: str) -> float:
    words = text.lower().split()
    if not words:
        return 0.0
    count = sum(1 for w in words if w.rstrip('.,;:!?') in HUMAN_MARKERS)
    return count / max(len(words), 1)


def vocabulary_richness(text: str) -> float:
    words = re.findall(r'\b[a-z]+\b', text.lower())
    if len(words) < 10:
        return 1.0
    return len(set(words)) / len(words)


def repetition_score(sentences: list[str]) -> float:
    if len(sentences) < 4:
        return 0.0
    starts = [s.split()[0].lower() if s.split() else '' for s in sentences]
    unique_starts = len(set(starts))
    repetition = 1.0 - (unique_starts / len(starts))
    return repetition


SELF_DEPRECATING = [
    "i must confess", "i am embarrassed", "i am embarassed", "frankly",
    "i confess", "i must admit", "to be honest", "i am afraid",
    "forgive me", "i apologize", "i apologise", "i regret",
]

AUTOBIOGRAPHICAL = [
    "memory lane", "i recall", "i remember", "in those days",
    "when i was", "years ago", "at that time",
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
    text_lower = text.lower()
    count = sum(1 for phrase in PROFESSIONAL_FIRST_PERSON if phrase in text_lower)
    return min(1.0, count / 4)


def spelling_variation_score(text: str) -> float:
    # FIX: these were matched with plain substring search ("ive" in text),
    # which fires inside ordinary words — "archives", "massive", "given",
    # "lives", "drive" all contain "ive"; "thrum" contains "thru"; "single"
    # contains "ngl"; "our " contains "ur ". A single fiction paragraph
    # could max out this score without a single real typo or SMS-ism in it.
    # Now matched as whole words only.
    common_variations = [
        "exhilerated", "embarassed", "wierd", "recieve", "occured",
        "seperately", "definately", "occurance", "untill",
        "dont", "cant", "wont", "ive", "youre", "theyre",
        "shud", "cud", "wud", "bcoz", "bcause", "thru",
        "gr8", "l8r", "b4", "pls", "plz", "sry",
        "omg", "lmao", "btw", "imo", "tbh", "ngl",
        "idk", "fyi", "asap", "ur",
    ]
    text_lower = text.lower()
    words_in_text = set(re.findall(r"[a-z0-9']+", text_lower))
    count = sum(1 for v in common_variations if v in words_in_text)
    if re.search(r'\bu r\b', text_lower):
        count += 1
    # FIX: this was meant to catch genuine non-Latin script (a real human
    # signal — e.g. text mixing in Hindi/Chinese/Arabic). But `ord(c) >
    # 0x00FF` also catches ordinary typographic punctuation — em-dash,
    # en-dash, curly quotes, ellipsis — purely because their Unicode
    # codepoints happen to sit above 255, not because they indicate
    # anything about script or language. Worse, em-dash usage is widely
    # observed as a common AI writing tendency, so counting it as a human
    # signal was likely backwards. Typographic punctuation is now excluded
    # from this count entirely.
    TYPOGRAPHIC_PUNCTUATION = {
        "\u2014", "\u2013",  # em dash, en dash
        "\u2018", "\u2019",  # curly single quotes
        "\u201C", "\u201D",  # curly double quotes
        "\u2026",            # ellipsis
        "\u00A0",            # non-breaking space
    }
    non_latin = sum(1 for c in text if ord(c) > 0x00FF and c not in TYPOGRAPHIC_PUNCTUATION)
    if non_latin > 2:
        count += 2
    emoji_ranges = [(0x1F300, 0x1F9FF), (0x2600, 0x26FF), (0x2700, 0x27BF)]
    for c in text:
        cp = ord(c)
        if any(s <= cp <= e for s, e in emoji_ranges):
            count += 1
            break
    if "http://" in text or "https://" in text:
        count += 1
    if "@" in text or "#" in text:
        count += 1
    return min(1.0, count * 0.4)


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

AI_POETRY_CLICHES = [
    "anchored deeply", "anchored in", "in the heart",
    "sweetest moments", "winding roads", "changing weather",
    "side by side", "time apart", "spoken word",
    "spirit sings", "shadows fall", "every call",
    "need to hide", "need to fear", "need to cry",
    "walking closely", "walking together", "hand in hand",
    "silent space", "refuge in", "measured by",
    "joy a memory", "songs your spirit",
    "through the storm", "through the rain", "through it all",
    "beacon of light", "light in the dark", "shining through",
    "stronger together", "never alone", "always there",
    "heart and soul", "body and soul", "heart and mind",
    "ups and downs", "highs and lows", "ebb and flow",
    "laughter and tears", "joy and pain", "sun and rain",
    "chapter of life", "journey of life", "path of life",
    "unspoken words", "written in the stars", "meant to be",
]

def specificity_score(text: str) -> dict:
    words = text.lower().split()
    text_lower = text.lower()

    CONCRETE_MARKERS = [
        "forty", "hundred", "thousand", "dozen", "twice", "three", "four",
        "asphalt", "microwave", "refrigerator", "flute", "paving", "glass lenses",
        "waterlogged", "metal ring", "telescope", "briefcase", "notebook",
        "sails", "ropes", "helmsman", "mast", "anchor", "vessel", "hull",
        "whipped", "ripped", "drown", "sink", "frown", "confined", "inclined",
        "dust", "toss", "fro", "frightened", "fearsome",
        "polishing", "huffing and puffing", "blinked slowly", "patted his pockets",
        "stepping out", "washed clean", "wet and shining",
        "kitchen", "hallway", "lighthouse", "cliff", "mainland", "street",
        "paving stones", "leaves", "asphalt",
        "tuesday", "january", "morning tide", "forty years",
        "schwarzschild", "geodesic", "singularity", "metric tensor",
        "entropy", "bayesian", "eliza effect", "hallucination",
    ]

    GENERIC_MARKERS = [
        "journey", "path", "road", "bond", "connection", "soul", "spirit",
        "heart", "moments", "memories", "time", "love", "care", "strength",
        "light", "dark", "shadow", "friend", "together", "apart",
        "laughter", "tears", "joy", "pain", "hope", "fear",
        "silence", "voice", "word", "song", "call", "hear",
        "walk", "stand", "hold", "know", "feel", "see",
    ]

    concrete_count = sum(1 for m in CONCRETE_MARKERS if m in text_lower)
    generic_count = sum(1 for m in GENERIC_MARKERS if m in text_lower)

    total = max(concrete_count + generic_count, 1)
    generic_ratio = generic_count / total
    concrete_ratio = concrete_count / total

    is_overly_generic = (generic_ratio > 0.85 and concrete_count == 0 and
                          generic_count >= 6)

    return {
        "concrete_count": concrete_count,
        "generic_count": generic_count,
        "generic_ratio": round(generic_ratio, 3),
        "is_overly_generic": is_overly_generic,
    }


def cliche_density(text: str) -> dict:
    text_lower = text.lower()
    found = [c for c in AI_POETRY_CLICHES if c in text_lower]
    density = len(found) / max(len(text.split()) / 10, 1)
    return {
        "cliche_count": len(found),
        "cliche_density": round(density, 3),
        "found_cliches": found[:5],
        "is_high_cliche": len(found) >= 2,
    }


def is_poetry(text: str) -> dict:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) < 3:
        return {"is_poetry": False}

    avg_line_len = sum(len(l.split()) for l in lines) / len(lines)
    is_short_lines = avg_line_len < 8

    words = text.split()
    line_to_word_ratio = len(lines) / max(len(words), 1)
    is_high_line_ratio = line_to_word_ratio > 0.15

    question_count = text.count("?")
    has_rhetorical = question_count >= 1

    sensory_words = [
        "rain", "light", "dark", "wet", "dry", "cold", "warm", "bright",
        "soft", "hard", "silence", "sound", "wind", "sun", "moon", "sky",
        "water", "stone", "leaf", "leaves", "air", "earth", "fire", "sea",
        "snow", "shadow", "gleam", "shine", "shining", "glow", "pale",
        "white", "black", "red", "golden", "silver", "blue", "green",
        "waves", "wave", "storm", "sail", "sails", "ship", "sea", "ocean",
        "tide", "shore", "wind", "drown", "sink", "night", "dust", "frown",
        "rope", "ropes", "mast", "helm", "helmsman", "morning", "evening",
        "cloud", "thunder", "lightning", "fog", "mist", "frost", "dew",
        "breeze", "gale", "flood", "river", "lake", "mountain", "valley",
        "flower", "rose", "tree", "bird", "song", "cry", "tears", "blood",
    ]
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

    from collections import Counter
    phrase_counts = Counter()
    for i in range(len(lines) - 1):
        for j in range(i+1, len(lines)):
            if lines[i].lower().strip(".,?!") == lines[j].lower().strip(".,?!"):
                phrase_counts[lines[i]] += 1
    has_poetic_repetition = len(phrase_counts) > 0

    line_starts = [l.lower().split()[0] if l.split() else "" for l in lines]
    start_counts = Counter(line_starts)
    has_anaphora = any(c >= 2 for c in start_counts.values() if c > 0)

    end_words = [l.rstrip(".,?!:; ").split()[-1].lower() if l.split() else "" for l in lines]
    rhyme_pairs = 0
    for i in range(len(end_words) - 1):
        w1, w2 = end_words[i], end_words[i+1]
        if len(w1) >= 2 and len(w2) >= 2 and w1[-2:] == w2[-2:] and w1 != w2:
            rhyme_pairs += 1
    has_rhyme = rhyme_pairs >= 2

    poetry_score = sum([
        is_short_lines,
        is_high_line_ratio,
        has_rhetorical,
        is_imagistic,
        has_poetic_repetition,
        has_rhyme,
        has_anaphora,
    ])

    is_minimalist = is_short_lines and is_high_line_ratio and avg_line_len < 5

    # FIX: poetry_score >= 2 could previously be satisfied entirely by
    # thematic/coincidental signals — rhetorical questions in dialogue,
    # sensory-word density that any sufficiently long descriptive story
    # accumulates, two paragraphs that happen to end in similar-sounding
    # words, two paragraphs that happen to open with the same word — none
    # of which require the text to actually be formatted as short verse
    # lines. A multi-paragraph prose story (avg ~35 words/"line") could
    # trip is_poetry=True with zero real line-structure evidence, wrongly
    # routing it into the poetry human-bonus branch. Now requires at least
    # one genuine line-structure signal before qualifying.
    has_line_structure_evidence = is_short_lines or is_high_line_ratio

    return {
        "is_poetry": (has_line_structure_evidence and poetry_score >= 2) or is_minimalist,
        "poetry_score": poetry_score,
        "avg_line_length": round(avg_line_len, 1),
        "has_rhetorical_questions": has_rhetorical,
        "has_sensory_imagery": is_imagistic,
        "has_poetic_repetition": has_poetic_repetition,
        "has_rhyme": has_rhyme,
        "rhyme_pairs": rhyme_pairs,
        "has_anaphora": has_anaphora,
        "sensory_count": sensory_count,
    }


def creative_writing_signals(text: str) -> dict:
    words = text.lower().split()
    sentences = tokenize_sentences(text)
    text_lower = text.lower()
    word_count = len(words)

    if word_count < 50:
        return {"is_creative": False}

    symbol_count = sum(1 for w in words if w.rstrip('.,;:!?"') in SYMBOLIC_WORDS)
    symbol_density = symbol_count / max(word_count, 1)

    has_dialogue = any(marker in text for marker in DIALOGUE_MARKERS[:3])
    dialogue_word_count = sum(1 for w in DIALOGUE_MARKERS[3:] if w in text_lower)
    has_dialogue_verbs = dialogue_word_count > 0

    efficiency_count = sum(1 for phrase in NARRATIVE_EFFICIENCY if phrase in text_lower)
    efficiency_density = efficiency_count / max(len(sentences), 1)

    plot_words = ["arrived", "found", "saw", "heard", "felt", "realized",
                  "discovered", "decided", "turned", "looked", "ran", "left",
                  "returned", "died", "fell", "rose", "came", "went", "took"]
    plot_count = sum(1 for w in words if w.rstrip('.,;:!?') in plot_words)
    plot_density = plot_count / max(len(sentences), 1)

    irony_phrases = [
        "guided dozens", "guided innocent", "led to their",
        "let his fire", "let her fire", "abandoned his post",
        "too late", "in vain", "only to find", "only to discover",
        "at the cost of", "by saving", "while trying to save",
        "in the exact", "tightly in", "completely still",
        "staring at", "pure disappointment", "judged him", "judged her",
        "the whole time", "all along", "had been holding",
        "seemed to whisper", "felt like pure",
    ]
    has_irony = any(phrase in text_lower for phrase in irony_phrases)

    narrative_words = [
        "tower", "vessel", "shore", "cliff", "lighthouse",
        "messenger", "keeper", "sailor", "captain", "village",
        "apartment", "kitchen", "office", "phone", "keys", "coffee",
        "hallway", "sofa", "microwave", "refrigerator", "boss",
        "bedroom", "bathroom", "door", "window", "car", "street",
        "laptop", "email", "desk", "bag", "wallet", "cat", "dog",
        "arthur", "emma", "john", "mary", "thomas", "emily", "james",
        # Period/historical-fiction setting vocabulary — the list above is
        # almost entirely modern-domestic or nautical, so any archaic-style
        # fiction had no way to trigger is_creative at all, regardless of
        # content, blocking every creative-writing-gated signal.
        "estate", "servant", "horseback", "moorland", "manor", "inn",
        "tavern", "carriage", "physician", "kingdom", "parish", "squire",
        "duke", "lady", "governess", "butler", "footman", "parlour",
        "hearth", "candle", "stagecoach", "chamber", "gentleman",
        "gallery", "portrait", "cloak", "candlestick", "baronet",
    ]
    words_in_text = set(re.findall(r'\b[a-z]+\b', text_lower))
    is_narrative = (sum(1 for w in narrative_words if w in words_in_text) > 1 or
                    plot_density > 0.3)

    return {
        "is_creative": is_narrative,
        "symbol_density": round(symbol_density, 4),
        "has_dialogue": has_dialogue or has_dialogue_verbs,
        "efficiency_density": round(efficiency_density, 4),
        "plot_density": round(plot_density, 4),
        "has_irony_structure": has_irony,
    }


def purple_prose_signals(text: str) -> dict:
    """
    A distinct AI-fiction tell from irony/dialogue-absence: a neatly
    symmetrical, thematically-explicit closing beat. None of this is
    inherently bad writing on its own, and a human author can absolutely
    write this way — but the combination is a very recognizable, common
    pattern in AI-generated fiction endings specifically:
      - a simile in the closing lines linking something mechanical/
        environmental to an abstract virtue (heart, hope, pulse)
      - a triadic loss/but-still antithesis ("they had lost X, they had
        lost Y, but Z")
      - literal continuity phrasing ("still breathing/beating/alive")
      - the story's title echoed verbatim in the closing lines
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    final_chunk = " ".join(paragraphs[-2:]) if len(paragraphs) >= 2 else text
    final_chunk_lower = final_chunk.lower()

    simile_match = bool(re.search(
        r'\blike a\b[^.!?]{0,25}\b(heart|hope|whisper|prayer|promise|memory|dream|pulse)\b',
        final_chunk_lower))

    triadic_antithesis = bool(re.search(
        r'(they had lost|we had lost|i had lost|had lost the)[^.!?]{0,60}(had lost)[^.!?]{0,60}\bbut\b',
        final_chunk_lower))

    still_continuity = bool(re.search(
        r'\bstill (breathing|alive|beating|standing|burning|glowing|here)\b',
        final_chunk_lower))

    # Title echo: if the first line reads like a short title (few words,
    # no closing sentence punctuation), check whether a distinctive
    # two-word phrase from it reappears verbatim in the closing lines.
    title_echo = False
    if paragraphs:
        first_line = paragraphs[0]
        if len(first_line.split()) <= 6 and not first_line.rstrip().endswith((".", "!", "?")):
            title_words = re.findall(r'[a-z]+', first_line.lower())
            for i in range(len(title_words) - 1):
                bigram = title_words[i] + " " + title_words[i + 1]
                if bigram in final_chunk_lower:
                    title_echo = True
                    break

    hit_count = sum([simile_match, triadic_antithesis, still_continuity, title_echo])

    return {
        "has_closing_simile": simile_match,
        "has_triadic_antithesis": triadic_antithesis,
        "has_still_continuity_phrase": still_continuity,
        "has_title_echo": title_echo,
        "purple_prose_hits": hit_count,
        "is_thematic_closure": hit_count >= 2,
    }


def tense_shift_signals(text: str) -> dict:
    """
    A present-tense copula/verb slipping into otherwise past-tense
    narration outside dialogue — e.g. "Leo is ten years old and right
    now, hysterical" embedded in past-tense narration. Human writers can
    do this too (typos, deliberate present-tense asides), so this is a
    mild contributing signal, not decisive on its own. Only meaningful
    when the passage is clearly, predominantly past-tense narration —
    otherwise present tense is just the narrative's natural voice.
    """
    text_no_dialogue = re.sub(r'"[^"]*"', '', text)
    sentences = re.split(r'(?<=[.!?])\s+', text_no_dialogue.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    present_copula = re.compile(r'\b(?!It\b)(He|She|They|[A-Z][a-z]+) (is|are)\b')
    past_markers = re.compile(r'\b\w+ed\b|\b(was|were|had|said|looked|walked|felt|saw|heard)\b', re.IGNORECASE)

    past_tense_sentence_count = sum(1 for s in sentences if past_markers.search(s))
    is_past_tense_narration = past_tense_sentence_count >= max(3, len(sentences) * 0.5)

    flagged = []
    for s in sentences:
        if present_copula.search(s) and not past_markers.search(s):
            flagged.append(s[:80])

    return {
        "is_past_tense_narration": is_past_tense_narration,
        "tense_shift_hits": len(flagged),
        "flagged_sentences": flagged[:3],
        "has_tense_shift_glitch": is_past_tense_narration and len(flagged) >= 1,
    }


def opening_vs_rest_comparison(text: str) -> dict:
    """
    Compares how AI-typical the opening reads versus the rest of the
    document. This can show WHERE a document's style changes — it
    cannot and does not indicate WHICH part was written first. A
    finished document carries no timestamp or edit history: the same
    stylistic difference (opening reads more AI-typical than the rest,
    or vice versa) is equally consistent with many different authoring
    orders — an AI-written opening a human continued, a human draft with
    only the opening AI-polished, or simply a human who writes more
    formally at the start and loosens up. Supplementary to the main
    verdict, not a replacement for it — and not a sequence detector.
    """
    clean = text.replace("¬", "").replace("­", "")
    sentences = tokenize_sentences(clean)
    total_words = len(clean.split())

    MIN_TOTAL_WORDS = 180
    MIN_SEGMENT_WORDS = 60
    MIN_SENTENCES_FOR_SPLIT = 8

    if total_words < MIN_TOTAL_WORDS or len(sentences) < MIN_SENTENCES_FOR_SPLIT:
        return {
            "applicable": False,
            "reason": "Text is too short for a reliable opening-vs-rest comparison.",
        }

    split_idx = max(3, len(sentences) // 4)
    opening_text = " ".join(sentences[:split_idx])
    while len(opening_text.split()) < MIN_SEGMENT_WORDS and split_idx < len(sentences) - 3:
        split_idx += 1
        opening_text = " ".join(sentences[:split_idx])

    remainder_text = " ".join(sentences[split_idx:])

    if len(opening_text.split()) < MIN_SEGMENT_WORDS or len(remainder_text.split()) < MIN_SEGMENT_WORDS:
        return {
            "applicable": False,
            "reason": "Could not split text into two segments with enough content for reliable comparison.",
        }

    opening_result = analyse_text(opening_text)
    remainder_result = analyse_text(remainder_text)

    opening_prob = opening_result.get("ai_probability", 0.0)
    remainder_prob = remainder_result.get("ai_probability", 0.0)
    delta = round(abs(opening_prob - remainder_prob), 3)

    return {
        "applicable": True,
        "opening_sentence_count": split_idx,
        "opening_word_count": len(opening_text.split()),
        "remainder_word_count": len(remainder_text.split()),
        "opening_ai_probability": opening_prob,
        "opening_label": opening_result.get("label"),
        "remainder_ai_probability": remainder_prob,
        "remainder_label": remainder_result.get("label"),
        "delta": delta,
        "is_notably_non_uniform": delta >= 0.25,
        "note": (
            "This compares how AI-typical the opening reads versus the rest "
            "of the document. A difference here shows WHERE the text's style "
            "changes — it does not and cannot indicate which part was "
            "written first. The same stylistic pattern is consistent with "
            "many different authoring orders."
        ),
    }


def analyse_text(text: str) -> dict:
    if len(text.strip()) < 50:
        return {
            "label": "insufficient_text",
            "category": category_for_label("insufficient_text"),
            "tier": tier_for_result("insufficient_text", 0),
            "confidence": 0,
            "explanation": "Text is too short for reliable analysis (minimum 50 characters).",
            "signals": {}
        }

    clean_text = text.replace("¬", "").replace("­", "")

    sentences = tokenize_sentences(clean_text)
    perp = compute_perplexity_proxy(clean_text)
    sl_var = sentence_length_variance(sentences)
    hedge = hedge_density(clean_text)
    human = human_marker_density(clean_text)
    vocab = vocabulary_richness(clean_text)
    rep = repetition_score(sentences)

    prof_narr = professional_narrative_score(clean_text)
    creative = creative_writing_signals(clean_text)
    poetry = is_poetry(clean_text)
    purple_prose = purple_prose_signals(clean_text)
    tense_shift = tense_shift_signals(clean_text)
    cliches = cliche_density(clean_text)
    specificity = specificity_score(clean_text)
    self_dep = self_deprecating_density(clean_text)
    autobio = autobiographical_density(clean_text)
    scan_art = has_scan_artifacts(text)
    pre_ai = pre_ai_formal_density(clean_text)
    spell_var = spelling_variation_score(clean_text)

    perp_score = max(0, 1 - (perp / 3.0))
    sl_score = max(0, 1 - (sl_var / 50.0))
    hedge_score = min(1.0, hedge * 15)
    human_score = min(1.0, human * 20)
    vocab_score = max(0, 1 - vocab)
    rep_score = rep

    ai_probability = (
        0.12 * perp_score +
        0.12 * sl_score +
        0.30 * hedge_score +
        0.15 * vocab_score +
        0.16 * rep_score +
        0.15 * (1 - human_score)
    )

    if self_dep > 0:
        ai_probability = max(0, ai_probability - self_dep * 0.20)
    if autobio > 0:
        ai_probability = max(0, ai_probability - autobio * 0.20)
    if scan_art:
        ai_probability = max(0, ai_probability - 0.25)
    if pre_ai > 0:
        ai_probability = max(0, ai_probability - pre_ai * 0.15)
    if spell_var > 0:
        ai_probability = max(0, ai_probability - spell_var * 0.10)

    if creative.get("is_creative"):
        cw_ai_score = 0.0
        hedge_boost = min(1.0, hedge * 15) * 0.30
        cw_ai_score -= hedge_boost * 0.5
        if not creative["has_dialogue"]:
            cw_ai_score += 0.20
        elif creative["has_irony_structure"]:
            cw_ai_score += 0.08
        if creative["symbol_density"] > 0.03:
            cw_ai_score += creative["symbol_density"] * 4
        if creative["efficiency_density"] > 0.3:
            cw_ai_score += creative["efficiency_density"] * 0.3
        if creative["has_irony_structure"]:
            cw_ai_score += 0.22
        if creative["plot_density"] > 0.4:
            cw_ai_score += 0.10
        if creative["plot_density"] > 0.3 and not creative["has_dialogue"]:
            cw_ai_score += 0.10
        # Thematic-closure signal — distinct from irony/dialogue-absence.
        # Gated to prose only (not poetry.is_poetry) since genuine short-line
        # poems already have their own calibrated human-bonus branch below;
        # this shouldn't compete with that.
        if purple_prose.get("is_thematic_closure") and not poetry.get("is_poetry"):
            cw_ai_score += 0.12 * purple_prose["purple_prose_hits"]
        # Tense-shift glitch — a present-tense character-attribute copula
        # slipping into otherwise past-tense narration. Mild signal (a
        # human can absolutely make this slip too), so weighted modestly
        # and capped, gated to prose like purple_prose above.
        if tense_shift.get("has_tense_shift_glitch") and not poetry.get("is_poetry"):
            cw_ai_score += min(0.15, 0.08 * tense_shift["tense_shift_hits"])
        ai_probability = min(1.0, ai_probability + cw_ai_score)

    # FIX (Aug 2026): cliche signal was gated behind poetry.is_poetry, which
    # only fires when the text has actual line breaks. A poem pasted/exported
    # as one continuous paragraph (common — many apps strip newlines on
    # paste) got zero benefit from the cliche count no matter how extreme,
    # because is_poetry() short-circuits to False with no line breaks to
    # analyse. A wall of stock AI-greeting-card phrases is suspicious
    # regardless of formatting, so this no longer requires poetry detection.
    if cliches.get("is_high_cliche"):
        ai_probability = min(1.0, ai_probability + cliches["cliche_count"] * 0.06)
        if cliches["cliche_count"] >= 4:
            ai_probability = min(1.0, ai_probability + 0.25)

    # FIX (Aug 2026): greeting card / storm poem regression.
    # Old logic required cliche_count>=3 AND is_overly_generic together — too
    # strict for greeting cards using fewer than 3 cliches, and separately
    # is_overly_generic false-fired on the storm poem (nautical words weren't
    # in CONCRETE_MARKERS). Fixed by (a) expanding CONCRETE_MARKERS with
    # nautical/nature vocab and (b) scoping the generic-only trigger to
    # short-form, non-rhyming text so long rhyming poems can't be caught by
    # genericness alone; cliche_count>=2 alone is enough to flag a greeting card.
    is_short_form = len(clean_text.split()) <= 60
    is_ai_greeting_card = (
        poetry.get("is_poetry") and (
            cliches.get("cliche_count", 0) >= 2 or
            (specificity.get("is_overly_generic", False) and
             is_short_form and
             not poetry.get("has_rhyme", False))
        )
    )

    if poetry.get("is_poetry") and not is_ai_greeting_card:
        perp_penalty = max(0, 1 - (perp / 3.0)) * 0.12
        ai_probability = max(0, ai_probability - perp_penalty)
        sl_penalty = max(0, 1 - (sl_var / 50.0)) * 0.12
        ai_probability = max(0, ai_probability - sl_penalty)
        if poetry.get("has_poetic_repetition") or poetry.get("has_anaphora"):
            ai_probability = max(0, ai_probability - rep * 0.16)
        if poetry.get("has_sensory_imagery"):
            ai_probability = max(0, ai_probability - 0.15)
        if poetry.get("has_rhetorical_questions"):
            ai_probability = max(0, ai_probability - 0.10)
        if poetry.get("has_rhyme"):
            ai_probability = max(0, ai_probability - 0.20)
            if poetry.get("rhyme_pairs", 0) >= 4:
                ai_probability = max(0, ai_probability - 0.10)

    elif poetry.get("is_poetry") and is_ai_greeting_card:
        perp_penalty = max(0, 1 - (perp / 3.0)) * 0.12
        ai_probability = max(0, ai_probability - perp_penalty)
        sl_penalty = max(0, 1 - (sl_var / 50.0)) * 0.12
        ai_probability = max(0, ai_probability - sl_penalty)
        if poetry.get("has_rhyme"):
            ai_probability = min(1.0, ai_probability + 0.12)
        if specificity.get("is_overly_generic"):
            ai_probability = min(1.0, ai_probability + 0.15)
        if specificity.get("generic_ratio", 0) > 0.90:
            ai_probability = min(1.0, ai_probability + 0.08)

    if prof_narr > 0:
        ai_probability = max(0, ai_probability - prof_narr * 0.20)

    if human_score > 0.1:
        ai_probability = max(0, ai_probability - human_score * 0.25)

    ai_probability = round(min(1.0, max(0.0, ai_probability)), 3)

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
    elif ai_probability >= 0.40:
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

    # Deadzone around the midpoint: a lean this close to 0.5 carries almost
    # no real evidence of direction (demonstrated concretely — a text with
    # known AI-first, human-edited origin scored ai_probability=0.449,
    # landing narrowly on the wrong side and asserting the reverse
    # direction with false confidence). Below this margin, don't guess.
    _raw_lean = ai_probability - 0.5
    _text_lean = _raw_lean if abs(_raw_lean) >= 0.08 else None

    return {
        "label": label,
        "category": category_for_label(label),
        "tier": tier_for_result(label, confidence, lean=_text_lean),
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
            "cliche_count": cliches.get("cliche_count", 0),
            "cliche_density": cliches.get("cliche_density", 0),
            "found_cliches": cliches.get("found_cliches", []),
            "concrete_count": specificity.get("concrete_count", 0),
            "generic_ratio": specificity.get("generic_ratio", 0),
            "is_overly_generic": specificity.get("is_overly_generic", False),
            "is_ai_greeting_card": is_ai_greeting_card,
            "purple_prose_hits": purple_prose.get("purple_prose_hits", 0),
            "is_thematic_closure": purple_prose.get("is_thematic_closure", False),
            "tense_shift_hits": tense_shift.get("tense_shift_hits", 0),
            "has_tense_shift_glitch": tense_shift.get("has_tense_shift_glitch", False),
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
    if not C2PA_OK:
        return {"has_c2pa": False, "error": "c2pa library not available", "manifests": []}
    try:
        import io as _io
        reader = c2pa.Reader(mime_type, _io.BytesIO(content))
        manifest_json = reader.json()
        import json as _json
        manifest_data = _json.loads(manifest_json)
        manifests = manifest_data.get("manifests", {})
        active_manifest = manifest_data.get("active_manifest", "")
        ai_assertions = []
        creator_tool = None
        is_ai_generated = False
        is_ai_trained = False
        for key, manifest in manifests.items():
            for assertion in manifest.get("assertions", []):
                label = assertion.get("label", "")
                data = assertion.get("data", {})
                if "c2pa.ai.generative" in label or "ai.generated" in label.lower():
                    is_ai_generated = True
                    ai_assertions.append(f"AI Generated: {label}")
                if "c2pa.ai.training" in label:
                    is_ai_trained = True
                    ai_assertions.append(f"AI Training: {label}")
                if "c2pa.training-mining" in label:
                    ai_assertions.append(f"Training/Mining: {label}")
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
            "compliant": True,
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
    if not NUMPY_OK or not PIL_OK:
        return {"available": False}
    try:
        gray = np.array(img_rgb.convert("L"), dtype=np.float32)
        h, w = gray.shape
        pil_gray = Image.fromarray(gray.astype(np.uint8))
        blurred = np.array(pil_gray.filter(ImageFilter.GaussianBlur(radius=2)), dtype=np.float32)
        residual = (gray - blurred).flatten()
        residual_std = float(np.std(residual))
        mean_r = float(np.mean(residual))
        std_r = float(np.std(residual))
        kurtosis = float(np.mean(((residual - mean_r) / max(std_r, 0.001)) ** 4))
        patch = 16
        local_vars = []
        for y in range(0, h - patch, patch):
            for x in range(0, w - patch, patch):
                local_vars.append(float(np.var(gray[y:y+patch, x:x+patch])))
        lv_cv = float(np.std(local_vars) / max(np.mean(local_vars), 1)) if local_vars else 0.5
        ai_contribution = 0.0
        fft_signals = []
        fft_human = []
        if residual_std < 15:
            ai_contribution += 0.20
            fft_signals.append(f"Very low high-frequency noise ({residual_std:.1f}) — AI images lack natural camera sensor noise")
        elif residual_std < 25:
            ai_contribution += 0.10
            fft_signals.append(f"Below-average high-frequency noise ({residual_std:.1f}) — possible AI smoothing")
        else:
            fft_human.append(f"Natural high-frequency sensor noise ({residual_std:.1f}) — consistent with real camera photograph")
            ai_contribution -= 0.10
        if 15 < kurtosis < 60:
            ai_contribution += 0.10
            fft_signals.append(f"Frequency noise pattern typical of AI generation (kurtosis: {kurtosis:.1f})")
        elif kurtosis > 60:
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
    width, height = img_rgb.size
    results = {}
    regions = {
        "center": (width//4, height//4, 3*width//4, 3*height//4),
        "top": (0, 0, width, height//3),
        "bottom": (0, 2*height//3, width, height),
        "left": (0, 0, width//3, height),
        "right": (2*width//3, 0, width, height),
    }
    region_stds = []
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
    if not region_stds:
        return {"error": "Could not analyse regions"}
    overall_std = sum(region_stds) / len(region_stds)
    std_variance = (sum((s - overall_std)**2 for s in region_stds) / len(region_stds)) ** 0.5
    center_std = results.get("center", overall_std)
    is_uniform = std_variance < 8.0
    bg_brightness = results.get("bottom", 128)
    has_removed_bg = bg_brightness > 220 and center_std > 30
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
    try:
        gray = img_rgb.convert('L')
        width, height = gray.size
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
        very_smooth_patches = sum(1 for s in patch_stds if s < 3.0) / len(patch_stds)
        patch_std_variance = (sum((s - avg_patch_std)**2 for s in patch_stds) / len(patch_stds)) ** 0.5
        return {
            "avg_patch_noise": round(avg_patch_std, 3),
            "smooth_patch_ratio": round(very_smooth_patches, 3),
            "noise_variance": round(patch_std_variance, 3),
            "is_unnaturally_smooth": avg_patch_std < 8.0 and very_smooth_patches > 0.3,
        }
    except Exception as e:
        return {"error": str(e)[:50]}


AI_WATERMARK_LOGOS = {
    "Grok": {"color_range": [(180, 180, 180), (255, 255, 255)], "position": "bottom_right"},
    "Midjourney": {"color_range": [(200, 200, 200), (255, 255, 255)], "position": "bottom"},
    "Adobe Firefly": {"color_range": [(255, 0, 0), (255, 100, 100)], "position": "bottom_right"},
    "DALL-E": {"color_range": [(0, 0, 0), (50, 50, 50)], "position": "bottom"},
}

def detect_compositing(img_rgb) -> dict:
    if not NUMPY_OK:
        return {"compositing_detected": False}
    try:
        gray = np.array(img_rgb.convert("L"), dtype=np.float32)
        h, w = gray.shape
        patch = 32
        edge_strengths = []
        for y in range(0, h - patch, patch):
            for x in range(0, w - patch, patch):
                region = gray[y:y+patch, x:x+patch]
                dy = np.diff(region, axis=0)
                dx = np.diff(region, axis=1)
                grad_mag = np.sqrt(dy[:dx.shape[0], :dx.shape[1]]**2 +
                                   dx[:dy.shape[0], :dy.shape[1]]**2)
                edge_strengths.append(float(np.mean(grad_mag)))
        if len(edge_strengths) < 4:
            return {"compositing_detected": False}
        es_arr = np.array(edge_strengths)
        es_mean = float(np.mean(es_arr))
        es_std = float(np.std(es_arr))
        es_cv = es_std / max(es_mean, 0.001)
        is_composited = es_cv > 1.2 and es_std > 8
        very_smooth = sum(1 for e in edge_strengths if e < es_mean * 0.3)
        very_sharp = sum(1 for e in edge_strengths if e > es_mean * 2.5)
        has_mixed_sharpness = very_smooth > 2 and very_sharp > 2
        return {
            "compositing_detected": is_composited or has_mixed_sharpness,
            "edge_cv": round(es_cv, 3),
            "edge_std": round(es_std, 3),
            "smooth_regions": very_smooth,
            "sharp_regions": very_sharp,
            "has_mixed_sharpness": has_mixed_sharpness,
        }
    except Exception as e:
        return {"compositing_detected": False, "error": str(e)[:50]}


def detect_ai_watermark(img_rgb) -> dict:
    try:
        width, height = img_rgb.size
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
            r_range = max(r_vals) - min(r_vals)
            g_range = max(g_vals) - min(g_vals)
            b_range = max(b_vals) - min(b_vals)
            max_range = max(r_range, g_range, b_range)
            if max_range > 150 and corner_name in ["bottom_right", "bottom_left", "bottom_center"]:
                white_pixels = sum(1 for p in pixels if p[0] > 200 and p[1] > 200 and p[2] > 200)
                white_ratio = white_pixels / len(pixels)
                black_pixels = sum(1 for p in pixels if p[0] < 50 and p[1] < 50 and p[2] < 50)
                black_ratio = black_pixels / len(pixels)
                if 0.05 < white_ratio < 0.4 and black_ratio < 0.3:
                    watermark_signals.append(f"Possible watermark detected in {corner_name.replace('_', ' ')} (contrast: {max_range}, white pixels: {white_ratio:.0%})")
                    if not found_watermark:
                        found_watermark = corner_name
            if corner_name == "bottom_right":
                grey_pixels = sum(1 for p in pixels if 150 < p[0] < 255 and 150 < p[1] < 255 and 150 < p[2] < 255)
                grey_ratio = grey_pixels / len(pixels)
                bg_pixels = sum(1 for p in pixels if p[0] < 100 and p[1] < 100 and p[2] < 100)
                bg_ratio = bg_pixels / len(pixels)
                if 0.1 < grey_ratio < 0.5 and bg_ratio > 0.1:
                    watermark_signals.append("AI tool logo pattern detected in bottom-right corner — consistent with Grok, Midjourney or similar AI image generator watermark")
                    found_watermark = "bottom_right_logo"
        bottom_strip = img_rgb.crop((0, height - 30, width, height))
        strip_pixels = list(bottom_strip.getdata())
        if strip_pixels:
            strip_r = [p[0] for p in strip_pixels]
            strip_std = (sum((v - sum(strip_r)/len(strip_r))**2 for v in strip_r) / len(strip_r)) ** 0.5
            if strip_std < 15:
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
    signals = {}
    ai_indicators = []
    human_indicators = []

    if not PIL_OK:
        return {
            "label": "unknown",
            "category": category_for_label("unknown"),
            "tier": tier_for_result("unknown", 0),
            "confidence": 0,
            "explanation": "PIL not available for image analysis.",
            "signals": {}
        }

    try:
        img = Image.open(io.BytesIO(content))
    except Exception as e:
        return {
            "label": "error",
            "category": category_for_label("error"),
            "tier": tier_for_result("error", 0),
            "confidence": 0,
            "explanation": f"Could not open image: {str(e)}",
            "signals": {}
        }

    exif_data = {}
    has_camera_exif = False
    has_ai_software_tag = False

    try:
        raw_exif = img._getexif()
        if raw_exif:
            for tag_id, value in raw_exif.items():
                tag = TAGS.get(tag_id, str(tag_id))
                exif_data[tag.lower()] = str(value)[:200]
            for marker in CAMERA_MARKERS:
                if marker in exif_data:
                    has_camera_exif = True
                    break
            software = exif_data.get("software", "").lower()
            for tool in AI_IMAGE_TOOLS:
                if tool in software:
                    has_ai_software_tag = True
                    ai_indicators.append(f"Software tag: {software}")
                    break
            if has_camera_exif:
                human_indicators.append("Camera EXIF data present (Make, Model, GPS, etc.) — note: genuine camera metadata confirms a real camera captured this file, but not that the scene in frame wasn't itself a screen playing AI-generated content")
            elif not has_ai_software_tag:
                ai_indicators.append("No camera EXIF metadata (typical of AI-generated images)")
        else:
            ai_indicators.append("No EXIF metadata found")
    except Exception:
        ai_indicators.append("Could not read EXIF metadata")

    signals["has_camera_exif"] = has_camera_exif
    signals["has_ai_software_tag"] = has_ai_software_tag
    signals["exif_fields"] = list(exif_data.keys())[:10]

    file_size_kb = len(content) / 1024
    signals["file_size_kb"] = round(file_size_kb, 1)

    # Cap the working copy used for per-pixel analysis. list(img.getdata())
    # on a full-resolution phone photo (12MP+) materialises millions of
    # Python tuple objects — repeated across region/noise/compositing/FFT
    # passes this was the main driver behind Render's memory-limit restarts.
    # Downscaling here doesn't change EXIF/format/dimension signals below,
    # only the pixel-statistics passes, which don't need full resolution.
    MAX_ANALYSIS_DIM = 1600
    orig_w, orig_h = img.size
    if max(orig_w, orig_h) > MAX_ANALYSIS_DIM:
        analysis_src = img.convert("RGB")
        analysis_src.thumbnail((MAX_ANALYSIS_DIM, MAX_ANALYSIS_DIM), Image.LANCZOS)
    else:
        analysis_src = img.convert("RGB")

    try:
        comp_result = detect_compositing(analysis_src)
        signals["compositing_check"] = comp_result
        if comp_result.get("compositing_detected"):
            ai_indicators.append(f"Edge sharpness inconsistency detected (cv: {comp_result.get('edge_cv', 0)}) — possible AI element composited into real photograph")
    except Exception as e:
        signals["compositing_error"] = str(e)[:50]

    try:
        wm_result = detect_ai_watermark(analysis_src)
        signals["watermark_check"] = wm_result
        if wm_result.get("watermark_detected"):
            for sig in wm_result.get("watermark_signals", []):
                ai_indicators.append(f"🔍 Watermark: {sig}")
    except Exception as e:
        signals["watermark_error"] = str(e)[:50]

    try:
        img_rgb = analysis_src
        width, height = orig_w, orig_h
        signals["dimensions"] = f"{width}x{height}"
        ai_dims = [(1024, 1024), (512, 512), (768, 768), (1024, 768),
                   (768, 1024), (1024, 1536), (1536, 1024), (2048, 2048)]
        if (width, height) in ai_dims:
            ai_indicators.append(f"Common AI generation resolution: {width}x{height}")
        elif width == height:
            ai_indicators.append(f"Square image ({width}x{height}) — common in AI generation")
        # Sample directly from the (already capped) analysis copy instead of
        # materialising a full getdata() list then slicing it down.
        aw, ah = img_rgb.size
        step = max(1, int(((aw * ah) / 10000) ** 0.5))
        sample = [img_rgb.getpixel((x, y))
                  for y in range(0, ah, step)
                  for x in range(0, aw, step)]
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
        channel_stds = [r_std, g_std, b_std]
        avg_std = sum(channel_stds) / 3
        signals["avg_channel_std"] = round(avg_std, 2)
        if avg_std > 60:
            human_indicators.append(f"High color variance ({avg_std:.1f}) — natural photograph characteristic")
        elif avg_std < 35:
            ai_indicators.append(f"Low color variance ({avg_std:.1f}) — smooth gradients typical of AI generation")
        region_data = analyse_regions(img_rgb)
        signals["region_analysis"] = region_data
        if region_data.get("has_removed_background"):
            human_indicators.append("Background removal detected — real photo with background edited out")
            signals["background_removed"] = True
        if region_data.get("is_texture_uniform"):
            ai_indicators.append(f"Unnaturally uniform texture across image regions (variance: {region_data.get('texture_variance_across_regions', 0)}) — typical of AI generation")
        if region_data.get("center_smoother_than_average"):
            ai_indicators.append(f"Center region unusually smooth (std: {region_data.get('center_std', 0)}) — possible AI portrait skin smoothing")
        noise_data = analyse_noise_pattern(img_rgb)
        signals["noise_analysis"] = noise_data
        if noise_data.get("is_unnaturally_smooth"):
            ai_indicators.append(f"Unnaturally low pixel noise (avg: {noise_data.get('avg_patch_noise', 0)}, smooth patches: {noise_data.get('smooth_patch_ratio', 0)*100:.0f}%) — AI images lack natural camera sensor noise")
        elif noise_data.get("avg_patch_noise", 0) > 12:
            human_indicators.append(f"Natural camera sensor noise detected (avg: {noise_data.get('avg_patch_noise', 0)}) — consistent with real photograph")
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

    fmt = img.format or "unknown"
    signals["format"] = fmt
    signals["mode"] = img.mode

    if fmt in ("JPEG", "JPG") and has_camera_exif:
        human_indicators.append("JPEG with camera metadata — consistent with real photograph")

    is_likely_screenshot = False
    try:
        is_png = (signals.get("format", "") == "PNG")
        is_small = file_size_kb < 100
        is_low_color = signals.get("avg_channel_std", 100) < 30
        no_exif = not has_camera_exif
        if 'color_stats' in signals:
            r_mean = signals['color_stats']['r']['mean']
            g_mean = signals['color_stats']['g']['mean']
            b_mean = signals['color_stats']['b']['mean']
            avg_brightness = (r_mean + g_mean + b_mean) / 3
            is_mostly_white = avg_brightness > 180
        else:
            is_mostly_white = False
        if is_png and no_exif and is_small and is_low_color and is_mostly_white:
            is_likely_screenshot = True
            ai_indicators = [i for i in ai_indicators if 'EXIF' not in i and 'color variance' not in i]
            human_indicators.append(f"Screenshot or document export detected (PNG, {file_size_kb:.1f}KB, white background, low color complexity)")
    except Exception:
        pass

    # No-EXIF is weak evidence on its own — most messaging/social platforms
    # (WhatsApp, Instagram, Messenger, Discord) strip EXIF automatically,
    # so its absence is common for ordinary human photos, not just AI
    # output. Discounted rather than removed entirely, since it's still
    # mild corroborating evidence when combined with other signals.
    EXIF_WEAK_SIGNAL_PHRASES = ("No EXIF metadata found", "No camera EXIF metadata (typical of AI-generated images)")
    has_weak_exif_signal = any(p in ai_indicators for p in EXIF_WEAK_SIGNAL_PHRASES)
    ai_score = len(ai_indicators) - (0.6 if has_weak_exif_signal else 0)

    # Genuine camera EXIF only proves a real camera captured the file — not
    # that the scene in frame wasn't itself a screen playing AI-generated
    # content (a real camera recording a monitor produces fully authentic
    # metadata). Narrower attack vector than EXIF stripping, so a lighter
    # discount than the no-EXIF case above, but still shouldn't carry full
    # weight on its own.
    has_camera_exif_signal = any("Camera EXIF data present" in h for h in human_indicators)
    human_score = len(human_indicators) - (0.3 if has_camera_exif_signal else 0)
    total = ai_score + human_score

    if is_likely_screenshot:
        return {
            "label": "human_created",
            "category": category_for_label("human_created"),
            "tier": tier_for_result("human_created", 85),
            "confidence": 85,
            "explanation": f"This appears to be a screenshot, diagram, or document export — not an AI-generated image. Signals: small file size ({file_size_kb:.1f}KB), white/light background, PNG format, no camera metadata.",
            "ai_indicators": ai_indicators,
            "human_indicators": human_indicators,
            "signals": signals,
        }

    fft_contribution = signals.get("fft_analysis", {}).get("ai_contribution", 0)
    wm_detected = signals.get("watermark_check", {}).get("watermark_detected", False)
    # Watermark corner-matching is a weak heuristic — busy/high-contrast real
    # photo content (bright floors, patterned surfaces, windows) can trigger
    # the same contrast/white-pixel-ratio pattern as an actual logo. It's
    # already counted once via ai_indicators/ai_score above, so it isn't
    # given any extra weight here — it blends with every other signal
    # instead of single-handedly forcing "ai_generated" regardless of what
    # the rest of the analysis found.
    fft_adjusted_ai = ai_score + (fft_contribution * 2)

    if has_ai_software_tag:
        label = "ai_generated"
        confidence = 95
        explanation = f"AI generation tool detected in image metadata. {'; '.join(ai_indicators)}"
    elif total == 0 and fft_contribution < 0.1:
        label = "unknown"
        confidence = 50
        explanation = "Insufficient signals to classify this image reliably."
    elif fft_adjusted_ai > human_score * 1.2 or (fft_contribution >= 0.12 and ai_score >= human_score):
        label = "ai_generated"
        # Was a hard min(92, ...) clamp — any score past the ceiling collapsed
        # to the identical number, so a mildly-AI image and an overwhelmingly
        # obvious one (e.g. one with a visible generator watermark vs. the
        # same image with it manually edited out) could read as the exact
        # same confidence. This exponential approach-to-ceiling keeps
        # climbing toward but never quite reaches 96, so a stronger combined
        # signal count still produces a visibly higher number.
        import math as _math
        s = fft_adjusted_ai * 8 + fft_contribution * 20
        confidence = round(55 + 41 * (1 - _math.exp(-s / 35)))
        fft_note = f" Frequency analysis: low sensor noise detected." if fft_contribution > 0.1 else ""
        wm_note = " Possible watermark match detected (weighted, not decisive on its own)." if wm_detected else ""
        explanation = f"Multiple AI generation signals detected: {'; '.join(ai_indicators)}.{fft_note}{wm_note}"
    elif human_score > fft_adjusted_ai:
        label = "human_captured"
        import math as _math
        s = human_score * 10
        confidence = round(55 + 37 * (1 - _math.exp(-s / 30)))
        explanation = f"Photograph characteristics detected: {'; '.join(human_indicators)}"
    else:
        comp_detected = signals.get("compositing_check", {}).get("compositing_detected", False)
        if comp_detected:
            label = "ai_assisted"
            confidence = 60
            explanation = "Mixed signals — possible AI element composited into a real photograph. Edge sharpness inconsistency detected between regions."
        else:
            label = "uncertain"
            confidence = 50
            explanation = f"Mixed signals. AI indicators: {'; '.join(ai_indicators) if ai_indicators else 'none'}. Human indicators: {'; '.join(human_indicators) if human_indicators else 'none'}."

    # "unknown" specifically means zero indicators either way — don't invent
    # a lean from a near-zero fft_contribution alone, that would spuriously
    # claim a direction the data doesn't support.
    img_lean = None if label == "unknown" else (fft_adjusted_ai - human_score)

    return {
        "label": label,
        "category": category_for_label(label),
        "tier": tier_for_result(label, confidence, lean=img_lean),
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
# ─────────────────────────────────────────────────────────────────────────────

AI_VIDEO_TOOLS = [
    "sora", "runway", "runwayml", "pika", "pika labs", "kling", "kling ai",
    "stable video", "stable video diffusion", "animatediff", "modelscope",
    "zeroscope", "gen-2", "gen-3", "synthesia", "heygen", "d-id",
    "invideo ai", "pictory", "lumen5", "veed.io", "fliki", "elai",
    "colossyan", "deepbrain", "hour one", "vidnoz", "capcut ai",
    "adobe firefly video", "adobe express", "canva ai video",
]

# FIX: split into distinctive vs generic tiers. A lone match on a generic
# word like "gps" or "make" is weak evidence given the crude raw-byte
# scanning method (no real structured metadata parsing) — it could be
# coincidental boilerplate from any export pipeline, AI tools included.
# A distinctive brand/device name is much less likely to appear by chance.
CAMERA_VIDEO_MARKERS_DISTINCTIVE = [
    "sony", "canon", "nikon", "gopro", "iphone", "samsung", "pixel",
    "dji", "fujifilm", "panasonic", "olympus",
]
CAMERA_VIDEO_MARKERS_GENERIC = [
    # "iso" removed — it isn't distinguishable from "isom"/"iso2", the
    # standard MP4 file-type brand every MP4 container has regardless of
    # source (a bare-minimum synthetic MP4 with zero real metadata was
    # confirmed to trigger a false "camera detected" match on this alone).
    "make", "model", "gps", "camera", "lens", "exposure",
]
CAMERA_VIDEO_MARKERS = CAMERA_VIDEO_MARKERS_DISTINCTIVE + CAMERA_VIDEO_MARKERS_GENERIC

# ── Real MP4/QuickTime atom-tree parser ──────────────────────────────────────
# MP4/MOV files are structured as nested "atoms" (4-byte size + 4-byte type,
# recursively containing more atoms) per Apple's public QuickTime File Format
# spec. This walks that real structure to extract genuine device metadata —
# a validated ISO 6709 GPS string, or an actual Make/Model/Software value —
# rather than scanning raw bytes for a keyword that could be coincidental.
MP4_CONTAINER_ATOMS = {
    b'moov', b'trak', b'mdia', b'minf', b'stbl', b'udta',
    b'edts', b'mvex', b'moof', b'traf', b'dinf',
}
MP4_QT_STRING_ATOMS = {
    b'\xa9xyz': 'gps_iso6709',   # GPS location, ISO 6709 format
    b'\xa9mak': 'make',
    b'\xa9mod': 'model',
    b'\xa9swr': 'software',
}
ISO6709_RE = re.compile(r'^[+-]\d{2,3}(\.\d+)?[+-]\d{3}(\.\d+)?([+-]\d+(\.\d+)?)?/?$')


def parse_mp4_atoms(data: bytes, max_depth: int = 8, _depth: int = 0) -> dict:
    """Walk an MP4/MOV atom tree, returning any recognized genuine
    structured metadata found. Non-container atoms (including the large
    'mdat' video payload) are skipped via their size field without being
    read into memory or decoded — fast even on large files, and safe
    against malformed/truncated/non-MP4 input (returns {} rather than
    raising)."""
    found = {}
    try:
        if _depth > max_depth:
            return found
        pos = 0
        n = len(data)
        while pos + 8 <= n:
            size = struct.unpack('>I', data[pos:pos+4])[0]
            atype = data[pos+4:pos+8]
            header_len = 8
            if size == 1:
                if pos + 16 > n:
                    break
                size = struct.unpack('>Q', data[pos+8:pos+16])[0]
                header_len = 16
            elif size == 0:
                size = n - pos
            if size < header_len or pos + size > n:
                break
            payload = data[pos+header_len: pos+size]

            if atype in MP4_CONTAINER_ATOMS:
                found.update(parse_mp4_atoms(payload, max_depth, _depth + 1))
            elif atype == b'meta':
                inner = payload[4:] if len(payload) > 4 else payload
                found.update(parse_mp4_atoms(inner, max_depth, _depth + 1))
            elif atype in MP4_QT_STRING_ATOMS:
                label = MP4_QT_STRING_ATOMS[atype]
                if len(payload) >= 4:
                    str_len = struct.unpack('>H', payload[0:2])[0]
                    text = payload[4:4+str_len].decode('utf-8', errors='ignore')
                    if text:
                        found[label] = text

            pos += size
    except Exception:
        return found
    return found


def validate_gps_iso6709(value: str) -> bool:
    return bool(ISO6709_RE.match(value.strip()))


def analyse_video(content: bytes, filename: str) -> dict:
    ai_indicators = []
    human_indicators = []
    signals = {}

    file_size_mb = len(content) / (1024 * 1024)
    signals["file_size_mb"] = round(file_size_mb, 2)
    signals["filename"] = filename

    fname = filename.lower()
    if fname.endswith(".mp4"):
        signals["format"] = "MP4"
    elif fname.endswith(".mov"):
        signals["format"] = "MOV"
    elif fname.endswith(".avi"):
        signals["format"] = "AVI"
    elif fname.endswith(".webm"):
        signals["format"] = "WebM"
        ai_indicators.append("WebM format — common AI video output format")
    else:
        signals["format"] = fname.split(".")[-1].upper()

    header = content[:65536]
    try:
        header_text = header.decode("latin-1", errors="ignore").lower()
    except Exception:
        header_text = ""

    # Genuine structured metadata (real atom-tree parsing, not keyword
    # scanning) — parsed over the FULL content since the metadata atom can
    # sit after a large video payload, not just in the first 64KB.
    structured_metadata = parse_mp4_atoms(content)
    signals["structured_metadata"] = structured_metadata
    if structured_metadata.get("make") or structured_metadata.get("model") or structured_metadata.get("software"):
        device_parts = [f"{k}={v}" for k, v in structured_metadata.items() if k in ("make", "model", "software")]
        human_indicators.append(f"Genuine structured device metadata found: {', '.join(device_parts)}")
    gps_raw = structured_metadata.get("gps_iso6709")
    if gps_raw and validate_gps_iso6709(gps_raw):
        human_indicators.append(f"GPS location metadata validates as genuine ISO 6709 format: {gps_raw}")
        signals["gps_validated"] = True

    found_ai_tools = []
    for tool in AI_VIDEO_TOOLS:
        if tool in header_text:
            found_ai_tools.append(tool)
    if found_ai_tools:
        ai_indicators.append(f"AI video tool signature detected: {', '.join(found_ai_tools)}")
        signals["ai_tools_found"] = found_ai_tools

    found_camera = []
    for marker in CAMERA_VIDEO_MARKERS:
        if re.search(r'\b' + re.escape(marker) + r'\b', header_text):
            found_camera.append(marker)
    found_distinctive = [m for m in found_camera if m in CAMERA_VIDEO_MARKERS_DISTINCTIVE]
    found_generic = [m for m in found_camera if m in CAMERA_VIDEO_MARKERS_GENERIC]
    is_meaningful_camera_evidence = len(found_distinctive) >= 1 or len(found_generic) >= 2
    if found_camera and is_meaningful_camera_evidence:
        human_indicators.append(f"Camera/device metadata found: {', '.join(found_camera[:3])}")
        signals["camera_markers"] = found_camera[:5]

    ai_res_signatures = [b"512x512", b"1024x576", b"576x1024", b"768x432", b"432x768"]
    for res in ai_res_signatures:
        if res in content[:65536]:
            ai_indicators.append(f"Common AI video resolution detected: {res.decode()}")
            break

    if file_size_mb < 2.0 and signals["format"] == "MP4":
        ai_indicators.append(f"Very small MP4 ({file_size_mb:.1f}MB) — may indicate AI-generated short clip")
    elif file_size_mb > 50:
        human_indicators.append(f"Large file size ({file_size_mb:.1f}MB) — consistent with real camera footage")

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
        "category": category_for_label(label),
        "tier": tier_for_result(label, confidence, lean=ai_score - human_score),
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
    text = payload.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(status_code=413, detail=f"Text too long — maximum {MAX_TEXT_CHARS:,} characters")
    cache_key = hashlib.sha256(text.encode()).hexdigest()
    if cache_key in cache:
        return cache[cache_key]
    result = analyse_text(text)
    result["opening_vs_rest"] = opening_vs_rest_comparison(text)
    cache[cache_key] = result
    return result


@app.post("/api/check-document")
async def check_document(file: UploadFile = File(...)):
    content = await file.read()
    if len(content) > MAX_DOCUMENT_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large — maximum {MAX_DOCUMENT_BYTES // (1024*1024)}MB")
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
    content = await file.read()
    if len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large — maximum {MAX_IMAGE_BYTES // (1024*1024)}MB")
    cache_key = hashlib.sha256(content).hexdigest()
    if cache_key in cache:
        return cache[cache_key]
    filename = file.filename.lower()
    allowed = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff")
    if not any(filename.endswith(ext) for ext in allowed):
        raise HTTPException(status_code=400, detail=f"Unsupported image format: {filename}")
    result = analyse_image(content)
    result["filename"] = file.filename
    fname = file.filename.lower()
    if fname.endswith(".jpg") or fname.endswith(".jpeg"):
        mime = "image/jpeg"
    elif fname.endswith(".webp"):
        mime = "image/webp"
    else:
        mime = "image/png"
    c2pa_result = check_c2pa(content, mime)
    result["c2pa"] = c2pa_result
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
    content = await file.read()
    if len(content) > MAX_VIDEO_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large — maximum {MAX_VIDEO_BYTES // (1024*1024)}MB")
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
    # Render assigns a dynamic port via the PORT env var and health-checks
    # against it — hardcoding a port causes "Port scan timeout reached".
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)

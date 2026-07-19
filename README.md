# AI Content Detector

Detects AI-generated, AI-assisted, or human-written content in text, documents, and images.

## Setup

```bash
pip install -r requirements.txt
python main.py
```

Then open: http://localhost:8001

## API Endpoints

### POST /api/check-text
```json
{ "text": "your text here" }
```

### POST /api/check-document
Upload file (multipart/form-data), field name: `file`  
Supports: .pdf, .docx, .txt

### POST /api/check-image
Upload file (multipart/form-data), field name: `file`  
Supports: .jpg, .jpeg, .png, .webp, .bmp, .tiff

### GET /api/health
Returns capability status.

## Response Format (Text/Document)

```json
{
  "label": "ai_generated | ai_assisted | human_written",
  "confidence": 75,
  "ai_probability": 0.75,
  "explanation": "...",
  "signals": {
    "entropy_variance": 0.42,
    "sentence_length_variance": 18.3,
    "hedge_word_density": 0.045,
    "human_marker_density": 0.002,
    "vocabulary_richness": 0.61,
    "repetition_score": 0.38
  },
  "word_count": 312,
  "sentence_count": 18
}
```

## Response Format (Image)

```json
{
  "label": "ai_generated | human_captured | uncertain",
  "confidence": 80,
  "explanation": "...",
  "ai_indicators": ["No camera EXIF metadata", "Common AI resolution 1024x1024"],
  "human_indicators": ["Camera EXIF data present"],
  "signals": {
    "dimensions": "1024x1024",
    "format": "PNG",
    "has_camera_exif": false,
    "avg_channel_std": 42.3
  }
}
```

## How It Works

### Text Analysis (Statistical NLP)
Uses 6 signals — same approach as early GPTZero:

1. **Entropy variance** — AI text has lower, more uniform character entropy (less "bursty")
2. **Sentence length variance** — AI tends toward more uniform sentence lengths
3. **Hedge word density** — AI uses formal transitions more (furthermore, moreover, it is worth noting)
4. **Human marker density** — colloquial words (honestly, kinda, tbh, i think)
5. **Vocabulary richness** — type-token ratio; AI can be repetitive
6. **Repetition score** — repeated sentence opening patterns

### Image Analysis (Metadata + Statistics)
1. **EXIF metadata** — real cameras embed Make, Model, GPS, exposure settings; AI tools don't
2. **Software tags** — checks for Stable Diffusion, Midjourney, DALL-E, Firefly etc.
3. **Dimensions** — checks for common AI generation sizes (1024x1024, 512x512 etc.)
4. **Color variance** — smooth gradients typical of AI; high variance typical of real photos

## Adding External API Detectors

To add Hive, SightEngine, or AI or Not:

```python
# In main.py, add to check_image():
if HIVE_API_KEY:
    hive_result = await hive_detect(content, HIVE_API_KEY)
    
if SIGHTENGINE_API_KEY:
    sight_result = await sightengine_detect(content, SIGHTENGINE_API_KEY)

# Fuse results
final_confidence = (local_confidence + hive_confidence + sight_confidence) / 3
```

## Accuracy Notes

- Text: Best with 500+ words. Short texts may be unreliable.
- Images: Metadata-based detection is very reliable; statistical signals are supplementary.
- This is a local statistical tool, not a trained ML model.
- For production use, combine with Hive Moderation / SightEngine / AI or Not APIs.

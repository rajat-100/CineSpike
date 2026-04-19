"""
tag_generator.py
Extracts genre + emotion tags from a video trailer via CLIP zero-shot classification.
Falls back gracefully to manual/default tags if torch/transformers are unavailable.
"""

GENRE_PROMPTS = [
    "action movie",
    "comedy film",
    "horror movie",
    "science fiction film",
    "romance movie",
    "drama film",
    "thriller movie",
    "animated film",
    "fantasy movie",
    "crime movie",
]

EMOTION_PROMPTS = [
    "tense and suspenseful",
    "happy and uplifting",
    "mysterious and dark",
    "romantic and emotional",
    "scary and frightening",
    "exciting and adventurous",
    "sad and melancholic",
    "funny and comedic",
]

KEYWORD_MAP = {
    "action movie":           ["action", "fight", "chase", "explosion", "combat", "hero"],
    "comedy film":            ["comedy", "humor", "funny", "laugh", "satire", "witty"],
    "horror movie":           ["horror", "fear", "monster", "supernatural", "scary", "dread"],
    "science fiction film":   ["sci-fi", "space", "future", "robot", "alien", "technology"],
    "romance movie":          ["romance", "love", "relationship", "passion", "heartbreak"],
    "drama film":             ["drama", "emotion", "family", "conflict", "struggle", "grief"],
    "thriller movie":         ["thriller", "suspense", "mystery", "crime", "twist", "paranoia"],
    "animated film":          ["animation", "cartoon", "family", "colorful", "fantastical"],
    "fantasy movie":          ["fantasy", "magic", "dragon", "quest", "mythical", "kingdom"],
    "crime movie":            ["crime", "detective", "heist", "gangster", "murder", "cop"],
}


def generate_tags(video_path: str) -> dict:
    """
    Primary entry point. Tries CLIP first; falls back to defaults.
    Returns:
        {
          genres: [str], emotions: [str], keywords: [str],
          confidence_scores: {genres:{}, emotions:{}}, method: str
        }
    """
    try:
        return _clip_tags(video_path)
    except Exception as exc:
        print(f"[TagGenerator] CLIP unavailable ({exc}). Using fallback.")
        return _fallback_tags()


def manual_tags(genres: list, emotions: list) -> dict:
    """Called when the user explicitly picks genres/emotions in the UI."""
    keywords = []
    for g in genres:
        for prompt, kws in KEYWORD_MAP.items():
            if g.lower() in prompt:
                keywords.extend(kws)
    n_g = len(genres) or 1
    n_e = len(emotions) or 1
    return {
        "genres":   genres,
        "emotions": emotions,
        "keywords": list(dict.fromkeys(keywords))[:12],
        "confidence_scores": {
            "genres":   {g: round(100 / n_g, 1) for g in genres},
            "emotions": {e: round(100 / n_e, 1) for e in emotions},
        },
        "method": "manual",
    }


# ── CLIP implementation ───────────────────────────────────────────────────────
def _clip_tags(video_path: str) -> dict:
    import torch
    from transformers import CLIPProcessor, CLIPModel
    import cv2
    from PIL import Image

    model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    # ── Sample frames every 2 s (max 20 frames) ──────────────────────────
    cap    = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 24
    step   = max(1, int(fps * 2))
    frames = []
    i      = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i % step == 0:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            if len(frames) >= 20:
                break
        i += 1
    cap.release()

    if not frames:
        return _fallback_tags()

    def score_prompts(prompts: list) -> dict:
        acc = {p: 0.0 for p in prompts}
        for frame in frames:
            inputs = processor(text=prompts, images=frame, return_tensors="pt", padding=True)
            with torch.no_grad():
                probs = model(**inputs).logits_per_image.softmax(dim=1)[0].tolist()
            for p, prob in zip(prompts, probs):
                acc[p] += prob
        return {p: v / len(frames) for p, v in acc.items()}

    genre_scores   = score_prompts(GENRE_PROMPTS)
    emotion_scores = score_prompts(EMOTION_PROMPTS)

    top_genres   = sorted(genre_scores,   key=genre_scores.get,   reverse=True)[:3]
    top_emotions = sorted(emotion_scores, key=emotion_scores.get, reverse=True)[:3]

    keywords = []
    for g in top_genres:
        keywords.extend(KEYWORD_MAP.get(g, []))

    return {
        "genres":   top_genres,
        "emotions": top_emotions,
        "keywords": list(dict.fromkeys(keywords))[:12],
        "confidence_scores": {
            "genres":   {g: round(genre_scores[g] * 100, 1)   for g in GENRE_PROMPTS},
            "emotions": {e: round(emotion_scores[e] * 100, 1) for e in EMOTION_PROMPTS},
        },
        "method": "clip",
    }


def _fallback_tags() -> dict:
    return {
        "genres":   ["drama film", "thriller movie"],
        "emotions": ["tense and suspenseful", "mysterious and dark"],
        "keywords": ["drama", "suspense", "mystery", "conflict"],
        "confidence_scores": {
            "genres":   {"drama film": 50.0, "thriller movie": 40.0},
            "emotions": {"tense and suspenseful": 55.0, "mysterious and dark": 35.0},
        },
        "method": "fallback",
    }

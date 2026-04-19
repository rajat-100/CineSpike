"""
release_planner.py
Recommends optimal release window using historical data from top-10 movies.
Campaign generation uses Groq's chat completions API when a GROQ_API_KEY is set.
"""

import os
import json
import requests
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()

MONTH_NAMES = {
    1: "January",   2: "February", 3: "March",     4: "April",
    5: "May",       6: "June",     7: "July",       8: "August",
    9: "September",10: "October", 11: "November",  12: "December",
}

SEASON_LABELS = {
    1: "Winter — Post-holiday, low competition",
    2: "Winter — Valentine's / Presidents' Day bump",
    3: "Spring — Spring Break kicks off",
    4: "Spring — Low competition, awards residuals",
    5: "Summer — Blockbuster season opens",
    6: "Summer — Peak box office",
    7: "Summer — Peak box office",
    8: "Late Summer — Winding down, niche openers",
    9: "Fall — Festival rollout / prestige season starts",
   10: "Fall — Halloween season",
   11: "Fall — Thanksgiving & year-end awards push",
   12: "Holiday — Awards season, family blockbusters",
}

# Months genres should generally avoid
AVOID = {
    "action":          [1, 2, 9],
    "horror":          [1, 2, 6, 7],
    "science fiction": [1, 2],
    "comedy":          [8, 9],
    "drama":           [6, 7, 8],
    "romance":         [3, 8, 9],
    "thriller":        [1, 2],
    "fantasy":         [1, 2, 8],
    "crime":           [1, 6, 7],
    "animation":       [9, 10],
}


def suggest_release_window(top_movies: list, tags: dict) -> dict:
    months      = _extract_months(top_movies)
    freq        = Counter(months)
    genres      = [g.lower() for g in tags.get("genres", [])]
    genre_key   = _match_genre(genres[0] if genres else "drama")
    avoid_months = AVOID.get(genre_key, [])

    # Pick best month (most frequent among comparables, not in avoid list)
    best_month = 6  # default: June
    for month, _ in freq.most_common():
        if month not in avoid_months:
            best_month = month
            break

    monthly_dist = {str(m): freq.get(m, 0) for m in range(1, 13)}
    movie_dates  = [
        {"title": m.get("title", "?"), "date": m.get("release_date", ""), "revenue": m.get("revenue", 0)}
        for m in top_movies if m.get("release_date")
    ]

    reasoning = _rule_reasoning(best_month, genre_key, freq)

    return {
        "recommended_month":      best_month,
        "recommended_month_name": MONTH_NAMES.get(best_month, ""),
        "season_label":           SEASON_LABELS.get(best_month, ""),
        "recommended_quarter":    f"Q{(best_month - 1) // 3 + 1}",
        "avoid_months":           [MONTH_NAMES[m] for m in avoid_months if m in MONTH_NAMES],
        "monthly_distribution":   monthly_dist,
        "reasoning":              reasoning,
        "method":                 "rule_based",
        "comparable_releases":    movie_dates[:5],
    }


# ── Helpers ──────────────────────────────────────────────────────────────────

def _extract_months(movies: list) -> list:
    months = []
    for m in movies:
        rd = m.get("release_date", "") or ""
        if len(rd) >= 7:
            try:
                months.append(int(rd[5:7]))
            except ValueError:
                pass
    return months


def _rule_reasoning(month: int, genre: str, freq: Counter) -> str:
    season  = SEASON_LABELS.get(month, f"Month {month}")
    top2    = [MONTH_NAMES[m] for m, _ in freq.most_common(2) if m in MONTH_NAMES]
    top_str = " and ".join(top2) if top2 else "similar periods"
    return (
        f"Analysis of comparable {genre} films shows a strong clustering of releases in {top_str}, "
        f"indicating proven audience demand during {MONTH_NAMES.get(month, str(month))}. "
        f"The {season} window offers the right competitive environment for this film's profile. "
        f"Targeting this window aligns with historical box-office peaks for the genre and maximises "
        f"overlap with the core demographic's peak cinema attendance."
    )


def _match_genre(raw: str) -> str:
    for key in AVOID:
        if key in raw or raw in key:
            return key
    return "drama"


def generate_ai_campaign(tags: dict, top_movies: list, audience: dict) -> dict:
    """
    On-demand campaign generation using the Groq chat completions API.
    Requires GROQ_API_KEY in the environment.
    """
    lines = "\n".join(
        f"- {m.get('title', '?')} ({m.get('release_date', '?')})" for m in top_movies[:10]
    )
    genres   = ", ".join(tags.get("genres", []))
    emotions = ", ".join(tags.get("emotions", []))
    
    subs = [s.get("name") for s in audience.get("primary_subreddits", [])]
    subs_str = ", ".join(subs) if subs else "relevant Reddit communities"
    demo_age = audience.get("demographics", {}).get("age_range", "adults")

    prompt = f"""You are an elite Hollywood marketing executive.

Your task is to design a high-impact, data-driven promotional campaign for a new movie.

**Film Profile:**
Genres: {genres}
Emotional Tone: {emotions}

**Target Audience:**
Age Range: {demo_age}
Key Online Communities: {subs_str}

**Comparable Films & Release Dates:**
{lines}

Create a structured AI campaign strategy. You MUST reply in valid, parseable JSON with the exact following schema and nothing else:

{{
  "overview": "A 2-sentence overarching theme for the campaign.",
  "release_analysis": "Based on the comparable films, explain why specific months are ideal and how this campaign timeline aligns.",
  "phases": [
    {{
      "name": "Phase 1: The Teaser (3 Months Out)",
      "tactics": ["Tactic 1", "Tactic 2", "Tactic 3"]
    }},
    {{
      "name": "Phase 2: The Push (1 Month Out)",
      "tactics": ["Tactic 1", "Tactic 2", "Tactic 3"]
    }}
  ],
  "guerrilla_marketing": "One specific, highly creative out-of-the-box marketing idea to go viral on the subreddits listed above."
}}

JSON only:
"""

    if not GROQ_API_KEY:
        raise Exception("Groq is not configured. Add GROQ_API_KEY to your .env file.")

    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "temperature": 0.7,
                "response_format": {"type": "json_object"},
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a precise marketing strategist. Respond with valid JSON only.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            },
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()
        response_text = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "{}")
        )

        parsed = json.loads(response_text)
        return parsed
    except requests.exceptions.HTTPError as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        raise Exception(f"Groq request failed: {detail}")
    except Exception as exc:
        raise Exception(f"Failed to generate campaign: {str(exc)}")


# Backward-compatible alias for existing imports.
generate_ollama_campaign = generate_ai_campaign

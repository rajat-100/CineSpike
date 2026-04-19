"""
reddit_mapper.py
Maps genre/emotion tags → synthetic Reddit subreddit audience data.
Includes demographics, sample posts, subreddit list, and marketing advice.
"""

from collections import defaultdict

# ── Subreddit Pool ────────────────────────────────────────────────────────────
_SUBREDDITS = {
    "action": {
        "primary": [
            {"name": "r/ActionMovies",  "subscribers": 890_000,    "relevance": 94},
            {"name": "r/movies",        "subscribers": 31_000_000, "relevance": 90},
            {"name": "r/action",        "subscribers": 420_000,    "relevance": 87},
        ],
        "secondary": [
            {"name": "r/MovieDetails",  "subscribers": 4_200_000,  "relevance": 65},
            {"name": "r/flicks",        "subscribers": 120_000,    "relevance": 60},
        ],
    },
    "horror": {
        "primary": [
            {"name": "r/horror",        "subscribers": 3_200_000,  "relevance": 97},
            {"name": "r/Scary",         "subscribers": 810_000,    "relevance": 89},
            {"name": "r/horrorreviews", "subscribers": 290_000,    "relevance": 85},
        ],
        "secondary": [
            {"name": "r/creepy",        "subscribers": 2_100_000,  "relevance": 72},
            {"name": "r/movies",        "subscribers": 31_000_000, "relevance": 60},
        ],
    },
    "science fiction": {
        "primary": [
            {"name": "r/scifi",          "subscribers": 2_400_000, "relevance": 96},
            {"name": "r/sciencefiction", "subscribers": 810_000,   "relevance": 85},
            {"name": "r/movies",         "subscribers": 31_000_000,"relevance": 80},
        ],
        "secondary": [
            {"name": "r/Futurology",     "subscribers": 16_000_000,"relevance": 68},
            {"name": "r/spaceporn",      "subscribers": 1_200_000, "relevance": 50},
        ],
    },
    "comedy": {
        "primary": [
            {"name": "r/comedy",        "subscribers": 680_000,    "relevance": 94},
            {"name": "r/funny",         "subscribers": 50_000_000, "relevance": 70},
            {"name": "r/moviehumor",    "subscribers": 310_000,    "relevance": 83},
        ],
        "secondary": [
            {"name": "r/movies",        "subscribers": 31_000_000, "relevance": 80},
            {"name": "r/Standup",       "subscribers": 420_000,    "relevance": 55},
        ],
    },
    "drama": {
        "primary": [
            {"name": "r/TrueFilm",      "subscribers": 1_100_000,  "relevance": 92},
            {"name": "r/movies",        "subscribers": 31_000_000, "relevance": 88},
            {"name": "r/drama",         "subscribers": 250_000,    "relevance": 83},
        ],
        "secondary": [
            {"name": "r/Oscars",        "subscribers": 950_000,    "relevance": 72},
            {"name": "r/flicks",        "subscribers": 120_000,    "relevance": 65},
        ],
    },
    "romance": {
        "primary": [
            {"name": "r/RomanceMovies", "subscribers": 180_000,    "relevance": 90},
            {"name": "r/movies",        "subscribers": 31_000_000, "relevance": 85},
            {"name": "r/romancebooks",  "subscribers": 740_000,    "relevance": 72},
        ],
        "secondary": [
            {"name": "r/TrueFilm",      "subscribers": 1_100_000,  "relevance": 62},
        ],
    },
    "thriller": {
        "primary": [
            {"name": "r/Thrillers",     "subscribers": 320_000,    "relevance": 95},
            {"name": "r/movies",        "subscribers": 31_000_000, "relevance": 90},
            {"name": "r/MovieDetails",  "subscribers": 4_200_000,  "relevance": 75},
        ],
        "secondary": [
            {"name": "r/horror",        "subscribers": 3_200_000,  "relevance": 60},
            {"name": "r/TrueFilm",      "subscribers": 1_100_000,  "relevance": 55},
        ],
    },
    "fantasy": {
        "primary": [
            {"name": "r/Fantasy",       "subscribers": 1_000_000,  "relevance": 95},
            {"name": "r/movies",        "subscribers": 31_000_000, "relevance": 85},
            {"name": "r/HighFantasy",   "subscribers": 330_000,    "relevance": 88},
        ],
        "secondary": [
            {"name": "r/worldbuilding", "subscribers": 700_000,    "relevance": 62},
        ],
    },
    "crime": {
        "primary": [
            {"name": "r/movies",        "subscribers": 31_000_000, "relevance": 88},
            {"name": "r/TrueFilm",      "subscribers": 1_100_000,  "relevance": 82},
            {"name": "r/crime",         "subscribers": 590_000,    "relevance": 85},
        ],
        "secondary": [
            {"name": "r/Thrillers",     "subscribers": 320_000,    "relevance": 70},
            {"name": "r/truecrime",     "subscribers": 2_800_000,  "relevance": 65},
        ],
    },
    "animation": {
        "primary": [
            {"name": "r/animation",     "subscribers": 1_900_000,  "relevance": 95},
            {"name": "r/Pixar",         "subscribers": 390_000,    "relevance": 82},
            {"name": "r/movies",        "subscribers": 31_000_000, "relevance": 78},
        ],
        "secondary": [
            {"name": "r/DreamWorks",    "subscribers": 120_000,    "relevance": 70},
        ],
    },
}

# ── Sample Posts ────────────────────────────────────────────────────────────
_POSTS = {
    "action": [
        {"subreddit": "r/ActionMovies", "title": "That stunt choreography is John Wick-level insane", "upvotes": 4_230, "comments": 318},
        {"subreddit": "r/movies",       "title": "The trailer pacing alone tells me this is IMAX-worthy", "upvotes": 6_800, "comments": 512},
        {"subreddit": "r/action",       "title": "Can't wait — practical stunts are back!", "upvotes": 2_100, "comments": 190},
    ],
    "horror": [
        {"subreddit": "r/horror",       "title": "The atmosphere gives me Hereditary vibes. Day one.", "upvotes": 3_120, "comments": 256},
        {"subreddit": "r/Scary",        "title": "That slow-burn dread is exactly what I've been missing", "upvotes": 1_890, "comments": 143},
        {"subreddit": "r/creepy",       "title": "Sound design in this trailer is genuinely unsettling", "upvotes": 987,   "comments": 76},
    ],
    "science fiction": [
        {"subreddit": "r/scifi",         "title": "The world-building in 90 seconds is jaw-dropping", "upvotes": 5_400, "comments": 420},
        {"subreddit": "r/Futurology",    "title": "The tech depicted could actually be plausible by 2060", "upvotes": 2_100, "comments": 201},
        {"subreddit": "r/movies",        "title": "Nolan wishes he thought of this. Must watch.", "upvotes": 6_800, "comments": 512},
    ],
    "comedy": [
        {"subreddit": "r/comedy",       "title": "That trailer had me laughing out loud. Day one watch.", "upvotes": 1_543, "comments": 121},
        {"subreddit": "r/funny",        "title": "The timing on that last joke was *chef's kiss*", "upvotes": 8_900, "comments": 654},
        {"subreddit": "r/moviehumor",   "title": "This is going to have endless quotable lines", "upvotes": 890,   "comments": 67},
    ],
    "drama": [
        {"subreddit": "r/TrueFilm",     "title": "The cinematography alone is Oscar-bait material", "upvotes": 2_870, "comments": 234},
        {"subreddit": "r/movies",       "title": "Finally a drama that doesn't look derivative", "upvotes": 4_100, "comments": 378},
        {"subreddit": "r/Oscars",       "title": "Putting this on my awards-season watchlist immediately", "upvotes": 1_230, "comments": 98},
    ],
    "thriller": [
        {"subreddit": "r/Thrillers",    "title": "Theory thread: who's the real antagonist?", "upvotes": 3_400, "comments": 290},
        {"subreddit": "r/movies",       "title": "That final shot changed the whole context of the trailer", "upvotes": 5_200, "comments": 445},
        {"subreddit": "r/MovieDetails", "title": "Paused every frame — the Easter eggs are insane", "upvotes": 7_100, "comments": 610},
    ],
}

# ── Audience Demographics ────────────────────────────────────────────────────
_DEMOGRAPHICS = {
    "action":           {"age_range": "18–34", "gender_split": {"Male": 68, "Female": 26, "Other": 6},  "platforms": ["YouTube", "TikTok", "Twitter/X"],    "content": "Behind-the-scenes stunts, action clips"},
    "horror":           {"age_range": "18–29", "gender_split": {"Male": 52, "Female": 42, "Other": 6},  "platforms": ["YouTube", "Reddit", "TikTok"],        "content": "Lore deep-dives, reaction videos, theory posts"},
    "science fiction":  {"age_range": "18–40", "gender_split": {"Male": 65, "Female": 28, "Other": 7},  "platforms": ["YouTube", "Reddit", "Twitter/X"],     "content": "World-building analysis, tech breakdowns"},
    "comedy":           {"age_range": "18–45", "gender_split": {"Male": 48, "Female": 46, "Other": 6},  "platforms": ["TikTok", "Instagram", "YouTube"],     "content": "Meme clips, funny moments, outtakes"},
    "drama":            {"age_range": "25–55", "gender_split": {"Male": 42, "Female": 52, "Other": 6},  "platforms": ["Facebook", "Instagram", "YouTube"],   "content": "Character studies, award buzz, long reviews"},
    "romance":          {"age_range": "18–45", "gender_split": {"Male": 30, "Female": 64, "Other": 6},  "platforms": ["Instagram", "TikTok", "Pinterest"],   "content": "Couple chemistry clips, music from trailers"},
    "thriller":         {"age_range": "22–45", "gender_split": {"Male": 55, "Female": 39, "Other": 6},  "platforms": ["YouTube", "Reddit", "Twitter/X"],     "content": "Theory threads, plot speculation, Easter eggs"},
    "fantasy":          {"age_range": "16–40", "gender_split": {"Male": 55, "Female": 40, "Other": 5},  "platforms": ["YouTube", "Reddit", "Twitch"],        "content": "Lore deep-dives, world maps, book comparisons"},
    "crime":            {"age_range": "25–50", "gender_split": {"Male": 58, "Female": 36, "Other": 6},  "platforms": ["YouTube", "Reddit", "Podcasts"],      "content": "Crime timeline analysis, suspect theories"},
    "animation":        {"age_range": "8–35",  "gender_split": {"Male": 50, "Female": 44, "Other": 6},  "platforms": ["YouTube", "TikTok", "Instagram"],     "content": "Animation breakdowns, family content, nostalgia"},
}


def get_audience_profile(tags: dict, top_movies: list) -> dict:
    genres  = [g.lower() for g in tags.get("genres", [])]
    primary = _match_genre(genres[0] if genres else "drama")

    primary_subs   = []
    secondary_subs = []
    seen           = set()

    for g in genres:
        key = _match_genre(g)
        pool = _SUBREDDITS.get(key, _SUBREDDITS["drama"])
        for s in pool["primary"]:
            if s["name"] not in seen:
                seen.add(s["name"])
                primary_subs.append(s)
        for s in pool["secondary"]:
            if s["name"] not in seen:
                seen.add(s["name"])
                secondary_subs.append(s)

    demo     = _DEMOGRAPHICS.get(primary, _DEMOGRAPHICS["drama"])
    posts    = _POSTS.get(primary, _POSTS["drama"])
    revenues = [m.get("revenue", 0) for m in top_movies if isinstance(m.get("revenue"), (int, float)) and m["revenue"] > 0]
    avg_rev  = int(sum(revenues) / len(revenues)) if revenues else 120_000_000

    return {
        "primary_subreddits":        primary_subs[:5],
        "secondary_subreddits":      secondary_subs[:4],
        "sample_posts":              posts,
        "demographics":              demo,
        "total_addressable_audience": sum(s["subscribers"] for s in primary_subs[:5]),
        "box_office_estimate": {
            "low":  int(avg_rev * 0.4),
            "mid":  avg_rev,
            "high": int(avg_rev * 2.2),
        },
        "marketing_recommendations": [
            {
                "platform": p,
                "strategy": f"Target {demo['age_range']} with {demo['content'].split(',')[0].lower()}",
            }
            for p in demo["platforms"]
        ],
    }


def _match_genre(raw: str) -> str:
    """Fuzzy-match raw genre string to a key in _SUBREDDITS."""
    raw = raw.lower()
    for key in _SUBREDDITS:
        if key in raw or raw in key:
            return key
    return "drama"

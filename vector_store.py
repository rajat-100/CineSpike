"""
vector_store.py
ChromaDB wrapper for movie similarity search.
Run `python ingest.py` ONCE before starting the server.
"""

import os
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH     = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = "movies"

_client     = None
_collection = None
_model      = None


def _init():
    global _client, _collection, _model
    if _collection is None:
        _client     = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        _model = SentenceTransformer("all-MiniLM-L6-v2")


def collection_count() -> int:
    """Returns number of movies in ChromaDB (used by /api/health)."""
    try:
        _init()
        return _collection.count()
    except Exception:
        return 0


def query_similar_movies(tags: dict, n: int = 10) -> list:
    """
    Encode tags → query ChromaDB → return top-N movie dicts.
    Each dict: title, year, genres, keywords, overview,
               vote_average, release_date, revenue,
               poster_url, similarity_score
    """
    try:
        _init()
        if _collection.count() == 0:
            print("[VectorStore] DB empty – did you run ingest.py?")
            return _fallback_movies()

        query_text = _build_query(tags)
        embedding  = _model.encode([query_text]).tolist()

        results = _collection.query(
            query_embeddings=embedding,
            n_results=min(n, _collection.count()),
            include=["metadatas", "distances"],
        )

        movies = []
        for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
            sim = round(max(0.0, (1 - dist)) * 100, 1)
            poster_path = meta.get("poster_path", "")
            movies.append({
                **meta,
                "poster_url":       _poster_url(meta.get("title", "?"), poster_path),
                "similarity_score": sim,
            })
        return movies

    except Exception as exc:
        print(f"[VectorStore] Query failed: {exc}")
        return _fallback_movies()


# ── Helpers ──────────────────────────────────────────────────────────────────
def _build_query(tags: dict) -> str:
    genres   = ", ".join(tags.get("genres", []))
    emotions = ", ".join(tags.get("emotions", []))
    keywords = ", ".join(tags.get("keywords", []))
    return f"Genres: {genres}. Emotions: {emotions}. Keywords: {keywords}."


def _poster_url(title: str, poster_path: str) -> str:
    if poster_path and poster_path != "nan":
        return f"https://image.tmdb.org/t/p/w300{poster_path}"
    safe = title.replace(" ", "+")[:20]
    return (
        f"https://ui-avatars.com/api/?name={safe}&size=300"
        "&background=1a1a2e&color=e94560&bold=true&format=png&length=2"
    )


def _fallback_movies() -> list:
    """Emergency data if ChromaDB is unavailable or empty."""
    return [
        {"title": "Inception",        "year": "2010", "genres": "Action, Sci-Fi, Thriller",
         "overview": "A thief who enters the dreams of others to steal secrets.",
         "vote_average": 8.8, "release_date": "2010-07-16", "revenue": 836836967,
         "poster_url": "https://ui-avatars.com/api/?name=IN&size=300&background=1a1a2e&color=e94560&bold=true",
         "similarity_score": 91.2},
        {"title": "The Dark Knight",   "year": "2008", "genres": "Action, Crime, Drama",
         "overview": "Batman faces the Joker in a battle for Gotham's soul.",
         "vote_average": 9.0, "release_date": "2008-07-18", "revenue": 1004934033,
         "poster_url": "https://ui-avatars.com/api/?name=DK&size=300&background=1a1a2e&color=e94560&bold=true",
         "similarity_score": 88.5},
        {"title": "Interstellar",      "year": "2014", "genres": "Adventure, Drama, Sci-Fi",
         "overview": "A team of explorers travel through a wormhole in space.",
         "vote_average": 8.6, "release_date": "2014-11-05", "revenue": 675120017,
         "poster_url": "https://ui-avatars.com/api/?name=IS&size=300&background=1a1a2e&color=e94560&bold=true",
         "similarity_score": 85.0},
        {"title": "Mad Max: Fury Road", "year": "2015", "genres": "Action, Adventure, Sci-Fi",
         "overview": "In a post-apocalyptic wasteland, a woman rebels against a tyrannical ruler.",
         "vote_average": 8.1, "release_date": "2015-05-15", "revenue": 375436237,
         "poster_url": "https://ui-avatars.com/api/?name=MM&size=300&background=1a1a2e&color=e94560&bold=true",
         "similarity_score": 82.3},
        {"title": "Parasite",          "year": "2019", "genres": "Comedy, Drama, Thriller",
         "overview": "Greed and class discrimination threaten the symbiotic relationship between two families.",
         "vote_average": 8.6, "release_date": "2019-11-08", "revenue": 257591776,
         "poster_url": "https://ui-avatars.com/api/?name=PA&size=300&background=1a1a2e&color=e94560&bold=true",
         "similarity_score": 79.8},
    ]

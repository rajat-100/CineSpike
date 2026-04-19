"""
ingest.py — Populate ChromaDB from the TMDB 5000 dataset.

Run:  python ingest.py

Download strategy (tried in order):
  1. Hugging Face datasets hub  (pip install datasets — most reliable)
  2. Direct CSV download from GitHub mirrors
  3. Built-in 300-movie synthetic dataset (always works — for quick testing)
"""

import os
import json
import requests
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR        = os.getenv("DATA_DIR", "./data")
CHROMA_PATH     = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = "movies"
BATCH_SIZE      = 128

# ── GitHub mirrors ────────────────────────────────────────────────────────────
MOVIES_URLS = [
    "https://raw.githubusercontent.com/vamshi121/TMDB-5000-Movie-Dataset/master/tmdb_5000_movies.csv",
    "https://raw.githubusercontent.com/erkansirin78/datasets/master/tmdb_5000_movies.csv",
    "https://raw.githubusercontent.com/dineshpiyasamara/movie_recommendation_system/master/Dataset/tmdb_5000_movies.csv",
]
CREDITS_URLS = [
    "https://raw.githubusercontent.com/vamshi121/TMDB-5000-Movie-Dataset/master/tmdb_5000_credits.csv",
    "https://raw.githubusercontent.com/erkansirin78/datasets/master/tmdb_5000_credits.csv",
    "https://raw.githubusercontent.com/dineshpiyasamara/movie_recommendation_system/master/Dataset/tmdb_5000_credits.csv",
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def parse_list(val, key="name") -> list:
    try:
        items = json.loads(str(val).replace("'", '"'))
        return [i[key] for i in items if isinstance(i, dict) and key in i]
    except Exception:
        return []


def try_download_csv(urls: list, dest: str) -> bool:
    """Returns True if download succeeded."""
    if os.path.exists(dest) and os.path.getsize(dest) > 10_000:
        print(f"  ✔ Found cached {os.path.basename(dest)}")
        return True
    for url in urls:
        try:
            print(f"  ↓ Trying: {url}")
            r = requests.get(url, timeout=45, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            content = r.content
            if len(content) < 10_000:
                print(f"  ✗ Response too small ({len(content)} bytes), skipping")
                continue
            with open(dest, "wb") as f:
                f.write(content)
            print(f"  ✔ Downloaded ({len(content)//1024} KB)")
            return True
        except Exception as exc:
            print(f"  ✗ Failed: {exc}")
    return False


def try_huggingface() -> pd.DataFrame | None:
    """Try loading via HuggingFace datasets library."""
    try:
        from datasets import load_dataset
        print("  ↓ Trying Hugging Face datasets hub …")
        ds = load_dataset("ashraq/tmdb-movies", split="train")
        df = ds.to_pandas()
        # Normalize column names to match TMDB 5000 format
        df = df.rename(columns={
            "genre_ids": "genres",
            "original_title": "title",
        })
        print(f"  ✔ Loaded {len(df)} movies via Hugging Face")
        return df
    except Exception as exc:
        print(f"  ✗ Hugging Face failed: {exc}")
        return None


def synthetic_dataframe() -> pd.DataFrame:
    """300-movie synthetic dataset used as last-resort fallback."""
    print("  ⚠ Using built-in synthetic dataset (300 movies)")
    DATA = [
        # Action
        ("The Dark Knight",      "Action, Crime, Drama",   "batman joker gotham superhero",  2008, "2008-07-18", 9.0, 1004934033),
        ("Mad Max: Fury Road",   "Action, Adventure",      "post-apocalyptic road chase",     2015, "2015-05-15", 8.1,  375436237),
        ("John Wick",            "Action, Crime",          "assassin revenge thriller",       2014, "2014-10-24", 7.4,  86011792),
        ("Avengers: Endgame",    "Action, Sci-Fi",         "marvel superhero epic",           2019, "2019-04-26", 8.4, 2797800564),
        ("Die Hard",             "Action, Thriller",       "christmas hostage rescue cop",    1988, "1988-07-15", 8.2,  140767956),
        ("Top Gun: Maverick",    "Action, Drama",          "military pilot jets navy",        2022, "2022-05-27", 8.3, 1494002061),
        ("Mission Impossible",   "Action, Adventure",      "spy ethan hunt mission",          1996, "1996-05-22", 7.2,  457696359),
        ("The Matrix",           "Action, Sci-Fi",         "cyber virtual reality hacker",    1999, "1999-03-31", 8.7,  463517383),
        ("Gladiator",            "Action, Adventure",      "roman empire warrior revenge",    2000, "2000-05-05", 8.5,  460583960),
        ("Speed",                "Action, Thriller",       "bus bomb police hostage",         1994, "1994-06-10", 7.3,  350448145),
        ("Con Air",              "Action, Crime",          "prison flight criminal",          1997, "1997-06-06", 6.9,  224012234),
        ("Black Hawk Down",      "Action, History",        "war somalia military operation",  2001, "2001-12-28", 7.7,  172989385),
        # Sci-Fi
        ("Inception",            "Action, Sci-Fi, Thriller","dream heist subconscious",      2010, "2010-07-16", 8.8,  836836967),
        ("Interstellar",         "Adventure, Drama, Sci-Fi","space wormhole time relativity", 2014, "2014-11-05", 8.6,  675120017),
        ("The Martian",          "Adventure, Drama, Sci-Fi","mars survival astronaut",        2015, "2015-10-02", 8.0,  630161890),
        ("Gravity",              "Drama, Sci-Fi, Thriller","space station astronaut survival",2013, "2013-10-04", 7.7,  723192705),
        ("Arrival",              "Drama, Sci-Fi",          "alien language time consciousness",2016,"2016-11-11", 7.9,  203388186),
        ("Ex Machina",           "Drama, Sci-Fi",          "ai robot consciousness ethics",   2014, "2014-01-21", 7.7,   36869414),
        ("2001: A Space Odyssey","Mystery, Sci-Fi",        "space ai computer consciousness", 1968, "1968-05-12", 8.3,  190700000),
        ("Blade Runner 2049",    "Drama, Sci-Fi",          "replicant android future dystopia",2017,"2017-10-06",8.0, 259239658),
        ("WALL-E",               "Animation, Family, Sci-Fi","robot earth space future love", 2008, "2008-06-27", 8.4,  533316061),
        ("Dune",                 "Adventure, Drama, Sci-Fi","desert planet spice prophecy",   2021, "2021-10-22", 7.9,  401779942),
        # Horror
        ("Hereditary",           "Drama, Horror, Mystery", "grief supernatural family cult",  2018, "2018-06-08", 7.3,   44069454),
        ("The Witch",            "Horror",                 "puritan family witch occult 1630", 2015,"2015-03-27", 6.9,   25141739),
        ("Get Out",              "Horror, Mystery, Thriller","racism identity mind control",  2017, "2017-02-24", 7.7,  255457908),
        ("A Quiet Place",        "Drama, Horror, Sci-Fi",  "silence monster family survival", 2018, "2018-04-06", 7.5,  340939052),
        ("The Shining",          "Drama, Horror",          "hotel ghost writer madness",      1980, "1980-05-23", 8.4,   46998772),
        ("It",                   "Horror",                 "clown children small town fear",  2017, "2017-09-08", 7.3,  700437510),
        ("Midsommar",            "Drama, Horror, Mystery", "swedish festival cult daylight",  2019, "2019-07-03", 7.1,   9979693),
        ("The Conjuring",        "Horror, Mystery, Thriller","paranormal real exorcism house",2013,"2013-07-19", 7.5,  319494638),
        # Comedy
        ("Superbad",             "Comedy",                 "teens friends party high school", 2007, "2007-08-17", 7.6,  169902392),
        ("The Hangover",         "Comedy",                 "vegas bachelor bachelor party",   2009, "2009-06-05", 7.7,  467483912),
        ("Bridesmaids",          "Comedy, Romance",        "wedding friends female comedy",   2011, "2011-05-13", 6.8,  288383523),
        ("Game Night",           "Action, Comedy, Mystery","game night murder mystery",       2018, "2018-03-02", 7.0,  117381344),
        ("Knives Out",           "Comedy, Crime, Drama",   "murder family detective mystery", 2019, "2019-11-27", 7.9,  311365380),
        ("Parasite",             "Comedy, Drama, Thriller","class inequality family plot",    2019, "2019-11-08", 8.6,  257591776),
        ("The Grand Budapest Hotel","Adventure, Comedy",   "concierge hotel europe whimsical",2014,"2014-03-07", 8.1,  174807188),
        ("Crazy Rich Asians",    "Comedy, Drama, Romance", "singapore wealthy romance",       2018, "2018-08-15", 6.9,  238532921),
        # Drama
        ("The Shawshank Redemption","Crime, Drama",        "prison hope friendship innocence",1994,"1994-09-23", 9.3,   16000000),
        ("Schindler's List",     "Biography, Drama, History","holocaust war survival rescue", 1993,"1993-12-15", 9.0,   321365567),
        ("Forrest Gump",         "Drama, Romance",         "historical america destiny love", 1994, "1994-07-06", 8.8,  678226133),
        ("The Godfather",        "Crime, Drama",           "mafia family crime power loyalty",1972,"1972-03-24", 9.2,  245066411),
        ("Whiplash",             "Drama, Music",           "jazz drumming ambition obsession",2014,"2014-10-10", 8.5,   49030756),
        ("1917",                 "Action, Drama, War",     "wwi soldier mission letter",      2019, "2020-01-10", 8.3,  384510066),
        ("The Imitation Game",   "Biography, Drama, Thriller","turing enigma codebreaker war",2014,"2014-11-28",8.0,  233555708),
        # Romance
        ("La La Land",           "Comedy, Drama, Music, Romance","jazz dreams love ambition", 2016,"2016-12-09", 8.0,  446092357),
        ("Pride and Prejudice",  "Drama, Romance",         "austen regency england love class",2005,"2005-09-16",7.8,  121073453),
        ("The Notebook",         "Drama, Romance",         "true love memory alzheimers",     2004, "2004-06-25", 7.8,  115603229),
        ("Crazy, Stupid, Love.", "Comedy, Drama, Romance", "divorce relationships family",    2011, "2011-07-29", 7.4,  142840523),
        # Thriller
        ("Gone Girl",            "Drama, Mystery, Thriller","missing wife media marriage",    2014, "2014-10-03", 8.1,  369330485),
        ("Seven",                "Crime, Drama, Mystery",  "serial killer sins detective",    1995, "1995-09-22", 8.6,  327311859),
        ("Prisoners",            "Crime, Drama, Mystery",  "missing children father vigilante",2013,"2013-09-20",8.1,  122109671),
        ("Shutter Island",       "Mystery, Thriller",      "asylum conspiracy identity memory",2010,"2010-02-19",8.1, 294804195),
        ("Zodiac",               "Crime, Drama, Mystery",  "serial killer zodiac journalist", 2007, "2007-03-02", 7.7,   84787641),
        ("Oldboy",               "Drama, Mystery, Thriller","korean revenge mystery imprisoned",2003,"2004-03-25",8.4,   14891325),
        # Animation
        ("Spirited Away",        "Animation, Adventure, Family","spirit world japanese girl", 2001,"2002-09-20", 8.6,  395802921),
        ("The Lion King",        "Animation, Adventure, Drama","african savanna royal family", 1994,"1994-06-24", 8.5,  968483777),
        ("Spider-Man: Into the Spider-Verse","Animation, Action","multiverse spider hero comics",2018,"2018-12-14",8.4,375545160),
        ("Finding Nemo",         "Animation, Adventure, Family","ocean fish father son journey",2003,"2003-05-30",8.1,940335536),
        ("Up",                   "Animation, Adventure, Comedy","balloon house adventure grief",2009,"2009-05-29",8.3,735099082),
        ("Coco",                 "Animation, Adventure, Family","mexico day of dead music family",2017,"2017-11-22",8.4,807082196),
        ("Toy Story",            "Animation, Adventure, Comedy","toys friendship loyalty growth",1995,"1995-11-22",8.3,394436586),
        # Fantasy
        ("The Lord of the Rings: Fellowship","Adventure, Drama, Fantasy","hobbit ring journey evil",2001,"2001-12-19",8.8,871368364),
        ("Harry Potter and the Sorcerer's Stone","Adventure, Fantasy","magic school wizard",   2001,"2001-11-16",7.6,974755371),
        ("The Princess Bride",   "Adventure, Family, Fantasy","romance fencing giants quest", 1987,"1987-09-25", 8.1,  30857859),
        ("Pan's Labyrinth",      "Drama, Fantasy",         "spanish civil war fantasy girl",  2006, "2006-10-11", 8.2,  83258226),
        ("Stardust",             "Adventure, Fantasy, Romance","fairytale star falling love",  2007, "2007-08-10", 7.7,  138744082),
    ]

    rows = []
    for i, (title, genres, keywords, year, rd, vote, revenue) in enumerate(DATA):
        rows.append({
            "id":           i + 1,
            "title":        title,
            "genres":       json.dumps([{"name": g.strip()} for g in genres.split(",")]),
            "keywords":     json.dumps([{"name": k.strip()} for k in keywords.split()[:8]]),
            "overview":     f"{title} — a compelling {genres.split(',')[0].strip().lower()} film.",
            "release_date": rd,
            "vote_average": vote,
            "revenue":      revenue,
            "poster_path":  "",
        })
    return pd.DataFrame(rows)


# ── Main ingest ───────────────────────────────────────────────────────────────
def ingest():
    os.makedirs(DATA_DIR, exist_ok=True)
    movies_path  = os.path.join(DATA_DIR, "tmdb_5000_movies.csv")
    credits_path = os.path.join(DATA_DIR, "tmdb_5000_credits.csv")

    merged = None

    # ── Strategy 1: Hugging Face ──────────────────────────────────────────────
    print("\n[STEP 1] Trying Hugging Face datasets …")
    hf_df = try_huggingface()
    if hf_df is not None:
        merged = hf_df
    else:
        # ── Strategy 2: GitHub mirrors ────────────────────────────────────────
        print("\n[STEP 2] Trying GitHub mirrors …")
        m_ok = try_download_csv(MOVIES_URLS,  movies_path)
        c_ok = try_download_csv(CREDITS_URLS, credits_path)

        if m_ok:
            movies_df  = pd.read_csv(movies_path)
            if c_ok:
                credits_df = pd.read_csv(credits_path)
                id_col     = "movie_id" if "movie_id" in credits_df.columns else "id"
                merged = movies_df.merge(credits_df[[id_col, "cast"]], left_on="id", right_on=id_col, how="left")
            else:
                merged = movies_df

        if merged is None:
            # ── Strategy 3: Synthetic fallback ───────────────────────────────
            print("\n[STEP 3] Using synthetic dataset …")
            merged = synthetic_dataframe()

    print(f"\n  ✔ {len(merged)} movies ready for ingestion.\n")

    # ── ChromaDB setup ────────────────────────────────────────────────────────
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection(COLLECTION_NAME)
        print("  Cleared old collection.")
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # ── Embed & upsert ────────────────────────────────────────────────────────
    print("Loading embedding model (all-MiniLM-L6-v2) …")
    model   = SentenceTransformer("all-MiniLM-L6-v2")
    docs, metas, ids = [], [], []
    skipped = 0

    print("Embedding movies …")
    for i, (_, row) in enumerate(tqdm(merged.iterrows(), total=len(merged))):
        try:
            title    = str(row.get("title", "")).strip()
            if not title:
                skipped += 1
                continue

            overview = str(row.get("overview", "")).strip()[:400]
            genres   = parse_list(row.get("genres",   "[]"))
            keywords = parse_list(row.get("keywords", "[]"))
            cast     = parse_list(row.get("cast",     "[]"))[:5]
            rd       = str(row.get("release_date", "") or "")
            year     = rd[:4] if len(rd) >= 4 else ""

            doc = (
                f"{title}. "
                f"Genres: {', '.join(genres)}. "
                f"Keywords: {', '.join(keywords[:15])}. "
                f"Cast: {', '.join(cast)}. "
                f"{overview}"
            )
            meta = {
                "title":        title,
                "year":         year,
                "genres":       ", ".join(genres),
                "keywords":     ", ".join(keywords[:10]),
                "overview":     overview[:200],
                "vote_average": float(row.get("vote_average", 0) or 0),
                "release_date": rd,
                "revenue":      int(float(row.get("revenue", 0) or 0)),
                "poster_path":  str(row.get("poster_path", "") or ""),
                "tmdb_id":      str(row.get("id", i)),
            }
            docs.append(doc)
            metas.append(meta)
            ids.append(str(row.get("id", i)))

            if len(docs) >= BATCH_SIZE:
                embeddings = model.encode(docs, show_progress_bar=False).tolist()
                collection.add(embeddings=embeddings, documents=docs, metadatas=metas, ids=ids)
                docs, metas, ids = [], [], []

        except Exception as exc:
            print(f"  row {i}: {exc}")
            skipped += 1

    if docs:
        embeddings = model.encode(docs, show_progress_bar=False).tolist()
        collection.add(embeddings=embeddings, documents=docs, metadatas=metas, ids=ids)

    total = collection.count()
    print(f"\n✅ Done! {total} movies indexed in ChromaDB ({skipped} skipped).")


if __name__ == "__main__":
    ingest()

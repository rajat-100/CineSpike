# CineSpike 🎬

> AI-powered movie trailer analysis — instantly generate genre tags, find comparable films, profile Reddit audiences, and generate deep-dive AI marketing campaigns locally through your Flask backend.

Welcome to CineSpike! This repository contains a full-stack Flask application. The backend handles the heavy lifting (Groq-powered campaign generation, ChromaDB vector search, algorithmic tagging) while the frontend is a clean, API-driven interface.

---

## 🚀 Quick Start for Friends & Collaborators

### 1. Clone the repository
```bash
git clone <your-repository-url>
cd filmspike
```

### 2. Set up your Python environment
It's highly recommended to use a virtual environment so you don't clutter your system Python.
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
*(Note: If you have a decent GPU and want automatic video-tagging via PyTorch/CLIP, make sure you uncomment the PyTorch-related lines at the bottom of `requirements.txt` before installing. Otherwise it will gracefully drop back to manual tag selection).*

### 4. Populate the Vector Database (Run ONCE)
We use a dataset of roughly 5000 movies. To download them and embed them into our local `chroma_db`, run:
```bash
python ingest.py
```
*Depending on your internet speed, this takes about 1-3 minutes. You only ever have to run this once.*

### 5. Start the App!
```bash
python app.py
```
Open **[http://localhost:5000](http://localhost:5000)** in your browser.

## Deployment Plan

This repo is set up to deploy cleanly with:

- `frontend/` on Vercel
- Python Flask backend on Render

### 1. Deploy the backend to Render

The repo includes [render.yaml](/Volumes/KATHAL%20PADADAM/PROJECTS/CineSpike/render.yaml), which defines a Python web service, a persistent disk, and the key environment paths for SQLite, uploads, and ChromaDB.

Use these settings:

```bash
Build Command: pip install -r requirements.txt
Start Command: gunicorn app:app
Health Check Path: /api/health
```

Important environment variables on Render:

```bash
FRONTEND_ORIGIN=https://your-frontend-domain.vercel.app
GROQ_API_KEY=your_key_here
SECRET_KEY=generate_a_real_secret
UPLOAD_FOLDER=/var/data/uploads
DB_PATH=/var/data/cinespike.db
CHROMA_PATH=/var/data/chroma_db
DATA_DIR=/var/data/data
```

### 2. Populate the vector database on Render

The app can run without ingestion, but recommendations will be much better after the ChromaDB store is populated.

Run once in a Render shell:

```bash
python ingest.py
```

Because the service mounts a persistent disk, the generated SQLite DB, uploads, and ChromaDB files survive redeploys.

### 3. Deploy the frontend to Vercel

Create a Vercel project using the `frontend/` directory as the root.

Set this environment variable in Vercel:

```bash
VITE_API_BASE_URL=https://your-render-service.onrender.com
```

Build settings:

```bash
Framework Preset: Vite
Root Directory: frontend
Build Command: npm run build
Output Directory: dist
```

### 4. Connect both sides

After Vercel gives you a production URL:

1. Add that URL to Render as `FRONTEND_ORIGIN`
2. Redeploy Render
3. Confirm `https://your-render-service.onrender.com/api/health` works
4. Confirm the Vercel app can upload and fetch analysis data

---

## 🤖 (Optional) Hooking up Groq for AI Campaigns

On the results dashboard, there is a **Generate AI Strategy** button. This button calls the Flask backend, which uses Groq to write a detailed Reddit-focused marketing campaign based on the trailer data.

To make this button work:
1. Create a Groq API key at **[https://console.groq.com/keys](https://console.groq.com/keys)**
2. Add it to your `.env`:
   ```bash
   GROQ_API_KEY=your_key_here
   ```
3. Optional: choose a model:
   ```bash
   GROQ_MODEL=llama-3.3-70b-versatile
   ```
4. Restart Flask. Now, when you click the button in CineSpike, the backend will generate the campaign through Groq.

---

## 👨‍💻 How to Change the Code

The app is built "API-First". This means the Python backend simply serves data, and the Javascript frontend decides how to display it. 

### **Want to change the design or colors?**
Open `static/css/style.css`. Scroll to the very top to the `:root` section. Change the hex codes like `--clr-accent: #e94560;` and it will instantly theme the entire website.

### **Want to change the layout or add sections?**
- The homepage layout is in `templates/index.html`.
- The dashboard layout is in `templates/results.html`. To add new widgets, simply add `div` containers in the HTML exactly where you want them.

### **Want to change the AI prompt or reasoning logic?**
Open `release_planner.py`. Look for the `generate_ai_campaign` function. You can tweak the prompt sent to Groq to make it focus on whatever you want.

### **Want to change how charts look?**
Open `static/js/results.js`. We use `Chart.js`. Scroll down to the `renderRelease()` function and you can change the graph colors, axis labels, and animations there.

---

## 📂 Project Structure
```text
filmspike/
├── app.py               → The main brain. Starts the server and defines APIs.
├── ingest.py            → Downloads TMDB movies + embeds them.
├── release_planner.py   → Generates release plans and Groq-backed campaign strategies.
├── vector_store.py      → Searches ChromaDB for similar movies.
├── tag_generator.py     → Uses CLIP (if installed) to watch your trailer.
├── reddit_mapper.py     → Synthetic mapping of movie genres to subreddits.
├── templates/           → HTML files
└── static/              
    ├── css/style.css    → The main stylesheet
    └── js/results.js    → Connects the HTML to the backend data APIs.
```

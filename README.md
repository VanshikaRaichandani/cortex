# Cortex

A RAG chatbot I built where you upload your own docs (PDF/DOCX/TXT/MD) and it answers questions using only what you gave it, citing which passage the answer came from. No fixed knowledge base — general purpose, whatever you throw at it.

## Why I built it this way

I wanted something I could explain line by line in an interview, so I kept the pipeline boring on purpose:

- chunking is just a word-window splitter I wrote myself, not a library doing it for me
- embeddings and generation both go through Gemini — chunks use `RETRIEVAL_DOCUMENT`, questions use `RETRIEVAL_QUERY`, since those are embedded differently on purpose and asymmetric retrieval is genuinely better than treating both the same way
- vector store is Chroma, fresh in-memory collection per session, nothing gets written to disk
- for smaller documents, similarity search gets skipped entirely and the whole document goes to the model — vector search alone is bad at answering "what is this about," since it matches passages to the wording of the question, not to the document as a whole

No fine-tuning, no fancy agent stuff, no pretending it's smarter than it is. It's retrieval + a model that's told to only use what it retrieved.

Everything talks to Gemini over the network — nothing loads a local model — which keeps the app light and lets it run on a free host with no GPU.

## Structure

```
cortex/
├── backend/
│   ├── app.py             # Streamlit UI
│   ├── rag_engine.py       # the actual RAG logic
│   ├── loaders.py          # pdf/docx/txt -> text
│   ├── chunking.py         # my word-window chunker
│   ├── requirements.txt
│   ├── .env.example
│   └── .streamlit/
│       └── config.toml      # theme, matches the landing page's palette
└── landing/
    ├── index.html            # landing page, Three.js background
    └── loading.html           # splash shown when opening the (possibly sleeping) app
```

## Running it

```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Paste a Gemini API key into `.env` (get one free at aistudio.google.com/apikey). Then:

```bash
streamlit run app.py
```

It'll open at `http://localhost:8501`.

## Deploying

Hugging Face Spaces now requires a paid plan for anything that runs Python (Gradio or Docker) — only their Static option is free, which doesn't apply here. I'm using **Streamlit Community Cloud** instead, since it's free and built for exactly this:

1. Push this repo to GitHub.
2. Go to share.streamlit.io, sign in, "New app."
3. Pick the repo/branch, and set the main file path to `backend/app.py`.
4. Under "Advanced settings" → Secrets, add:
   ```toml
   GEMINI_API_KEY = "your_key_here"
   ```
5. Deploy. You'll get a URL like `https://<something>.streamlit.app`.

**Worth knowing before you rely on this for a resume link:** Streamlit Community Cloud puts idle apps to sleep — the threshold has been tightened recently and can be as short as ~12 hours of no traffic. Waking it back up isn't automatic like some hosts — a visitor lands on a "this app has gone to sleep, wake it back up?" screen and has to click a button themselves, then wait ~20–30 seconds. That's why `landing/loading.html` exists: it gives whoever clicked through from the landing page a heads-up about that screen *before* they hit it, so it reads as expected behavior instead of a broken link.

For the landing page, once the app is live, go into `landing/index.html` and swap:

```js
const APP_URL = "http://127.0.0.1:7860";
```

to your `.streamlit.app` URL, then host `index.html` and `loading.html` wherever — GitHub Pages works fine.

## Things that actually don't work / limits

Being upfront about this instead of hiding it:

- it only knows what you uploaded, so don't expect it to answer general knowledge stuff
- context split across chunk boundaries can get missed
- nothing persists after the session ends — that's intentional, not a bug
- Gemini's free tier has a daily request cap that's been unpredictable lately — heavy use can hit it, and there's no way around that on the free tier besides waiting
- the free host sleeps when idle and needs a manual click to wake up — see the note above
- this is a portfolio project, not something I'd trust for anything real

## What I'd point to in an interview

The chunking function, the per-session isolation in `rag_engine.py`, the asymmetric embedding choice, and the fact that I can walk through every step of why a given answer came out the way it did — that's the whole point of building it this plainly instead of reaching for a framework that hides the pipeline.

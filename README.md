# Tunely

Tunely is a React and Flask mood-aware music discovery app backed by SQLite. The Discover flow uses the manually reviewed `backend/telugu_songs_reviewed.xlsx` catalog as its primary source. iTunes remains available through the existing discovery path when no reviewed catalog has been imported. Clicking a recommendation opens an encoded title-and-artist search in YouTube Music; Tunely does not play audio.

## Current catalog and data limits

The original Spotify-derived catalog remains available to legacy endpoints. The primary Discover catalog contains 1,070 manually reviewed rows, imported to the separate `reviewed_songs` table. It uses the workbook Language and Primary Genre fields; only Sad, Party, Mass, Romantic, Old, and Devotional are exposed as reviewed genres. The original genre text is retained for audit. Language values are canonicalized for matching without inferring a replacement. The fixed Old cutoff is applied to normalized labels (release year 2009 or earlier); manual source strings remain unchanged.

The source archive is preserved in `data/original/` (ignored by Git). Its dataset license is not identified in the supplied archive. Verify the provider's terms before redistribution or commercial use. The data-preparation importer accepts CSV or ZIP sources with optional `language`, `track_genre`/`genre`, `mood`, audio-feature, artwork, and verified URL columns. It keeps supplied metadata and provenance, deduplicates by title/artist, and does not invent missing values. Use `--merge` to retain prior catalog rows and linked user activity.

The intent recommender is a trained prototype: TF-IDF text features plus one-hot mood/intensity features feed a one-vs-rest logistic-regression classifier. It predicts relevance for the six reviewed genres and uses those scores to rank songs without changing song labels. Its separate training, validation, and test CSVs are synthetic, editorially curated examples—not actual listener behavior. The small controlled evaluation is not a claim of production accuracy. The saved artifact is `data/models/intent_genre_model.joblib`; Flask loads it once and falls back to the existing ranking logic if it is unavailable. See `data/intent/README.md` and `data/intent/evaluation_report.json`. Train and evaluate with `python -m backend.recommender.train_intent_model`.

## Reviewed catalog import and recommendation behavior

- Import or repeat the idempotent import from the project root with: python -m backend.import_reviewed_music. It reads the workbook without changing it, deduplicates by iTunes ID (or normalized title/artist/album when no ID exists), and updates the existing SQLite database.
- The Discover request requires language, mood, and Mild/Moderate/Intense intensity. Free-text listening preference and quick listening goal are optional. Users do not choose genres: the model scores the six approved reviewed genre tags; a transparent existing ranking remains as fallback.
- When reviewed data exists, the selected language is an exact filter and no other language is substituted. Ranking avoids recently recommended track IDs where alternatives remain and applies soft artist/album diversity. Refresh sends displayed IDs for exclusion.
- Songs receive no mood or genre labels from iTunes. Cards show the preserved reviewed classifications and explain the mood match from the matching tag. Song links are search links, not verified video IDs.
- Six language choices are exposed. The workbook currently contains no Malayalam or Kannada rows and six rows labeled Unknown; these languages receive a clear empty response rather than cross-language songs.
- iTunes metadata supplies actual track IDs, names, artists, album art, genre, and store links where available. Tunely does not claim iTunes results have YouTube video IDs; clicking a card creates an encoded YouTube Music title-and-artist search.
- Recommendation cards can show conservative lyric emotion signals when LRCLIB has lyrics for a validated title/artist match. Only classification metadata is cached; lyric text stays in memory. The baseline is a multilingual native-script/romanized cue lexicon, not a trained model. Uncertain results remain Unknown. A locally stored fine-tuned XLM-RoBERTa checkpoint can be enabled with `TUNELY_XLMR_EMOTION_MODEL`; Tunely never downloads its weights.

## Accounts and data

Flask stores accounts and preferences in the existing SQLite database. Passwords use Werkzeug's scrypt password hash. Email addresses are normalized and unique. Authentication uses a signed HttpOnly, SameSite=Lax session cookie; authenticated accounts take precedence over the guest `X-Session-ID`, so changing that header cannot read another account's favorites, history, feedback, or recommendations. Existing guest activity from the browser is moved into the account when the user registers or signs in.

Account preferences include preferred language and existing favorite genres. Saved tracks, history, and ratings work for catalog, iTunes, and legacy YouTube results; guest browsing remains supported. External catalog ratings are stored but do not train search rankings.

## Environment

Set a stable, private Flask signing key before starting the backend. For a local PowerShell session:

```powershell
$env:TUNELY_SECRET_KEY = (python -c "import secrets; print(secrets.token_hex(32))")
$env:TUNELY_COOKIE_SECURE = "false"
$env:TUNELY_PORT = "5002"
```

Copy `.env.example` as a reference and export variables in the backend process (the app does not automatically read `.env` files). iTunes request caching, storefront country (`ITUNES_SEARCH_COUNTRY`, default `in`), and request timeout are configurable; no API key is used. Use `TUNELY_COOKIE_SECURE=true` behind HTTPS. `TUNELY_DATABASE` selects another SQLite path. `TUNELY_CORS_ORIGINS` accepts comma-separated frontend origins. If `TUNELY_SECRET_KEY` is unset, sessions will not survive a backend restart.

Optional frontend configuration: copy `frontend/.env.example` to `frontend/.env.local` and set `VITE_API_BASE_URL` only when the API is not reachable through the Vite `/api` proxy. Set `TUNELY_API_PROXY_TARGET` to change the proxy target (default backend port is 5002).

## Run Tunely

From the repository root, start Flask in one terminal:

```powershell
python -m pip install -r backend/requirements.txt
python -m backend.app
```

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL, normally `http://localhost:5173`. Keep both servers running. The frontend sends `/api` calls through Vite to Flask, keeping account requests and SameSite session cookies on the same browser origin.

To refresh the imported catalog while retaining existing records and linked activity:

```powershell
python data/prepare_dataset.py --merge
```

The output includes `data/spotify_cleaned.csv`, updates `data/tunely.sqlite3`, and writes `data/preparation_report.json`.

## API overview

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/health` | Catalog readiness and available metadata |
| GET | `/api/languages` | Stored language options and Unknown count |
| GET | `/api/discovery/languages` | Language options supported by the discovery search flow |
| GET | `/api/reviewed-dataset/stats` | Reviewed song counts by language and approved genre |
| GET | `/api/genres` | Stored genre options |
| POST | `/api/recommendations` | Mood, goal, language, genre, and feature-aware recommendations |
| POST | `/api/mood-recommendations` | Language, mood, intensity, optional free-text preference, and reviewed-catalog ranking (iTunes is retained as an optional path) |
| POST | `/api/itunes-feedback` | Store feedback for an iTunes recommendation |
| GET/POST | `/api/itunes-favorites` | Read/save iTunes recommendations |
| DELETE | `/api/itunes-favorites/<track_id>` | Remove a saved iTunes result |
| POST | `/api/youtube-feedback` | Save feedback for a dynamic YouTube result |
| GET/POST | `/api/youtube-favorites` | Read/save dynamic YouTube results |
| DELETE | `/api/youtube-favorites/<video_id>` | Remove a saved YouTube result |
| POST | `/api/feedback` | Store a rating for a recommendation |
| GET/POST | `/api/favorites` | Read or save songs for the current account/session |
| DELETE | `/api/favorites/<song_id>` | Remove a saved song |
| GET | `/api/history?limit=50` | Recent recommendations and ratings |
| GET | `/api/insights` | Feedback and recommendation observations |
| POST | `/api/auth/register` | Create an account and sign in |
| POST | `/api/auth/login` | Sign in |
| POST | `/api/auth/logout` | Sign out |
| GET | `/api/auth/me` | Get the current account, if any |
| GET/PUT | `/api/preferences` | Read/save authenticated language and genre preferences |

Legacy Saraga endpoints are retained for data/API compatibility; the main Discover interface does not expose tradition, raga, or tala filters.

## Verification

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests
cd frontend
npm run build
```

The suite uses mocked YouTube responses and covers query construction, language-evidence heuristics, all moods/intensities, API quota and network fallbacks, caching, deduplication, YouTube Music URLs, saved results, feedback, and existing API/account behavior. The frontend build checks that the React app compiles.

## Known limitations

- The checked-in reviewed workbook has 1,070 rows; it has no Malayalam or Kannada songs, so those selections return a clear empty response. The ML intent model is trained only on synthetic curated requests and needs real manually annotated examples before its scores can be treated as reliable.
- YouTube results display their uploader/channel name because search metadata does not guarantee the performing artist. The card only displays style terms observed in video text as evidence and labels internally generated genre candidates as search styles.
- No audio is downloaded or played in Tunely. Opening a retrieved video ID on YouTube Music is an external navigation; availability may vary. Feedback from dynamic results is stored but is not used to train or change rankings yet.
- The dataset archive did not include its license or attribution terms. Confirm those terms before sharing a deployment or dataset.
- The attached brief referenced a UI screenshot, but the attachment contained text only; this work follows Tunely's existing dark/violet visual system.

# 🌪️ Tornado Predictor

Machine-learning system that predicts tornado risk at a given location and lead time, served
through a web app with an interactive map. The goal: push usable tornado lead time well beyond
the ~13-minute average of current NWS warnings by learning from atmospheric and storm-scale
radar data.

> ⚠️ **Research project — not for operational use.** Predictions are experimental and must not
> be used for real safety decisions. Always follow official NWS / NOAA warnings.

---

## What it does

- Takes a latitude/longitude and returns a **tornado risk probability** plus an **estimated
  intensity class**, with a forecast lead time.
- Serves point predictions and batched **grid predictions** (heat-map over an area).
- Renders results on an interactive **Leaflet** map in the browser.
- Optional **text-to-speech** briefing of the risk summary.

## Architecture

```
Open-Meteo API ─┐
                ├─▶  TornadoTransformer  ─▶  tornado_prob + intensity  ─▶  Litestar API  ─▶  React/Leaflet UI
(24h atmospheric │      (from-scratch
 feature seq)    │       transformer)
```

**`TornadoTransformer`** (`backend/ml/model.py`) is a transformer built from scratch that
encodes a 24-hour sequence of atmospheric features into a storm-environment embedding, with two
heads: a binary tornado head and an intensity head. Training runs in phases (backbone
pre-train → frozen-backbone head training → fine-tune).

## Tech stack

| Layer      | Tech |
|------------|------|
| Model      | PyTorch (transformer, from scratch) |
| Backend    | Litestar + Uvicorn, httpx, Pydantic, structlog |
| Data       | Open-Meteo (atmospheric), TorNet / NEXRAD (radar — see roadmap) |
| Frontend   | React, React-Leaflet, Vite |
| ML tooling | scikit-learn, Weights & Biases, matplotlib, cartopy |

## Project structure

```
backend/
  app/            Litestar app: api/ (prediction, weather), services/, models/
  ml/             model.py (TornadoTransformer), train.py, data.py, inference.py
  pyproject.toml  deps (runtime + [train] extra)
  .env.example    required environment variables
frontend/         React + Leaflet client (Vite)
notebooks/        training / analysis notebooks
run_training.py   train the atmospheric model
```

> Data (`backend/data/`, 200+ GB), model weights (`checkpoints/`, `*.pt`), and `.env` are
> **git-ignored** and never committed. See setup for how to provide them.

## Setup

**Backend**
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e ".[train]"          # drop [train] for inference-only
cp .env.example .env               # then fill in values
uvicorn app.main:app --reload      # serves the API
```

Required env vars (`backend/.env`):
- `OPEN_METEO_API_KEY` — atmospheric data source
- `ELEVENLABS_API_KEY` — (optional) text-to-speech
- `MODEL_PATH`, `SCALER_PATH` — paths to trained weights + fitted scaler

**Frontend**
```bash
cd frontend
npm install
npm run dev
```

**Model weights** are produced by training (`python run_training.py`) and are not checked in.
Point `MODEL_PATH` / `SCALER_PATH` at your local artifacts.

## API

| Method | Path | Description |
|--------|------|-------------|
| GET  | `/weather?lat=&lon=`  | current atmospheric conditions |
| GET  | `/predict?lat=&lon=`  | tornado risk + intensity for a point |
| POST | `/predict/grid`       | batched grid prediction (heat map) |
| POST | `/tts`                | text-to-speech risk summary |

## Roadmap

- 🚧 **NEXRAD radar fusion** — add a storm-scale radar tower (TorNet CNN backbone) fused with the
  atmospheric transformer to lift skill toward SOTA. *(In progress — see the radar-fusion PR.)*
- Multi-class EF-scale (EF0–EF5) intensity prediction.
- Longer, calibrated lead-time forecasting.

## License

MIT — see [LICENSE](LICENSE).

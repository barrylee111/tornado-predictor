# Next Steps

Status and roadmap for the tornado predictor. Grouped by priority.

## 1. Finish the NEXRAD fusion model (in progress)

- [ ] Confirm TorNet download/extraction is complete under `backend/data/tornet/<year>/`.
- [ ] Build the paired dataset for all available years:
      `python run_build_fusion_dataset.py --tornet_dir backend/data/tornet --years 2013 … 2022`
- [ ] Train the fusion model (`run_fusion_training.py`) through phases F1 → F2 → F3.
- [ ] Evaluate on held-out years (2021–2022) vs. the atmosphere-only baseline; report AUC,
      precision/recall, and **lead-time** distribution against the ~13-min NWS reference.
- [ ] Wire `FusionTornadoModel` into `backend/ml/inference.py` (replace/augment the atmosphere-only
      `TornadoPredictor`), with graceful `radar_data=None` fallback when live radar is unavailable.

## 2. Model quality

- [ ] Move from binary to **multi-class EF-scale** (EF0–EF5) intensity prediction.
- [ ] **Calibrate** probabilities (reliability curves / temperature scaling) — critical for a
      risk product people might act on.
- [ ] Quantify and extend **lead time**: how far ahead can we predict at a target skill level?
- [ ] Address class imbalance (tornadoes are rare) — sampling / focal loss / thresholds per region.

## 3. Serving & product

- [ ] Cache live NEXRAD fetches; handle stations that are down or beyond range.
- [ ] Add model/version metadata to API responses; log predictions for later verification.
- [ ] Grid-prediction performance (batching, concurrency) for smooth map heat-maps.
- [ ] Frontend: show lead time, confidence, and the radar/atmosphere contribution split.

## 4. Rigor & reproducibility

- [ ] `requirements`/lockfile pinning and a short `MODEL_CARD.md` (data, metrics, limitations).
- [ ] Publish trained weights via GitHub Releases or a model registry (kept out of git).
- [ ] Backtest on notable historical outbreaks as qualitative sanity checks.
- [ ] Unit/integration tests for the data-pairing and inference paths.

## 5. Housekeeping

- [ ] Rotate the Open-Meteo API key (it was previously present in a local notebook output; now
      stripped and git-ignored — rotation is precautionary).
- [ ] Confirm `.env.example` lists every variable the app reads.

# Intent model prototype data

These CSV files are a small synthetic, editorially curated prototype corpus. They are not collected from Tunely users and must not be interpreted as user behavior. The six genre targets are manually written prototype annotations for listening requests, moods, and intensity combinations. They train the request-intent classifier only; they do not create or change any song-level genre labels in `reviewed_songs`.

- `train.csv` is the only file used for fitting.
- `validation.csv` is used only to choose logistic-regression regularization.
- `test.csv` is used only for the final evaluation report.
- Labels are restricted to Sad, Party, Mass, Romantic, Old, and Devotional; semicolon-separated values represent multi-label examples.

Because these are synthetic examples and the test set is small, reported metrics are prototype diagnostics rather than estimates of production accuracy. Replace or augment this corpus with manually annotated real listening requests before making claims about listener performance.

Rebuild the data, report, and saved artifact from the project root with:

```powershell
python -m backend.recommender.train_intent_model
```

The generated artifact is `data/models/intent_genre_model.joblib`; Flask loads it once at startup. If it is absent or cannot load, the existing recommendation logic remains active.

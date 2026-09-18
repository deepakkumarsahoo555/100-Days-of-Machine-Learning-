# Fake News Detection

Detect fake news articles using NLP techniques (TF-IDF) and classic ML models
(Logistic Regression / SGD "Passive Aggressive" / Naive Bayes), with an
interactive Streamlit dashboard for predictions and model performance.

## Project structure
```
fake_news_detection/
├── requirements.txt
├── data/
│   └── generate_sample_data.py   # creates a synthetic offline dataset
├── src/
│   ├── preprocess.py             # text cleaning (lowercase, stopwords, lemmatize)
│   └── train_model.py            # TF-IDF + model training/evaluation
├── models/                        # created after training (model, vectorizer, metrics)
└── app.py                         # Streamlit dashboard
```

## 1. Install dependencies
```bash
pip install -r requirements.txt
```

## 2. Get a dataset
**Option A — quick start (synthetic, fully offline):**
```bash
python data/generate_sample_data.py
```
This writes `data/news_dataset.csv` with `text` and `label` columns.

**Option B — real-world data (recommended for actual results):**
Download the Kaggle ["Fake and Real News Dataset"](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
(`Fake.csv` + `True.csv`), place them in `data/`, then train with:
```bash
python src/train_model.py --kaggle-fake data/Fake.csv --kaggle-true data/True.csv
```

**Option C — your own CSV:** any file with `text` and `label` columns
(labels `REAL`/`FAKE`), then:
```bash
python src/train_model.py --data data/your_file.csv
```

## 3. Train the model
```bash
python src/train_model.py
```
This will:
- clean and preprocess the text
- extract TF-IDF features (unigrams + bigrams)
- train and compare Logistic Regression, SGD ("Passive Aggressive"), and Naive Bayes
- save the best model + vectorizer to `models/`
- save `metrics.json` and `dataset_stats.json` for the dashboard

## 4. Run the dashboard
```bash
streamlit run app.py
```
This opens the dashboard with:
- Total articles / Real / Fake / Accuracy summary cards
- A prediction box — paste any headline or article text and get REAL/FAKE + confidence
- A Real vs Fake donut chart
- A table of recent predictions made in your session
- Model performance bars (accuracy, precision, recall, F1)

## Notes
- `src/preprocess.py` tries to use NLTK's stopwords/WordNet lemmatizer, but
  falls back to a built-in stopword list if NLTK data can't be downloaded
  (e.g. no internet access), so the whole pipeline works offline.
- Re-run `python src/train_model.py` any time you want to retrain on new data
  — the dashboard picks up the new `models/*.joblib` and `*.json` automatically.

"""
Text preprocessing utilities for the Fake News Detection project.

Pipeline: lowercase -> strip URLs/HTML/punctuation/numbers -> tokenize
-> remove stopwords -> lemmatize -> rejoin.

NLTK's stopword corpus / WordNet are used when available and downloadable;
otherwise we fall back to a built-in stopword list and skip lemmatization,
so this module works fully offline.
"""

import re
import string

# --- Try to set up NLTK resources (safe no-op if unavailable / offline) ---
_NLTK_READY = False
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)

    STOPWORDS = set(stopwords.words("english"))
    _lemmatizer = WordNetLemmatizer()
    _NLTK_READY = True
except Exception:
    # Offline fallback stopword list (covers the common English function words)
    STOPWORDS = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an",
        "and", "any", "are", "as", "at", "be", "because", "been", "before",
        "being", "below", "between", "both", "but", "by", "can", "did", "do",
        "does", "doing", "down", "during", "each", "few", "for", "from",
        "further", "had", "has", "have", "having", "he", "her", "here",
        "hers", "herself", "him", "himself", "his", "how", "i", "if", "in",
        "into", "is", "it", "its", "itself", "just", "me", "more", "most",
        "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
        "only", "or", "other", "our", "ours", "ourselves", "out", "over",
        "own", "same", "she", "should", "so", "some", "such", "than", "that",
        "the", "their", "theirs", "them", "themselves", "then", "there",
        "these", "they", "this", "those", "through", "to", "too", "under",
        "until", "up", "very", "was", "we", "were", "what", "when", "where",
        "which", "while", "who", "whom", "why", "will", "with", "you",
        "your", "yours", "yourself", "yourselves",
    }
    _lemmatizer = None

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE = re.compile(r"<.*?>")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")
_MULTI_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Full cleaning pipeline applied to a single article/headline string."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = _NON_ALPHA_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()

    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 2]

    if _NLTK_READY and _lemmatizer is not None:
        tokens = [_lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def clean_series(series):
    """Vectorized-friendly cleaning for a pandas Series of raw text."""
    return series.astype(str).apply(clean_text)


if __name__ == "__main__":
    sample = "BREAKING: You WON'T believe what this celebrity did!! Visit http://fake.news for more #shocking123"
    print("Raw:    ", sample)
    print("Cleaned:", clean_text(sample))

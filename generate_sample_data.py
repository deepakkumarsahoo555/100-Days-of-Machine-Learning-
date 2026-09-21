"""
Generates a small synthetic labeled news dataset (data/news_dataset.csv)
so the pipeline can be trained and tested fully offline, without needing
to download the Kaggle "Fake and Real News" dataset.

For a real project, swap this out for Fake.csv / True.csv from:
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
(see src/train_model.py load_kaggle_dataset() for how to load those instead)
"""

import random
import pandas as pd

random.seed(42)

REAL_TEMPLATES = [
    "Government announces new policy on {topic} to support citizens",
    "Central bank raises interest rates amid concerns over {topic}",
    "New study from {org} shows benefits of regular {topic}",
    "{org} reports quarterly earnings above analyst expectations",
    "City council approves budget for new {topic} infrastructure project",
    "Health officials release updated guidance on {topic} prevention",
    "Scientists at {org} publish peer-reviewed research on {topic}",
    "Local officials confirm progress on {topic} initiative this year",
    "{org} announces partnership to expand access to {topic} services",
    "Election results confirmed after official count in {topic} region",
]

FAKE_TEMPLATES = [
    "You won't believe what this celebrity did with {topic}!!!",
    "Breaking: {topic} secretly caused by shadowy government group",
    "Doctors HATE this one simple {topic} trick, click to find out",
    "Shocking: {org} caught hiding the truth about {topic}",
    "Miracle {topic} cure banned by big pharma, they don't want you to know",
    "Aliens spotted near {org} facility during {topic} experiment",
    "This {topic} conspiracy will change how you see the world forever",
    "Leaked documents reveal {org} plans to control {topic} worldwide",
    "Local mom discovers {topic} secret that scientists can't explain",
    "You've been lied to about {topic} your whole life, here's proof",
]

TOPICS = [
    "healthcare", "education", "climate change", "the economy", "vaccines",
    "exercise", "technology", "immigration", "housing", "taxes",
    "renewable energy", "public transport", "social media", "elections", "AI",
]

ORGS = [
    "Harvard University", "the World Health Organization", "NASA",
    "the Federal Reserve", "Google", "the United Nations", "Stanford",
    "the CDC", "the World Bank", "MIT",
]


def _fill(template):
    return template.format(topic=random.choice(TOPICS), org=random.choice(ORGS))


def generate_dataset(n_per_class=350):
    rows = []
    for _ in range(n_per_class):
        rows.append({"text": _fill(random.choice(REAL_TEMPLATES)), "label": "REAL"})
        rows.append({"text": _fill(random.choice(FAKE_TEMPLATES)), "label": "FAKE"})

    df = pd.DataFrame(rows).drop_duplicates(subset="text").sample(frac=1, random_state=42)
    return df.reset_index(drop=True)


if __name__ == "__main__":
    df = generate_dataset(n_per_class=400)
    out_path = "data/news_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    print(df["label"].value_counts())

"""
STEP 3: Extract features (turn text into numbers).

Machine learning models can't read words, they need numbers. This
step converts each cleaned SMS message into a numeric vector using
TF-IDF (Term Frequency - Inverse Document Frequency).

CRITICAL ORDERING RULE: we split into train/test BEFORE fitting the
vectorizer. If we fit TF-IDF on the whole dataset first, the
vectorizer "sees" words from the test set while building its
vocabulary and weights, that's data leakage. It would make our
evaluation in step 5 lie to us by looking better than the model
will actually perform on truly unseen messages.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import joblib

# ----------------------------------------------------------------
# 1. Load the cleaned dataset from step 2
# ----------------------------------------------------------------
df = pd.read_csv("data/spam_clean.csv")

# Drop rows where cleaning left nothing behind (e.g. emoticon-only
# messages like ":)" became empty strings, a model can't learn
# from empty text)
df = df.dropna(subset=["clean_message"])
df = df[df["clean_message"].str.strip() != ""]
print(f"{len(df)} messages after dropping empties")

# ----------------------------------------------------------------
# 2. Encode labels as numbers
# ----------------------------------------------------------------
# scikit-learn models expect numeric labels, not strings.
# ham -> 0, spam -> 1. We map spam to 1 deliberately: in the
# evaluation metrics later (precision/recall), "1" is treated as
# the "positive class", the one we care about catching.
df["label_num"] = (df["label"] == "spam").astype(int)

# ----------------------------------------------------------------
# 3. Split into train and test sets FIRST
# ----------------------------------------------------------------
# test_size=0.2 -> 80% of messages train the model, 20% are held
# back to test on messages the model has never seen.
#
# stratify=df["label_num"] -> this keeps the same spam/ham ratio
# (86%/14%) in BOTH the train and test sets. Without this, random
# chance could give you a test set with very few spam examples,
# making your evaluation numbers unreliable.
#
# random_state=42 -> makes the split reproducible. Anyone running
# this script gets the exact same train/test split every time,
# which matters for comparing model results fairly.
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["clean_message"],
    df["label_num"],
    test_size=0.2,
    stratify=df["label_num"],
    random_state=42,
)

print(f"Train set: {len(X_train_text)} messages")
print(f"Test set:  {len(X_test_text)} messages")

# ----------------------------------------------------------------
# 4. Fit TF-IDF on the TRAINING text only
# ----------------------------------------------------------------
# What TF-IDF does, in plain terms:
#   - Term Frequency (TF): how often a word appears in THIS message
#   - Inverse Document Frequency (IDF): words that appear in almost
#     every message (like "the", "to") get a LOW score, because
#     they don't help distinguish spam from ham. Rare, distinctive
#     words (like "urgent", "winner", "moneytoken") get a HIGH score.
# The result: each message becomes a vector of numbers where
# distinctive fraud-language words get more weight than filler words.
#
# ngram_range=(1, 2) -> use both single words ("winner") AND two-word
# phrases ("call now", "claim now"). Phrases often carry more fraud
# signal than single words alone.
#
# min_df=2 -> ignore words/phrases that appear in fewer than 2
# messages in the training set. This filters out typos and one-off
# noise that won't generalize to new messages anyway.
#
# max_features=3000 -> cap the vocabulary size at the 3000 most
# informative terms, keeps the model fast and avoids overfitting
# to extremely rare words.
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_features=3000,
)

X_train_tfidf = vectorizer.fit_transform(X_train_text)
# NOTE: fit_transform on TRAIN, but only transform (no fit) on TEST.
# This is the same leakage rule as the split, above: the vectorizer's
# vocabulary and IDF weights are learned ONLY from training data.
X_test_tfidf = vectorizer.transform(X_test_text)

print(f"\nTF-IDF vocabulary size: {len(vectorizer.vocabulary_)}")
print(f"Train matrix shape: {X_train_tfidf.shape}")
print(f"Test matrix shape:  {X_test_tfidf.shape}")

# ----------------------------------------------------------------
# 5. Add manual engineered features (message length)
# ----------------------------------------------------------------
# We noticed in step 1 that spam messages are longer on average
# (139 vs 71 characters). TF-IDF captures WORDS, but not this kind
# of structural signal, so we add it as an extra column.
#
# We compute length from the ORIGINAL df, using the same train/test
# row indices as the split above, so lengths line up with the right
# messages.
train_len = df.loc[X_train_text.index, "message"].str.len().values.reshape(-1, 1)
test_len = df.loc[X_test_text.index, "message"].str.len().values.reshape(-1, 1)

# hstack "glues" the sparse TF-IDF matrix and the dense length
# column together side by side into one final feature matrix.
X_train_final = hstack([X_train_tfidf, csr_matrix(train_len)])
X_test_final = hstack([X_test_tfidf, csr_matrix(test_len)])

print(f"\nFinal train matrix shape (TF-IDF + length): {X_train_final.shape}")
print(f"Final test matrix shape (TF-IDF + length):  {X_test_final.shape}")

# ----------------------------------------------------------------
# 6. Save everything for step 4 (model training)
# ----------------------------------------------------------------
# We save the fitted vectorizer too, we'll need the EXACT SAME
# vectorizer (same vocabulary, same weights) later to transform any
# new message the demo app receives.
joblib.dump(vectorizer, "data/tfidf_vectorizer.joblib")
joblib.dump(X_train_final, "data/X_train.joblib")
joblib.dump(X_test_final, "data/X_test.joblib")
joblib.dump(y_train, "data/y_train.joblib")
joblib.dump(y_test, "data/y_test.joblib")

print("\nSaved vectorizer and train/test feature matrices to data/")

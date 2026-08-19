"""
STEP 1: Load and explore the data.

Goal for this step: before touching any machine learning, we need to
know our data. How many messages? How balanced are spam vs ham?
How long are the messages? What do a few examples actually look like?

Skipping this step is the #1 reason ML projects go wrong later -
you end up debugging a model when the real problem was the data.
"""

import pandas as pd

# ----------------------------------------------------------------
# 1. Load the data
# ----------------------------------------------------------------
# The raw CSV has 5 columns because of some stray commas in the
# original text messages (extra ,,, at the end of most rows).
# We only care about the first two: v1 (label) and v2 (message text).
df = pd.read_csv(
    "data/spam.csv",
    encoding="latin-1",          # this file has some special characters (£ signs etc.)
    usecols=["v1", "v2"],        # ignore the 3 empty stray columns
)

# Rename columns to something readable
df.columns = ["label", "message"]

print("=" * 60)
print("SHAPE OF THE DATA")
print("=" * 60)
print(f"Rows (messages): {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

# ----------------------------------------------------------------
# 2. Check class balance (spam vs ham)
# ----------------------------------------------------------------
# This matters a LOT. If 87% of messages are "ham", a lazy model
# that just always predicts "ham" would score 87% accuracy while
# being useless. This is why we won't trust accuracy alone later.
print("\n" + "=" * 60)
print("CLASS BALANCE")
print("=" * 60)
counts = df["label"].value_counts()
percentages = df["label"].value_counts(normalize=True) * 100
for label in counts.index:
    print(f"{label:5s}: {counts[label]:5d} messages ({percentages[label]:.1f}%)")

# ----------------------------------------------------------------
# 3. Look at message length by class
# ----------------------------------------------------------------
# Fraud/spam messages are very often longer than normal texts,
# because they're trying to convince you of something (urgency,
# a prize, a call to action). This is a useful signal we can
# even turn into a feature later.
df["length"] = df["message"].str.len()

print("\n" + "=" * 60)
print("MESSAGE LENGTH BY CLASS (characters)")
print("=" * 60)
print(df.groupby("label")["length"].describe()[["mean", "min", "max"]])

# ----------------------------------------------------------------
# 4. Look at real examples of each class
# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("SAMPLE HAM MESSAGES (legitimate)")
print("=" * 60)
for msg in df[df["label"] == "ham"]["message"].sample(3, random_state=42):
    print(f"- {msg}")

print("\n" + "=" * 60)
print("SAMPLE SPAM MESSAGES (fraud/spam)")
print("=" * 60)
for msg in df[df["label"] == "spam"]["message"].sample(3, random_state=42):
    print(f"- {msg}")

# ----------------------------------------------------------------
# 5. Check for missing/duplicate data
# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("DATA QUALITY CHECKS")
print("=" * 60)
print(f"Missing values:\n{df.isnull().sum()}")
print(f"\nDuplicate messages: {df.duplicated(subset='message').sum()}")

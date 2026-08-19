"""
STEP 2: Clean and preprocess the text.

Goal: raw SMS text is messy (mixed case, punctuation, URLs, phone
numbers, duplicates). Before we can turn text into numbers for a
model, we clean it up so the model learns real patterns instead of
noise like capitalization or stray punctuation.

Key idea for fraud detection specifically: don't just delete things
like phone numbers and URLs, they ARE a fraud signal (fraud texts
almost always contain a call-back number or a link). So instead of
deleting them, we replace them with a placeholder token that keeps
the SIGNAL ("this message has a phone number") without keeping the
exact number, which would be useless noise to a model anyway (it
will never see that exact number again in the test set).
"""

import re
import pandas as pd

# ----------------------------------------------------------------
# 1. Load the base dataset
# ----------------------------------------------------------------
df = pd.read_csv("data/spam.csv", encoding="latin-1", usecols=["v1", "v2"])
df.columns = ["label", "message"]
print(f"Loaded {len(df)} messages")

# ----------------------------------------------------------------
# 2. Remove duplicate messages
# ----------------------------------------------------------------
# Why this matters: if the same message appears in both train and
# test sets after we split later, the model would just be
# "memorizing" instead of generalizing, which makes our accuracy
# numbers look better than they really are.
before = len(df)
df = df.drop_duplicates(subset="message").reset_index(drop=True)
print(f"Removed {before - len(df)} duplicate messages, {len(df)} remain")


def clean_text(text: str) -> str:
    """Clean a single SMS message while preserving fraud signals."""
    text = text.lower()

    # Replace URLs with a placeholder (keeps the "has a link" signal)
    text = re.sub(r"http\S+|www\.\S+", " urltoken ", text)

    # Replace money amounts / currency symbols with a placeholder
    text = re.sub(r"[£$€₦]|(\bn\d{2,}\b)", " moneytoken ", text)

    # Replace phone numbers (7+ consecutive digits) with a placeholder
    text = re.sub(r"\b\d{7,}\b", " phonetoken ", text)

    # Replace remaining standalone numbers with a placeholder
    text = re.sub(r"\b\d+\b", " numtoken ", text)

    # Remove punctuation (keep letters, spaces, and our tokens)
    text = re.sub(r"[^a-z\s]", " ", text)

    # Collapse multiple spaces into one
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ----------------------------------------------------------------
# 3. Apply cleaning to every message
# ----------------------------------------------------------------
df["clean_message"] = df["message"].apply(clean_text)

print("\n" + "=" * 60)
print("BEFORE / AFTER EXAMPLES")
print("=" * 60)
for i in df.sample(4, random_state=1).index:
    print(f"\nRAW:   {df.loc[i, 'message']}")
    print(f"CLEAN: {df.loc[i, 'clean_message']}")

# ----------------------------------------------------------------
# 4. Check for any messages that became empty after cleaning
# ----------------------------------------------------------------
empty_after_cleaning = (df["clean_message"].str.strip() == "").sum()
print(f"\nMessages empty after cleaning: {empty_after_cleaning}")

# ----------------------------------------------------------------
# 5. Save the cleaned dataset for the next step
# ----------------------------------------------------------------
df.to_csv("data/spam_clean.csv", index=False)
print(f"\nSaved cleaned dataset to data/spam_clean.csv ({len(df)} messages)")

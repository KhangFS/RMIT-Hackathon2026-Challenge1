import pandas as pd
import re
from collections import Counter


# ==================================================
# ĐỌC DỮ LIỆU
# ==================================================

df = pd.read_csv(
    "/workspace/challenge_1/rmit_hackathon_2026/data/train.csv"
)

# ==================================================
# THỐNG KÊ NGÔN NGỮ
# ==================================================

print("\n" + "=" * 50)
print("LANGUAGE DISTRIBUTION")
print("=" * 50)

language_counts = df["language"].value_counts()

print(language_counts)

print("\nTổng số ngôn ngữ:",
      df["language"].nunique())

# ==================================================
# THỐNG KÊ LABEL
# ==================================================

print("\n" + "=" * 50)
print("LABEL DISTRIBUTION")
print("=" * 50)

label_counts = df["label"].value_counts()

print(label_counts)

print("\nTổng số label:",
      df["label"].nunique())

# ==================================================
# THỐNG KÊ LABEL THEO NGÔN NGỮ
# ==================================================

print("\n" + "=" * 50)
print("LANGUAGE x LABEL")
print("=" * 50)

cross_table = pd.crosstab(
    df["language"],
    df["label"]
)

print(cross_table)

# ==================================================
# PHÁT HIỆN KÝ TỰ ĐẶC BIỆT
# ==================================================

special_pattern = r'[^a-zA-Z0-9\s]'

special_rows = []

all_special_chars = []

for idx, row in df.iterrows():

    text = str(row["text"])

    found_chars = re.findall(
        special_pattern,
        text
    )

    if found_chars:

        unique_chars = sorted(
            list(set(found_chars))
        )

        special_rows.append({
            "id": row["id"],
            "language": row["language"],
            "label": row["label"],
            "row_index": idx,
            "special_char_count": len(found_chars),
            "special_chars": " ".join(unique_chars),
            "text": text
        })

        all_special_chars.extend(
            found_chars
        )

# ==================================================
# TÓM TẮT KÝ TỰ ĐẶC BIỆT
# ==================================================

print("\n" + "=" * 50)
print("SPECIAL CHARACTER SUMMARY")
print("=" * 50)

print("Tổng số dòng:",
      len(df))

print("Dòng chứa ký tự đặc biệt:",
      len(special_rows))

ratio = len(special_rows) / len(df)

print(
    f"Tỷ lệ dòng có ký tự đặc biệt: {ratio:.2%}"
)

# ==================================================
# IN 10 DÒNG ĐẦU TIÊN CHỨA KÝ TỰ ĐẶC BIỆT
# ==================================================

print("\n" + "=" * 50)
print("FIRST 10 ROWS WITH SPECIAL CHARS")
print("=" * 50)

for row in special_rows[:10]:

    print("\n--------------------")
    print("ID:", row["id"])
    print("Language:", row["language"])
    print("Label:", row["label"])
    print("Special Chars:",
          row["special_chars"])
    print("Count:",
          row["special_char_count"])

# ==================================================
# THỐNG KÊ TẦN SUẤT KÝ TỰ ĐẶC BIỆT
# ==================================================

special_counter = Counter(
    all_special_chars
)

freq_df = pd.DataFrame(
    special_counter.items(),
    columns=[
        "character",
        "frequency"
    ]
)

freq_df = freq_df.sort_values(
    by="frequency",
    ascending=False
)

print("\n" + "=" * 50)
print("TOP 30 SPECIAL CHARACTERS")
print("=" * 50)

print(
    freq_df.head(30)
)

# ==================================================
# LƯU FILE 1:
# DANH SÁCH DÒNG CHỨA KÝ TỰ ĐẶC BIỆT
# ==================================================

special_df = pd.DataFrame(
    special_rows
)

special_df.to_csv(
    "special_character_rows.csv",
    index=False,
    encoding="utf-8-sig"
)

# ==================================================
# LƯU FILE 2:
# TẦN SUẤT KÝ TỰ ĐẶC BIỆT
# ==================================================

freq_df.to_csv(
    "special_character_frequency.csv",
    index=False,
    encoding="utf-8-sig"
)

# ==================================================
# THỐNG KÊ ĐỘ DÀI PROMPT
# ==================================================

df["text_length"] = (
    df["text"]
    .astype(str)
    .apply(len)
)

print("\n" + "=" * 50)
print("TEXT LENGTH")
print("=" * 50)

print(
    df["text_length"].describe()
)

# ==================================================
# LƯU THỐNG KÊ NGÔN NGỮ
# ==================================================

language_counts.to_csv(
    "language_distribution.csv",
    header=["count"]
)

# ==================================================
# LƯU THỐNG KÊ LABEL
# ==================================================

label_counts.to_csv(
    "label_distribution.csv",
    header=["count"]
)

print("\n" + "=" * 50)
print("FILES SAVED")
print("=" * 50)

print("special_character_rows.csv")
print("special_character_frequency.csv")
print("language_distribution.csv")
print("label_distribution.csv")
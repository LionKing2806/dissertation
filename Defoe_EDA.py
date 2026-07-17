"""
Basic exploratory data analysis for the Defoe biblical-reference dataset.

Supervisor-informed interpretation
-----------------------------------
1. Each row is treated as one interpretive biblical-reference entry.
2. The same Defoe passage may legitimately appear more than once when it refers to different biblical figures, books, verses, or interpretive targets.
3. Weak / Middling / Strong represents confidence that Defoe is making a biblical reference.
4. Incorrect entries are retained. They indicate that the recorded in-text interpretation or biblical reference may not make sense.
5. Multiple references attached to the same passage are grouped so that they can later be analyzed together at passage level.
6. Stance targets are not automatically assigned during this EDA. Candidate theological issues should be developed from the data and manually validated.

Input:
    D:/downloads/Defoe References.xlsx

Main sheet:
    Everything

Outputs:
    outputs/defoe_eda/
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd


# ============================================================
# 1. FILE PATHS
# ============================================================

INPUT_FILE = Path("D:/downloads/Defoe References.xlsx")
SHEET_NAME = "Everything"

OUTPUT_DIR = Path("outputs/defoe_eda")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

df_raw = pd.read_excel(
    INPUT_FILE,
    sheet_name=SHEET_NAME,
    dtype=object
)

# Preserve the original Excel row number.
# Row 1 contains the headers, so the first data row is Excel row 2.
df_raw.insert(
    0,
    "Excel row",
    np.arange(2, len(df_raw) + 2)
)

print(f"Loaded sheet: {SHEET_NAME}")
print(f"Rows: {len(df_raw):,}")
print(f"Columns before cleaning: {len(df_raw.columns):,}")


# ============================================================
# 3. STANDARDISE COLUMN NAMES
# ============================================================

def clean_column_name(column):
    column = str(column).strip().lower()
    column = column.replace("?", "")
    column = column.replace("/", "_")
    column = column.replace(",", "")
    column = column.replace("(", "")
    column = column.replace(")", "")
    column = re.sub(r"\s+", "_", column)
    column = re.sub(r"_+", "_", column)
    return column.strip("_")


df = df_raw.copy()
df.columns = [clean_column_name(column) for column in df.columns]

# Remove fully empty unnamed columns.
empty_unnamed_columns = [
    column
    for column in df.columns
    if column.startswith("unnamed") and df[column].isna().all()
]

df = df.drop(columns=empty_unnamed_columns)


# Rename fields to shorter working names.
df = df.rename(
    columns={
        "excel_row": "excel_row",
        "publisher_journal_date": "publisher_journal_date",
        "type_direct_quote_allusion_story_parallels": "reference_type",
        "is_spoken_written": "spoken_written",
        "right_wrong": "interpretation_status",
        "vague_middling_strong": "reference_confidence",
        "spotted_by_me_other": "spotted_by",
        "type_of_book": "biblical_book_type"
    }
)

print("\nColumns used:")
for column in df.columns:
    print(f"  - {column}")


# ============================================================
# 4. CLEAN VALUES
# ============================================================

MISSING_MARKERS = {
    "",
    "-",
    "–",
    "—",
    "n/a",
    "na",
    "none",
    "null",
    "nan",
    "unknown",
    "not known"
}


def clean_text_value(value):
    if pd.isna(value):
        return pd.NA

    text = str(value).strip()

    if text.lower() in MISSING_MARKERS:
        return pd.NA

    return text


for column in df.columns:
    if column != "excel_row":
        df[column] = df[column].apply(clean_text_value)


def normalise_category(value):
    if pd.isna(value):
        return pd.NA

    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


categorical_columns = [
    "classification",
    "bible_book",
    "testament",
    "biblical_book_type",
    "reference_type",
    "who_by",
    "spoken_written",
    "interpretation_status",
    "reference_confidence",
    "spotted_by"
]

for column in categorical_columns:
    if column in df.columns:
        df[column] = df[column].apply(normalise_category)


# ============================================================
# 5. NORMALISE COMMON CATEGORY VARIANTS
# ============================================================

category_replacements = {
    "reference_type": {
        "allsuion": "allusion",
        "alluson": "allusion",
        "direct quotation": "direct quote",
        "quotation": "direct quote",
        "quote": "direct quote",
        "story parallel": "story parallels",
        "parrallel": "story parallels",
        "paraphrases": "paraphrase"
    },

    "spoken_written": {
        "write": "written",
        "writen": "written",
        "speech": "spoken"
    },

    "interpretation_status": {
        "right": "correct",
        "wrong": "incorrect",
        "unclear": "uncertain",
        "not sure": "uncertain",
        "possibly": "uncertain"
    },

    "reference_confidence": {
        "weak": "weak",
        "vague": "weak",
        "vauge": "weak",
        "medium": "middling",
        "middle": "middling",
        "middling": "middling",
        "strong": "strong"
    },

    "spotted_by": {
        "myself": "me",
        "beck": "other",
        "becks": "other",
        "someone else": "other",
        "others": "other"
    },

    "testament": {
        "ot": "old",
        "old testament": "old",
        "nt": "new",
        "new testament": "new",
        "both testaments": "both"
    }
}

for column, replacements in category_replacements.items():
    if column in df.columns:
        df[column] = df[column].replace(replacements)


# ============================================================
# 6. UNIT OF ANALYSIS
# ============================================================

unit_of_analysis = (
    "Each row is treated as one interpretive biblical-reference entry: "
    "a recorded claim that a Defoe passage relates to a biblical person, "
    "book, verse, story, phrase or broader biblical material. The same "
    "Defoe passage may legitimately occur in more than one row when it "
    "contains multiple references or interpretive targets."
)

print("\n" + "=" * 70)
print("UNIT OF ANALYSIS")
print("=" * 70)
print(unit_of_analysis)


# ============================================================
# 7. BASIC COUNTS
# ============================================================

total_rows = len(df)

print("\n" + "=" * 70)
print("BASIC COUNTS")
print("=" * 70)

print(f"Total interpretive-reference entries: {total_rows:,}")

for column, label in [
    ("title", "Unique Defoe titles"),
    ("author", "Unique authors"),
    ("editor", "Unique editors"),
    ("publisher_journal_date", "Unique publisher/journal/date values")
]:
    if column in df.columns:
        print(
            f"{label}: "
            f"{df[column].nunique(dropna=True):,}"
        )


# ============================================================
# 8. YEAR ANALYSIS
# ============================================================

# The Year column is treated as the publication year of the Defoe work.
DEFOE_EARLIEST_YEAR = 1660
DEFOE_LATEST_YEAR = 1731


def extract_year(value):
    """
    Extract the first four-digit year from a value.

    Examples:
        1719       -> 1719
        "c. 1719"  -> 1719
        "[1729?]"  -> 1729
    """
    if pd.isna(value):
        return np.nan

    match = re.search(
        r"\b(1[5-9]\d{2}|20\d{2})\b",
        str(value)
    )

    return float(match.group(1)) if match else np.nan


# Convert Year to numeric where possible.
df["year_numeric"] = (
    df["year"]
    .apply(extract_year)
    .astype("float64")
)


# Flag year values containing explicit uncertainty markers.
df["uncertain_year_flag"] = (
    df["year"]
    .fillna("")
    .astype(str)
    .str.contains(
        r"\?|circa|\bc\.",
        case=False,
        regex=True
    )
)


# Flag years outside Defoe's lifetime.
df["outside_defoe_lifetime_flag"] = (
    df["year_numeric"].notna()
    & (
        (df["year_numeric"] < DEFOE_EARLIEST_YEAR)
        | (df["year_numeric"] > DEFOE_LATEST_YEAR)
    )
)


# Flag missing or non-parsable years.
df["unparseable_year_flag"] = (
    df["year_numeric"].isna()
)


# Any of these conditions may require checking.
df["unusual_year_flag"] = (
    df["uncertain_year_flag"]
    | df["outside_defoe_lifetime_flag"]
    | df["unparseable_year_flag"]
)


# ------------------------------------------------------------
# Year-range summary
# ------------------------------------------------------------

valid_years = df["year_numeric"].dropna()

print("\n" + "=" * 70)
print("YEAR RANGE")
print("=" * 70)

if valid_years.empty:
    print("No numeric publication years could be identified.")
else:
    print(f"Earliest publication year: {int(valid_years.min())}")
    print(f"Latest publication year:   {int(valid_years.max())}")
    print(f"Median publication year:   {valid_years.median():.1f}")


# ------------------------------------------------------------
# Unusual years
# ------------------------------------------------------------

unusual_years = df.loc[
    df["unusual_year_flag"],
    [
        column
        for column in [
            "excel_row",
            "title",
            "author",
            "year",
            "year_numeric",
            "uncertain_year_flag",
            "outside_defoe_lifetime_flag",
            "unparseable_year_flag"
        ]
        if column in df.columns
    ]
].copy()


print("\nUnusual publication years requiring review:")

if unusual_years.empty:
    print("  None")
    print(
        "  All recorded publication years fall within "
        "Defoe's lifetime and contain no explicit uncertainty."
    )
else:
    print(f"  {len(unusual_years)} entries")
    print(unusual_years.to_string(index=False))

# ============================================================
# 9. FREQUENCY TABLE FUNCTION
# ============================================================

def frequency_table(data, column):
    if column not in data.columns:
        return pd.DataFrame(
            columns=["category", "count", "percentage"]
        )

    values = data[column].fillna("[missing]")

    result = (
        values.value_counts(dropna=False)
        .rename_axis("category")
        .reset_index(name="count")
    )

    result["percentage"] = (
        result["count"] / len(data) * 100
    ).round(2)

    return result


frequency_columns = {
    "classification": "entries_by_classification.csv",
    "bible_book": "entries_by_biblical_book.csv",
    "testament": "entries_by_testament.csv",
    "biblical_book_type": "entries_by_biblical_book_type.csv",
    "reference_type": "entries_by_reference_type.csv",
    "spoken_written": "entries_by_spoken_written.csv",
    "interpretation_status": "entries_by_interpretation_status.csv",
    "reference_confidence": "entries_by_reference_confidence.csv",
    "spotted_by": "entries_by_spotted_by.csv"
}

frequency_results = {}

for column, output_name in frequency_columns.items():
    if column not in df.columns:
        continue

    result = frequency_table(df, column)
    frequency_results[column] = result

    print("\n" + "=" * 70)
    print(column.replace("_", " ").upper())
    print("=" * 70)
    print(result.to_string(index=False))


# ============================================================
# 10. MISSING VALUES
# ============================================================

key_fields = [
    "title",
    "publisher_journal_date",
    "year",
    "book_quote",
    "bible_book",
    "chapter",
    "verse",
    "full_bible_verse",
    "reference_type",
    "interpretation_status",
    "reference_confidence"
]

existing_key_fields = [
    column
    for column in key_fields
    if column in df.columns
]

missing_summary = pd.DataFrame({
    "field": existing_key_fields,
    "missing_count": [
        df[column].isna().sum()
        for column in existing_key_fields
    ]
})

missing_summary["missing_percentage"] = (
    missing_summary["missing_count"]
    / total_rows
    * 100
).round(2)

missing_summary = missing_summary.sort_values(
    "missing_count",
    ascending=False
)


print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)
print(missing_summary.to_string(index=False))


# ============================================================
# 11. CREATE PASSAGE-LEVEL IDENTIFIERS
# ============================================================

def normalise_for_grouping(value):
    """
    Minimal normalisation for grouping only.

    The original text is not altered. This is used to recognise the same
    passage when differences are limited to whitespace or capitalisation.
    """
    if pd.isna(value):
        return ""

    text = str(value).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


grouping_fields = [
    column
    for column in [
        "title",
        "page_no",
        "book_quote"
    ]
    if column in df.columns
]

if grouping_fields:
    passage_key_text = (
        df[grouping_fields]
        .apply(
            lambda column: column.map(normalise_for_grouping)
        )
        .agg(" || ".join, axis=1)
    )

    df["passage_id"] = pd.factorize(
        passage_key_text
    )[0] + 1
else:
    df["passage_id"] = np.arange(1, len(df) + 1)


passage_counts = (
    df.groupby("passage_id")
    .size()
    .rename("entries_for_passage")
)

df = df.merge(
    passage_counts,
    on="passage_id",
    how="left"
)

df["multiple_entries_same_passage_flag"] = (
    df["entries_for_passage"] > 1
)


# ============================================================
# 12. DISTINGUISH LEGITIMATE MULTIPLE REFERENCES FROM DUPLICATES
# ============================================================

reference_identity_fields = [
    column
    for column in [
        "bible_book",
        "chapter",
        "verse",
        "full_bible_verse",
        "reference_type"
    ]
    if column in df.columns
]


def combine_reference_identity(row):
    values = []

    for column in reference_identity_fields:
        value = row.get(column, pd.NA)
        values.append(normalise_for_grouping(value))

    return " || ".join(values)


df["reference_identity"] = df.apply(
    combine_reference_identity,
    axis=1
)


# Number of distinct recorded biblical references attached to each passage.
distinct_reference_counts = (
    df.groupby("passage_id")["reference_identity"]
    .nunique(dropna=False)
    .rename("distinct_references_for_passage")
)

df = df.merge(
    distinct_reference_counts,
    on="passage_id",
    how="left"
)


# A passage with multiple rows and more than one reference identity is
# treated as a legitimate multi-reference passage, not an automatic error.
df["multi_reference_passage_flag"] = (
    (df["entries_for_passage"] > 1)
    & (df["distinct_references_for_passage"] > 1)
)


# Exact analytical duplicates:
# same passage AND same recorded reference AND same interpretation fields.
duplicate_comparison_fields = [
    column
    for column in [
        "title",
        "author",
        "editor",
        "publisher_journal_date",
        "year",
        "classification",
        "page_no",
        "book_quote",
        "bible_book",
        "chapter",
        "verse",
        "testament",
        "biblical_book_type",
        "full_bible_verse",
        "reference_type",
        "who_by",
        "spoken_written",
        "interpretation_status",
        "reference_confidence",
        "spotted_by",
        "notes"
    ]
    if column in df.columns
]

df["exact_record_duplicate_flag"] = df.duplicated(
    subset=duplicate_comparison_fields,
    keep=False
)


# Rows may look identical because the distinct referents were recorded only
# conceptually, rather than in separate Bible-book fields.
# These are retained and sent for review, not labelled as errors.
df["same_passage_same_reference_multiple_rows_flag"] = (
    (df["entries_for_passage"] > 1)
    & (df["distinct_references_for_passage"] == 1)
)

print("\n" + "=" * 70)
print("REPEATED BIBLICAL REFERENCES")
print("=" * 70)

reference_columns = [
    "bible_book",
    "chapter",
    "verse"
]

repeated_bible_refs = (
    df
    .dropna(subset=reference_columns)
    .groupby(reference_columns)
    .size()
    .reset_index(name="count")
)

repeated_bible_refs = repeated_bible_refs[
    repeated_bible_refs["count"] > 1
].sort_values(
    "count",
    ascending=False
)

print(
    f"Distinct biblical references appearing more than once: "
    f"{len(repeated_bible_refs)}"
)

if not repeated_bible_refs.empty:
    print("\nTop 20 repeated biblical references:")
    print(
        repeated_bible_refs
        .head(20)
        .to_string(index=False)
    )
# ============================================================
# 13. SPECIAL HANDLING OF EXCEL ROWS 468 AND 469
# ============================================================

# Becks confirmed that these rows are intentional:
# one refers to St. Paul and one to King David.
KNOWN_INTENTIONAL_MULTI_REFERENCE_ROWS = {468, 469}

df["known_intentional_multi_reference_flag"] = (
    df["excel_row"].isin(
        KNOWN_INTENTIONAL_MULTI_REFERENCE_ROWS
    )
)

# They should not be treated as accidental duplicates.
df.loc[
    df["known_intentional_multi_reference_flag"],
    "exact_record_duplicate_flag"
] = False

df.loc[
    df["known_intentional_multi_reference_flag"],
    "same_passage_same_reference_multiple_rows_flag"
] = False

df.loc[
    df["known_intentional_multi_reference_flag"],
    "multi_reference_passage_flag"
] = True


# Add an explicit interpretation note.
df["duplicate_interpretation"] = pd.NA

df.loc[
    df["entries_for_passage"] == 1,
    "duplicate_interpretation"
] = "single recorded entry for passage"

df.loc[
    df["multi_reference_passage_flag"],
    "duplicate_interpretation"
] = (
    "same Defoe passage with multiple biblical references"
)

df.loc[
    df["same_passage_same_reference_multiple_rows_flag"],
    "duplicate_interpretation"
] = (
    "same passage and same recorded reference; manual review required"
)

df.loc[
    df["exact_record_duplicate_flag"],
    "duplicate_interpretation"
] = (
    "fully repeated record; manual review required"
)

df.loc[
    df["known_intentional_multi_reference_flag"],
    "duplicate_interpretation"
] = (
    "intentional: separate references to St. Paul and King David"
)


# ============================================================
# 14. INCOMPLETE BIBLICAL REFERENCES
# ============================================================

if all(
    column in df.columns
    for column in ["bible_book", "chapter", "verse"]
):
    df["incomplete_bible_reference_flag"] = (
        df["bible_book"].isna()
        | df["chapter"].isna()
        | df["verse"].isna()
    )

    df["book_present_but_chapter_or_verse_missing_flag"] = (
        df["bible_book"].notna()
        & (
            df["chapter"].isna()
            | df["verse"].isna()
        )
    )
else:
    df["incomplete_bible_reference_flag"] = False
    df["book_present_but_chapter_or_verse_missing_flag"] = False


incomplete_references = df.loc[
    df["incomplete_bible_reference_flag"],
    [
        column
        for column in [
            "excel_row",
            "passage_id",
            "title",
            "page_no",
            "book_quote",
            "bible_book",
            "chapter",
            "verse",
            "full_bible_verse",
            "reference_type",
            "reference_confidence",
            "notes"
        ]
        if column in df.columns
    ]
]


# ============================================================
# 15. REFERENCE CONFIDENCE
# ============================================================

# Becks clarified that Weak/Middling/Strong expresses how confident
# she is that Defoe is referencing the Bible.

if "reference_confidence" in df.columns:
    df["weak_reference_flag"] = (
        df["reference_confidence"] == "weak"
    )

    df["middling_reference_flag"] = (
        df["reference_confidence"] == "middling"
    )

    df["strong_reference_flag"] = (
        df["reference_confidence"] == "strong"
    )
else:
    df["weak_reference_flag"] = False
    df["middling_reference_flag"] = False
    df["strong_reference_flag"] = False


weak_references = df.loc[
    df["weak_reference_flag"],
    [
        column
        for column in [
            "excel_row",
            "passage_id",
            "title",
            "page_no",
            "book_quote",
            "bible_book",
            "chapter",
            "verse",
            "full_bible_verse",
            "reference_type",
            "reference_confidence",
            "notes"
        ]
        if column in df.columns
    ]
]


# ============================================================
# 16. CORRECT / INCORRECT / UNCERTAIN
# ============================================================

# Incorrect rows are retained. They are not automatically excluded.
# The field describes the plausibility of the in-text interpretation
# or the recorded chapter-and-verse reference.

if "interpretation_status" in df.columns:
    df["incorrect_interpretation_flag"] = (
        df["interpretation_status"] == "incorrect"
    )

    df["uncertain_interpretation_flag"] = (
        df["interpretation_status"] == "uncertain"
    )
else:
    df["incorrect_interpretation_flag"] = False
    df["uncertain_interpretation_flag"] = False


interpretive_problem_rows = df.loc[
    df["incorrect_interpretation_flag"]
    | df["uncertain_interpretation_flag"],
    [
        column
        for column in [
            "excel_row",
            "passage_id",
            "title",
            "page_no",
            "book_quote",
            "bible_book",
            "chapter",
            "verse",
            "full_bible_verse",
            "reference_type",
            "interpretation_status",
            "reference_confidence",
            "notes"
        ]
        if column in df.columns
    ]
]


# ============================================================
# 17. UNCERTAIN ATTRIBUTION
# ============================================================

attribution_pattern = (
    r"\b(?:"
    r"uncertain|unclear|unknown|possibly|probably|"
    r"questionable|doubtful|attributed|attribution|"
    r"disputed|anonymous|anon"
    r")\b"
)

attribution_fields = [
    column
    for column in [
        "author",
        "editor",
        "title",
        "publisher_journal_date",
        "notes"
    ]
    if column in df.columns
]

if attribution_fields:
    attribution_text = (
        df[attribution_fields]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.lower()
    )

    df["uncertain_attribution_flag"] = (
        attribution_text.str.contains(
            attribution_pattern,
            regex=True,
            na=False
        )
    )
else:
    df["uncertain_attribution_flag"] = False


uncertain_attribution = df.loc[
    df["uncertain_attribution_flag"],
    [
        column
        for column in [
            "excel_row",
            "title",
            "author",
            "editor",
            "publisher_journal_date",
            "year",
            "notes"
        ]
        if column in df.columns
    ]
]


# ============================================================
# 18. POSSIBLE QUOTE–VERSE ALIGNMENT ISSUES
# ============================================================

# This is only a note-based screening method.
# It does not computationally prove that the texts fail to align.

alignment_pattern = (
    r"\b(?:"
    r"not exact|does not match|do not match|"
    r"doesn't match|no match|unclear match|"
    r"not clear|does not align|do not align|"
    r"wrong verse|wrong reference|different verse|"
    r"very vague|too vague|not a biblical reference|"
    r"not an exact ref"
    r")\b"
)

if "notes" in df.columns:
    df["possible_alignment_issue_flag"] = (
        df["notes"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.contains(
            alignment_pattern,
            regex=True,
            na=False
        )
    )
else:
    df["possible_alignment_issue_flag"] = False


possible_alignment_issues = df.loc[
    df["possible_alignment_issue_flag"],
    [
        column
        for column in [
            "excel_row",
            "passage_id",
            "title",
            "page_no",
            "book_quote",
            "bible_book",
            "chapter",
            "verse",
            "full_bible_verse",
            "interpretation_status",
            "reference_confidence",
            "notes"
        ]
        if column in df.columns
    ]
]


# ============================================================
# 19. MANUAL REVIEW FLAGS
# ============================================================

# Legitimate multi-reference passages are NOT automatically review errors.
review_flag_columns = [
    "outside_defoe_lifetime_flag",
    "uncertain_attribution_flag",
    "weak_reference_flag",
    "incorrect_interpretation_flag",
    "uncertain_interpretation_flag",
    "book_present_but_chapter_or_verse_missing_flag",
    "exact_record_duplicate_flag",
    "same_passage_same_reference_multiple_rows_flag",
    "possible_alignment_issue_flag"
]

df["manual_review_flag_count"] = (
    df[review_flag_columns]
    .astype(int)
    .sum(axis=1)
)

df["requires_manual_review"] = (
    df["manual_review_flag_count"] > 0
)


manual_review_columns = [
    column
    for column in [
        "excel_row",
        "passage_id",
        "title",
        "author",
        "year",
        "page_no",
        "book_quote",
        "bible_book",
        "chapter",
        "verse",
        "full_bible_verse",
        "reference_type",
        "interpretation_status",
        "reference_confidence",
        "spotted_by",
        "notes",
        "entries_for_passage",
        "distinct_references_for_passage",
        "duplicate_interpretation"
    ]
    if column in df.columns
] + review_flag_columns + [
    "manual_review_flag_count"
]

rows_for_manual_review = (
    df.loc[
        df["requires_manual_review"],
        manual_review_columns
    ]
    .sort_values(
        [
            "manual_review_flag_count",
            "excel_row"
        ],
        ascending=[False, True]
    )
)


# ============================================================
# 20. MAIN TABLE SUMMARY
# ============================================================

summary_rows = []


def add_summary(feature, category, count, denominator=total_rows):
    percentage = (
        round(count / denominator * 100, 2)
        if denominator
        else 0
    )

    summary_rows.append({
        "feature": feature,
        "category": category,
        "count": int(count),
        "percentage": percentage
    })


add_summary(
    "Dataset",
    "Interpretive reference entries",
    total_rows
)

add_summary(
    "Dataset",
    "Unique Defoe passages",
    df["passage_id"].nunique()
)

if "title" in df.columns:
    add_summary(
        "Dataset",
        "Unique Defoe titles",
        df["title"].nunique(dropna=True)
    )


summary_categories = {
    "Classification": "classification",
    "Testament": "testament",
    "Biblical book type": "biblical_book_type",
    "Reference type": "reference_type",
    "Spoken or written": "spoken_written",
    "Interpretation status": "interpretation_status",
    "Reference confidence": "reference_confidence",
    "Spotted by": "spotted_by"
}

for feature, column in summary_categories.items():
    if column not in df.columns:
        continue

    counts = (
        df[column]
        .fillna("[missing]")
        .value_counts()
    )

    for category, count in counts.items():
        add_summary(
            feature,
            category,
            count
        )

add_summary(
    "Data quality",
    "Weak-confidence references",
    df["weak_reference_flag"].sum()
)

add_summary(
    "Data quality",
    "Incorrect interpretations retained",
    df["incorrect_interpretation_flag"].sum()
)

add_summary(
    "Data quality",
    "Uncertain interpretations retained",
    df["uncertain_interpretation_flag"].sum()
)

add_summary(
    "Data quality",
    "References missing chapter or verse despite named Bible book",
    df["book_present_but_chapter_or_verse_missing_flag"].sum()
)

add_summary(
    "Data quality",
    "Possible fully repeated records",
    df["exact_record_duplicate_flag"].sum()
)

add_summary(
    "Data quality",
    "Rows requiring manual review",
    df["requires_manual_review"].sum()
)


table_summary = pd.DataFrame(summary_rows)


print("\n" + "=" * 70)
print("TABLE SUMMARY")
print("=" * 70)
print(table_summary.to_string(index=False))


# ============================================================
# 21. AUTOMATIC EDA PARAGRAPH
# ============================================================

unique_passages = df["passage_id"].nunique()

multi_passage_entries = int(
    df["multi_reference_passage_flag"].sum()
)

weak_count = int(
    df["weak_reference_flag"].sum()
)

incorrect_count = int(
    df["incorrect_interpretation_flag"].sum()
)

manual_review_count = int(
    df["requires_manual_review"].sum()
)

summary_paragraph = (
    f"The dataset contains {total_rows:,} interpretive biblical-reference "
    f"entries representing {unique_passages:,} distinct Defoe passages. "
    f"Because one passage may contain more than one biblical reference, "
    f"repeated Defoe quotations have not automatically been treated as "
    f"data errors. In total, {multi_passage_entries:,} entries belong to "
    f"passages with more than one distinct recorded reference. The "
    f"Weak/Middling/Strong field is interpreted as confidence that Defoe "
    f"is making a biblical reference; {weak_count:,} entries are currently "
    f"marked weak and should therefore be used cautiously rather than as "
    f"the main basis of an argument. The {incorrect_count:,} entries marked "
    f"incorrect have been retained because this label records a potentially "
    f"implausible in-text interpretation or biblical citation, rather than "
    f"indicating that the row should automatically be deleted. Overall, "
    f"{manual_review_count:,} entries received at least one review flag. "
    f"These flags identify records requiring closer checking and do not by "
    f"themselves establish that the records are erroneous."
)

with open(
    OUTPUT_DIR / "automatic_eda_paragraph.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(summary_paragraph)

print("\n" + "=" * 70)
print("AUTOMATIC EDA PARAGRAPH")
print("=" * 70)
print(summary_paragraph)


# ============================================================
# 22. SAVE CLEANED ENTRY-LEVEL DATA
# ============================================================

df.to_csv(
    OUTPUT_DIR / "defoe_entries_cleaned_with_flags.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 23. SAVE COMBINED EXCEL REPORT
# ============================================================

excel_output = OUTPUT_DIR / "defoe_eda_results.xlsx"

with pd.ExcelWriter(
    excel_output,
    engine="openpyxl"
) as writer:

    table_summary.to_excel(
        writer,
        sheet_name="Table summary",
        index=False
    )

    missing_summary.to_excel(
        writer,
        sheet_name="Missing values",
        index=False
    )

    weak_references.to_excel(
        writer,
        sheet_name="Weak references",
        index=False
    )

    interpretive_problem_rows.to_excel(
        writer,
        sheet_name="Incorrect uncertain",
        index=False
    )

    rows_for_manual_review.to_excel(
        writer,
        sheet_name="Manual review",
        index=False
    )

    for column, result in frequency_results.items():
        sheet_name = (
            column.replace("_", " ").title()[:31]
        )

        # Avoid duplicate Excel sheet names.
        if sheet_name not in writer.sheets:
            result.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False
            )


print("\nEDA completed.")
print(f"Output directory: {OUTPUT_DIR.resolve()}")
print(f"Combined workbook: {excel_output.resolve()}")

# print(df["unnamed:_21"].value_counts(dropna=False))
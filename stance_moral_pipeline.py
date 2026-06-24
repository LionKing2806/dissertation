import argparse
import os
import re
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix,f1_score

import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)


FOUNDATIONS = [
    "care",
    "fairness",
    "loyalty",
    "authority",
    "sanctity"
]


def load_mfd2_dictionary(dic_path):

    id_to_category = {}
    word_to_categories = {}

    with open(dic_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    phase = 0

    for line in lines:

        line = line.strip()

        if not line:
            continue

        if line == "%":
            phase += 1
            continue

        # category mapping
        if phase == 1:

            parts = line.split()

            idx = int(parts[0])
            category = parts[1]

            id_to_category[idx] = category

        # word mapping
        elif phase >= 2:

            parts = line.split()

            if len(parts) < 2:
                continue

            word = parts[0].lower()

            for cat in parts[1:]:

                if cat.isdigit():
                    category = id_to_category[int(cat)]
                else:
                    category = cat.lower()

                if word not in word_to_categories:
                    word_to_categories[word] = []

                word_to_categories[word].append(category)

    return word_to_categories


def tokenize(text):

    return re.findall(
        r"[a-zA-Z']+",
        str(text).lower()
    )


def score_mfd2_text(text, word_to_categories):

    tokens = tokenize(text)

    result = {}

    for foundation in FOUNDATIONS:

        result[foundation + "_virtue"] = 0
        result[foundation + "_vice"] = 0

    for token in tokens:

        if token not in word_to_categories:
            continue

        for category in word_to_categories[token]:

            if category.startswith("care"):
                if "virtue" in category:
                    result["care_virtue"] += 1
                else:
                    result["care_vice"] += 1

            elif category.startswith("fairness"):
                if "virtue" in category:
                    result["fairness_virtue"] += 1
                else:
                    result["fairness_vice"] += 1

            elif category.startswith("loyalty"):
                if "virtue" in category:
                    result["loyalty_virtue"] += 1
                else:
                    result["loyalty_vice"] += 1

            elif category.startswith("authority"):
                if "virtue" in category:
                    result["authority_virtue"] += 1
                else:
                    result["authority_vice"] += 1

            elif category.startswith("sanctity"):
                if "virtue" in category:
                    result["sanctity_virtue"] += 1
                else:
                    result["sanctity_vice"] += 1

    return result


def add_mfd2_features(df, text_col, dic_path):

    word_to_categories = load_mfd2_dictionary(
        dic_path
    )

    moral_scores = df[text_col].apply(
        lambda x: score_mfd2_text(
            x,
            word_to_categories
        )
    )

    moral_df = pd.DataFrame(
        list(moral_scores)
    )

    return pd.concat(
        [df.reset_index(drop=True), moral_df],
        axis=1
    )


def load_dataset(path):

    if path.endswith(".csv"):
        return pd.read_csv(path)

    elif path.endswith(".tsv"):
        return pd.read_csv(path, sep="\t")

    elif path.endswith(".txt"):
        return pd.read_csv(
            path,
            sep="\t",
            encoding="latin1"
        )

    elif path.endswith(".json"):
        return pd.read_json(path)

    elif path.endswith(".jsonl"):
        return pd.read_json(
            path,
            lines=True
        )

    else:
        raise ValueError(
            f"Unsupported file format: {path}"
        )
def clean_text_basic(text):
    text = str(text)

    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def historical_spelling_normalise(text):
    text = str(text)

    replacements = {
        "publick": "public",
        "shew": "show",
        "shewed": "showed",
        "chuse": "choose",
        "chused": "chose",
        "tho": "though",
        "altho": "although",
        "connexion": "connection",
        "compleat": "complete",
        "honour": "honor",
        "labour": "labor"
    }

    for old, new in replacements.items():
        text = re.sub(
            r"\b" + re.escape(old) + r"\b",
            new,
            text,
            flags=re.IGNORECASE
        )

    return text


def preprocess_text(text, historical=False):
    text = clean_text_basic(text)

    if historical:
        text = historical_spelling_normalise(text)

    return text

def detect_columns(df):

    text_candidates = [
        "tweet",
        "text",
        "sentence",
        "content",
        "comment",
        "post",
        "body",
        "book quote",
        "full bible verse",
        "notes"
    ]

    target_candidates = [
        "target",
        "topic",
        "question",
        "title",
        "classification",
        "bible book",
        "type of book"
    ]

    label_candidates = [
        "stance",
        "label",
        "class",
        "annotation",
        "right/wrong",
        "vague, middling, strong ?",
        "type (direct quote, allusion, story parallels)"
    ]

    cols_lower = {
        c.lower().strip(): c
        for c in df.columns
    }

    text_col = None
    target_col = None
    label_col = None

    for c in text_candidates:
        if c in cols_lower:
            text_col = cols_lower[c]
            break

    for c in target_candidates:
        if c in cols_lower:
            target_col = cols_lower[c]
            break

    for c in label_candidates:
        if c in cols_lower:
            label_col = cols_lower[c]
            break

    return text_col, target_col, label_col

def prepare_input(df, text_col, target_col=None):
    df = df.copy()
    df[text_col] = df[text_col].fillna("").astype(str)

    if target_col and target_col in df.columns:
        df[target_col] = df[target_col].fillna("").astype(str)
        df["model_input"] = df[target_col] + " [SEP] " + df[text_col]
    else:
        df["model_input"] = df[text_col]

    return df


def train_stance_baseline(train_df, text_col, label_col):
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

    model.fit(train_df[text_col], train_df[label_col])
    return model


def predict_stance(model, df):
    df = df.copy()
    df["stance_prediction"] = model.predict(df["model_input"])

    if hasattr(model[-1], "predict_proba"):
        probs = model.predict_proba(df["model_input"])
        df["stance_confidence"] = probs.max(axis=1)

    return df


def train_stance_mbert(train_df, text_col, label_col, model_name="bert-base-multilingual-cased"):

    labels = sorted(train_df[label_col].unique())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    train_df = train_df.copy()
    train_df["label"] = train_df[label_col].map(label2id)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = Dataset.from_pandas(train_df[[text_col, "label"]])

    def tokenize_batch(batch):
        return tokenizer(
            batch[text_col],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    dataset = dataset.map(tokenize_batch, batched=True)
    dataset = dataset.remove_columns([text_col])
    dataset.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir="tmp",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=50,
        save_strategy="no",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer
    )

    trainer.train()

    return trainer, tokenizer, id2label


def predict_stance_mbert(trainer, tokenizer, id2label, df, text_col):

    dataset = Dataset.from_pandas(df[[text_col]])

    def tokenize_batch(batch):
        return tokenizer(
            batch[text_col],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    dataset = dataset.map(tokenize_batch, batched=True)
    dataset = dataset.remove_columns([text_col])
    dataset.set_format("torch")

    outputs = trainer.predict(dataset)
    logits = outputs.predictions

    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    pred_ids = np.argmax(probs, axis=1)

    df = df.copy()
    df["stance_prediction_mbert"] = [id2label[i] for i in pred_ids]
    df["stance_confidence_mbert"] = probs.max(axis=1)

    return df

def create_summary(df, target_col=None):

    moral_cols = [
        "care_score",
        "fairness_score",
        "loyalty_score",
        "authority_score",
        "sanctity_score"
    ]

    agg_dict = {
        "total_texts": ("model_input", "count"),
        "care_score": ("care_score", "mean"),
        "fairness_score": ("fairness_score", "mean"),
        "loyalty_score": ("loyalty_score", "mean"),
        "authority_score": ("authority_score", "mean"),
        "sanctity_score": ("sanctity_score", "mean")
    }

    if target_col and target_col in df.columns:

        summary = df.groupby(target_col).agg(**agg_dict).reset_index()

        if "stance_prediction" in df.columns:
            stance_dist = pd.crosstab(
                df[target_col],
                df["stance_prediction"],
                normalize="index"
            ).reset_index()

            summary = summary.merge(
                stance_dist,
                on=target_col,
                how="left"
            )

    else:

        summary = pd.DataFrame({
            "total_texts": [len(df)],
            "care_score": [df["care_score"].mean()],
            "fairness_score": [df["fairness_score"].mean()],
            "loyalty_score": [df["loyalty_score"].mean()],
            "authority_score": [df["authority_score"].mean()],
            "sanctity_score": [df["sanctity_score"].mean()]
        })

    return summary


def write_interpretation(df, summary, output_dir):
    path = os.path.join(output_dir, "interpretation.txt")

    moral_cols = [
        "care_score",
        "fairness_score",
        "loyalty_score",
        "authority_score",
        "sanctity_score"
    ]
    total_moral = df[moral_cols].sum().sort_values(ascending=False)

    with open(path, "w", encoding="utf-8") as f:
        f.write("Stance and Moral Framing Pipeline Interpretation\n\n")

        f.write("1. Stance output\n")
        if "stance_prediction" in df.columns:
            f.write("The pipeline predicts stance labels for each text instance.\n")
            f.write("These labels should be interpreted as computational indicators, not direct proof of authorial belief.\n\n")
        else:
            f.write("No stance model was applied.\n\n")

        f.write("2. Moral framing output\n")
        f.write("The moral framing analysis counts lexical indicators associated with moral foundations.\n")
        f.write("The most frequent moral foundations are:\n\n")

        for foundation, value in total_moral.items():
            f.write(f"- {foundation}: {value}\n")

        f.write("\n3. Interpretation warning\n")
        f.write("For historical or literary texts, these outputs should be treated as exploratory evidence.\n")
        f.write("Keyword counts cannot fully capture irony, genre, quotation, indirect argument, or historical spelling variation.\n")

    return path


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--train", default=None)
    parser.add_argument("--text_col", default=None)
    parser.add_argument("--target_col", default=None)
    parser.add_argument("--label_col", default=None)
    parser.add_argument(
        "--historical_preprocess",
        action="store_true",
        help="apply basic historical spelling normalisation"
    )
    parser.add_argument(
        "--mfd2_path",
        required=True,
        help="path to MFD 2.0 dictionary .dic file"
    )

    parser.add_argument("--output", default="outputs")

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    df = load_dataset(args.input)
    detected_text_col, detected_target_col, detected_label_col = detect_columns(df)

    if args.text_col is None:
        args.text_col = detected_text_col

    if args.target_col is None:
        args.target_col = detected_target_col

    if args.label_col is None:
        args.label_col = detected_label_col

    print("Detected columns:")
    print("text_col:", args.text_col)
    print("target_col:", args.target_col)
    print("label_col:", args.label_col)

    if args.text_col is None:
        raise ValueError(
            f"Could not detect text column. Available columns: {list(df.columns)}"
        )
    df[args.text_col] = df[args.text_col].apply(
        lambda x: preprocess_text(
            x,
            historical=args.historical_preprocess
        )
    )

    df = prepare_input(df, args.text_col, args.target_col)

    if args.train and args.label_col:
        train_df = load_dataset(args.train)
        train_df = prepare_input(train_df, args.text_col, args.target_col)

        # Logistic Regression baseline
        lr_model = train_stance_baseline(
            train_df,
            "model_input",
            args.label_col
        )

        df = predict_stance(lr_model, df)
        df = df.rename(columns={
            "stance_prediction": "stance_prediction_lr",
            "stance_confidence": "stance_confidence_lr"
        })

        # mBERT model
        mbert_trainer, mbert_tokenizer, id2label = train_stance_mbert(
            train_df,
            "model_input",
            args.label_col
        )

        df = predict_stance_mbert(
            mbert_trainer,
            mbert_tokenizer,
            id2label,
            df,
            "model_input"
        )

        if args.label_col in df.columns:
            lr_f1 = f1_score(
                df[args.label_col],
                df["stance_prediction_lr"],
                average="macro"
            )

            mbert_f1 = f1_score(
                df[args.label_col],
                df["stance_prediction_mbert"],
                average="macro"
            )

            if mbert_f1 >= lr_f1:
                best_model = "mbert"
                df["stance_prediction"] = df["stance_prediction_mbert"]
                df["stance_confidence"] = df["stance_confidence_mbert"]
            else:
                best_model = "logistic_regression"
                df["stance_prediction"] = df["stance_prediction_lr"]
                df["stance_confidence"] = df["stance_confidence_lr"]

            with open(os.path.join(args.output, "model_comparison.txt"), "w", encoding="utf-8") as f:
                f.write("Model comparison based on macro-F1\n\n")
                f.write(f"Logistic Regression macro-F1: {lr_f1:.4f}\n")
                f.write(f"mBERT macro-F1: {mbert_f1:.4f}\n")
                f.write(f"Selected model: {best_model}\n")

        else:
            # 如果 input 数据没有真实标签，就默认用 mBERT
            df["stance_prediction"] = df["stance_prediction_mbert"]
            df["stance_confidence"] = df["stance_confidence_mbert"]
    df = add_mfd2_features(df,
                           args.text_col,
                           args.mfd2_path)
    df["care_score"] = df["care_virtue"] - df["care_vice"]
    df["fairness_score"] = df["fairness_virtue"] - df["fairness_vice"]
    df["loyalty_score"] = df["loyalty_virtue"] - df["loyalty_vice"]
    df["authority_score"] = df["authority_virtue"] - df["authority_vice"]
    df["sanctity_score"] = df["sanctity_virtue"] - df["sanctity_vice"]

    score_cols = [
        "care_score",
        "fairness_score",
        "loyalty_score",
        "authority_score",
        "sanctity_score"
    ]

    # 最大绝对 moral score
    df["max_moral_score"] = (
        df[score_cols]
        .abs()
        .max(axis=1)
    )

    # 找到最大的 foundation
    df["dominant_foundation"] = (
        df[score_cols]
        .abs()
        .idxmax(axis=1)
    )

    df["dominant_foundation"] = (
        df["dominant_foundation"]
        .str.replace("_score", "")
    )

    # 如果全部都是0，说明没有检测到 moral foundation
    df.loc[
        df["max_moral_score"] == 0,
        "dominant_foundation"
    ] = "none"

    df.drop(
        columns=["max_moral_score"],
        inplace=True
    )

    summary = create_summary(df, args.target_col)

    df.to_csv(os.path.join(args.output, "predictions_and_moral_features.csv"), index=False)
    summary.to_csv(os.path.join(args.output, "summary_by_target.csv"), index=False)

    if args.label_col and args.label_col in df.columns and "stance_prediction" in df.columns:
        report = classification_report(df[args.label_col], df["stance_prediction"])
        labels = sorted(df[args.label_col].unique())

        cm = confusion_matrix(
            df[args.label_col],
            df["stance_prediction"],
            labels=labels
        )

        cm_df = pd.DataFrame(
            cm,
            index=[f"True_{x}" for x in labels],
            columns=[f"Pred_{x}" for x in labels]
        )

        cm_df.to_csv(
            os.path.join(
                args.output,
                "confusion_matrix.csv"
            )
        )
        with open(os.path.join(args.output, "stance_evaluation.txt"), "w", encoding="utf-8") as f:
            f.write(report)

        errors = df[df[args.label_col] != df["stance_prediction"]]
        errors.to_csv(os.path.join(args.output, "error_analysis.csv"), index=False)

    write_interpretation(df, summary, args.output)

    print("Pipeline completed.")
    print(f"Outputs saved to: {args.output}")


if __name__ == "__main__":
    main()
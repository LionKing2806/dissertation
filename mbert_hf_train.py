
print("script started")

import os
print("import os ok")

import jsonlines
print("import jsonlines ok")

import torch
print("import torch ok")

from torch.utils.data import Dataset
print("import Dataset ok")

from sklearn.metrics import f1_score
print("import sklearn ok")

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
print("import transformers ok", flush=True)

DATA_DIR = "data/xstance-data-v1.0"
MODEL_NAME = "bert-base-multilingual-cased"
OUTPUT_DIR = "mbert_hf_model"
PRED_PATH = "predictions/mbert_hf_pred.jsonl"

LABEL2ID = {
    "AGAINST": 0,
    "FAVOR": 1,
}

ID2LABEL = {
    0: "AGAINST",
    1: "FAVOR",
}


def load_jsonl(path):
    data = []
    with jsonlines.open(path) as reader:
        for item in reader:
            data.append(item)
    return data


class XStanceDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512, has_labels=True):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.has_labels = has_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        encoding = self.tokenizer(
            item["question"],
            item["comment"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        sample = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
        }

        if self.has_labels:
            sample["labels"] = torch.tensor(LABEL2ID[item["label"]], dtype=torch.long)

        return sample


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)

    macro_f1 = f1_score(labels, preds, average="macro")
    accuracy = (preds == labels).mean()

    return {
        "macro_f1": macro_f1,
        "accuracy": accuracy,
    }


def write_predictions(predictions, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with jsonlines.open(output_path, "w") as writer:
        for pred in predictions:
            label = ID2LABEL[int(pred)]
            writer.write({"label": label})


def main():
    print("main started", flush=True)

    train_path = os.path.join(DATA_DIR, "train.jsonl")
    valid_path = os.path.join(DATA_DIR, "valid.jsonl")
    test_path = os.path.join(DATA_DIR, "test.jsonl")

    print(train_path, flush=True)
    print(valid_path, flush=True)
    print(test_path, flush=True)

    train_data = load_jsonl(train_path)
    print("train loaded", len(train_data), flush=True)

    valid_data = load_jsonl(valid_path)
    print("valid loaded", len(valid_data), flush=True)

    test_data = load_jsonl(test_path)
    print("test loaded", len(test_data), flush=True)

    # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME) Use this if wish to train again
    tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-multilingual-cased"
    )
    train_dataset = XStanceDataset(train_data, tokenizer, has_labels=True)
    valid_dataset = XStanceDataset(valid_data, tokenizer, has_labels=True)
    test_dataset = XStanceDataset(test_data, tokenizer, has_labels=False)

    # model = AutoModelForSequenceClassification.from_pretrained(
    #     MODEL_NAME,
    #     num_labels=2,
    #     id2label=ID2LABEL,
    #     label2id=LABEL2ID,
    # )  Use this if wish to train again
    model = AutoModelForSequenceClassification.from_pretrained(
        "mbert_hf_model/checkpoint-17115"
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        warmup_ratio=0.1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    predictions = trainer.predict(test_dataset)
    pred_ids = predictions.predictions.argmax(axis=-1)

    write_predictions(pred_ids, PRED_PATH)

    print(f"Saved predictions to {PRED_PATH}")


print("before direct main")
if __name__ == "__main__":
    main()
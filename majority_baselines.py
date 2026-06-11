import argparse
import jsonlines
from collections import Counter, defaultdict


def load_jsonl(path):
    with jsonlines.open(path) as reader:
        return list(reader)


def write_predictions(predictions, output_path):
    with jsonlines.open(output_path, "w") as writer:
        for label in predictions:
            writer.write({"label": label})


def global_majority(train_data):
    labels = [x["label"] for x in train_data]
    return Counter(labels).most_common(1)[0][0]


def target_wise_majority(train_data):
    counts = defaultdict(Counter)

    for x in train_data:
        qid = x["question_id"]
        label = x["label"]
        counts[qid][label] += 1

    majority_by_question = {}
    for qid, counter in counts.items():
        majority_by_question[qid] = counter.most_common(1)[0][0]

    return majority_by_question


def predict_global(train_data, test_data):
    majority_label = global_majority(train_data)
    return [majority_label for _ in test_data]


def predict_target_wise(train_data, test_data):
    global_label = global_majority(train_data)
    majority_by_question = target_wise_majority(train_data)

    predictions = []
    for x in test_data:
        qid = x["question_id"]

        if qid in majority_by_question:
            predictions.append(majority_by_question[qid])
        else:
            predictions.append(global_label)

    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--mode", required=True, choices=["global", "target-wise"])
    parser.add_argument("--pred", required=True)
    args = parser.parse_args()

    train_data = load_jsonl(args.train)
    test_data = load_jsonl(args.test)

    if args.mode == "global":
        predictions = predict_global(train_data, test_data)
    else:
        predictions = predict_target_wise(train_data, test_data)

    write_predictions(predictions, args.pred)
    print(f"Saved predictions to {args.pred}")

    import subprocess

    subprocess.run([
        "python",
        "evaluate.py",
        "--gold",
        "data\\xstance-data-v1.0\\test.jsonl",
        "--pred",
        args.pred
    ])


if __name__ == "__main__":
    main()
# X-STANCE Reproduction

This repository contains a reproduction of several baseline methods from the X-STANCE stance detection benchmark.

## Dataset

The experiments use the X-STANCE dataset:

* German (DE)
* French (FR)
* Italian (IT)

The dataset is included in:

```text
data/xstance-data-v1.0/
```

## Implemented Baselines

### 1. Majority Class (Global)

Predicts the globally most frequent stance label in the training set.

Script:

```text
majority_baselines.py
```

Mode:

```text
global
```

### 2. Majority Class (Target-wise)

Predicts the most frequent label for each target question.

Script:

```text
majority_baselines.py
```

Mode:

```text
target-wise
```

### 3. FastText Baseline

Implementation based on the original X-STANCE FastText baseline.

Location:

```text
fasttext_baseline/
```

Main script:

```text
fasttext_baseline/run.py
```

### 4. Multilingual BERT (mBERT)

The original repository used AllenNLP. Due to compatibility issues with newer Python versions, the mBERT baseline was reimplemented using HuggingFace Transformers.

Script:

```text
mbert_hf_train.py
```

Model:

```text
bert-base-multilingual-cased
```

## Evaluation

Evaluation follows the original X-STANCE evaluation procedure.

Script:

```text
evaluate.py
```

Example:

```bash
python evaluate.py --gold data/xstance-data-v1.0/test.jsonl --pred predictions/mbert_hf_pred.jsonl
```

## Prediction Files

Example prediction outputs are stored in:

```text
predictions/
```

Including:

```text
majority_global.jsonl
majority_targetwise.jsonl
fasttext_pred.jsonl
mbert_hf_pred.jsonl
```

## Repository Structure

```text
.
├── data/
├── fasttext_baseline/
├── mbert_baseline/
├── predictions/
├── evaluate.py
├── majority_baselines.py
├── mbert_hf_train.py
├── README.md
└── LICENSE
```

## Notes

* Majority baselines and FastText follow the original X-STANCE implementation.
* The mBERT baseline was reproduced using HuggingFace Transformers instead of AllenNLP.
* Trained mBERT checkpoints are not included in this repository.

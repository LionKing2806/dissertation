## Disclaimer

This repository is a reproduction of the X-STANCE benchmark introduced by Vamvas and Sennrich (2020). The original dataset, benchmark design and baseline methods belong to the original authors.

Original repository:
https://github.com/ZurichNLP/xstance

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
code/fasttext_baseline/
```

Main script:

```text
code/fasttext_baseline/run.py
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
python evaluate.py --gold data/xstance-data-v1.0/test.jsonl --pred outputs/predictions/mbert_hf_pred.jsonl
```

## Prediction Files

Example prediction outputs are stored in:

```text
outputs/predictions/
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
├── code/
│   └── fasttext_baseline/
├── data/
│   └── xstance-data-v1.0/
├── mbert_baseline/
├── mbert_hf_model/
├── notes/
│   ├── Evaluation of the X-STANCE Reproduction.md
│   └── Project Log – Week 1.md
├── outputs/
│   └── predictions/
├── figures/
├── dissertation/
├── evaluate.py
├── majority_baselines.py
├── mbert_hf_train.py
├── README.md
└── LICENSE
```
## Environment

For Majority, FastText and Evaluation:

```bash
pip install -r requirements_xstance.txt
```

For mBERT:

```bash
pip install -r requirements_mbert.txt
```
## Quick Start

### Majority Global

```bash
python majority_baselines.py --mode global
```

### Majority Target-wise

```bash
python majority_baselines.py --mode target-wise
```

### FastText

```bash
python code/fasttext_baseline/run.py
```

### mBERT

```bash
python mbert_hf_train.py
```

### Evaluation

```bash
python evaluate.py --gold data/xstance-data-v1.0/test.jsonl --pred outputs/predictions/mbert_hf_pred.jsonl
```
## Notes

* Majority baselines and FastText follow the original X-STANCE implementation.
* The mBERT baseline was reproduced using HuggingFace Transformers instead of AllenNLP.
* Trained mBERT checkpoints are not included in this repository.


## References

Vamvas, J. and Sennrich, R. (2020). X-STANCE: A Multilingual Multi-Target Dataset for Stance Detection. Proceedings of the 5th Swiss Text Analytics Conference (SwissText) and 16th Conference on Natural Language Processing (KONVENS).




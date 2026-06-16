# Project Log – Week 1

**Project:** Stance, Argument, and Moral Framing in the Works of Daniel Defoe: A Computational Text Analysis

**Reporting Period:** 09 June 2026 – 15 June 2026

## Summary

The primary objective of Week 1 was to establish a foundation for the project by reviewing the X-STANCE benchmark, understanding the principles of stance detection, and creating a reproducible experimental workflow that can later support analysis of the Defoe corpus.

During this week, I focused on reading introductory material, setting up the development environment, reproducing benchmark models, and organising the project repository.

---

## Key Decisions

### 1. Reproduction of X-STANCE Baselines

The following baseline models were successfully reproduced:

* Majority Class (Global)
* Majority Class (Target-wise)
* FastText
* Multilingual BERT (mBERT)

The purpose of reproducing these models was to gain practical experience with stance detection workflows, including data preparation, model training, prediction generation, and evaluation using benchmark metrics.

The reproduction process also provided an opportunity to compare simple baselines with stronger neural approaches and to understand the role of benchmark evaluation in NLP research.

---

### 2. Adoption of HuggingFace Transformers for mBERT

The original X-STANCE implementation used AllenNLP for multilingual BERT training and evaluation.

During reproduction, significant dependency and compatibility issues were encountered due to the age of the AllenNLP framework and its supporting libraries. Rather than attempting to recreate an outdated software environment, the decision was made to implement the mBERT baseline using HuggingFace Transformers.

This decision was made for several reasons:

* HuggingFace Transformers is actively maintained.
* It is widely used in contemporary NLP research.
* It provides direct support for multilingual BERT models.
* It improves reproducibility and long-term maintainability.
* Future experiments can be implemented using the same framework.

The methodological principle of the original experiment remains unchanged while the implementation becomes more practical and sustainable.

---

### 3. Repository Organisation

A GitHub repository was created to support reproducible research and version control.

The repository currently includes:

* dataset files and documentation;
* baseline implementations;
* evaluation scripts;
* prediction outputs;
* experiment notes and supporting materials.

The repository will serve as the central location for all code, outputs, notes and future dissertation-related work.

---

## Methodological Role of X-STANCE

X-STANCE is being used as a benchmark dataset for evaluating stance detection methods.

Its main value lies in the availability of:

* labelled stance data;
* predefined train/test splits;
* established evaluation metrics;
* published baseline results.

These properties allow different stance detection approaches to be tested and compared under controlled conditions.

However, X-STANCE should not be viewed as a direct representation of the final research problem.

The dataset consists of modern political comments written by election candidates in response to explicit political questions. In contrast, the Defoe project focuses on eighteenth-century historical and literary texts, which differ substantially in language, style, genre and interpretive context.

As a result, successful reproduction of X-STANCE demonstrates that stance detection methods can be implemented and evaluated correctly, but it does not demonstrate that the same methods will automatically transfer to historical literary texts.

The reproduction therefore serves as a methodological evaluation exercise and as preparation for later work on the Defoe corpus.

---

## Progress Achieved

By the end of Week 1:

* introductory reading on stance detection had been completed;
* the X-STANCE dataset structure had been examined;
* baseline models had been reproduced;
* a HuggingFace-based workflow had been established;
* a GitHub repository had been created and populated;
* an initial glossary of key project terminology had been started.

---

## Reflection

Week 1 successfully established the technical and methodological foundations of the project.

The most important outcome was recognising the distinction between benchmark stance classification and the analysis of historical literary texts. While benchmark datasets provide labelled examples and quantitative evaluation metrics, the Defoe corpus will require greater attention to transferability, interpretability and historical context.

Future work will therefore focus not only on model performance but also on assessing whether computational methods produce meaningful and defensible interpretations when applied to historical texts.

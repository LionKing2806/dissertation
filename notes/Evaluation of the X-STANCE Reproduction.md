# Evaluation of the X-STANCE Reproduction

The reproduction of the X-STANCE baseline models demonstrates that the experimental pipeline for stance detection is functioning correctly and that established stance detection methods can be successfully applied to a benchmark dataset.

By reproducing the Majority Class, FastText and mBERT baselines, I gained practical experience with dataset preparation, model training, prediction generation and evaluation using standard metrics such as F1 score. The reproduction also provided a useful comparison between simple baseline methods and more advanced transformer-based approaches, showing why models such as mBERT are widely used for stance detection research.

X-STANCE itself is a valuable benchmark because it contains labelled examples, clear train/test splits and a well-defined stance classification task.

## What the Reproduction Demonstrates

The reproduction demonstrates that:

* the experimental workflow can be implemented successfully;
* benchmark stance detection models can be trained and evaluated correctly;
* standard evaluation procedures can be reproduced;
* different modelling approaches can be compared within a controlled environment;
* transformer-based approaches generally outperform simpler baselines on the benchmark task.

The exercise therefore provides confidence that the computational pipeline is functioning as intended and can be used for further methodological experiments.

## What the Reproduction Cannot Demonstrate

The reproduction cannot directly demonstrate that the same methods will work equally well on the Defoe project.

X-STANCE consists of modern political comments written by election candidates in response to clearly defined political questions, and the stance labels are already provided by the dataset creators.

In contrast, the Defoe corpus contains eighteenth-century historical and literary texts, which differ substantially in language, style, genre and purpose. The Defoe texts are not accompanied by predefined stance labels, and many expressions of belief, argument and morality may be indirect, ambiguous or context-dependent.

The project therefore involves interpretive analysis rather than straightforward benchmark classification.

## Relevance to the Defoe Project

For this reason, X-STANCE should be viewed primarily as a methodological testbed rather than a direct representation of the final research problem.

The reproduction demonstrates that stance detection models can be implemented, evaluated and compared in a controlled benchmark setting. It does not demonstrate that these models will automatically transfer to historical literary texts.

Additional work will be required to assess transferability, interpretability and suitability for analysing the writings of Daniel Defoe.

## Conclusion

The X-STANCE reproduction successfully establishes a reproducible benchmark workflow and provides practical experience with modern stance detection methods. However, benchmark performance alone is insufficient evidence that the same approaches will be effective for eighteenth-century literary texts.

Consequently, the reproduction should be regarded as a methodological evaluation exercise and a foundation for later experimentation rather than a direct solution to the Defoe research problem.

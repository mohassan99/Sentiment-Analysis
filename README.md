# PSL Project 3 — Movie Review Sentiment Analysis

**Course:** CS 598 Practical Statistical Learning, University of Illinois Urbana-Champaign  
**Term:** Fall 2022  
**Author:** Mohammad Hassanpour (`mohassan99`)

---

## Overview

Built a binary sentiment classifier for IMDB movie reviews using **Lasso logistic regression** on a bag-of-words representation. The core challenge was vocabulary design: constructing a compact, predictive feature set from 50,000 reviews that generalizes to held-out data under a vocabulary size constraint.

Key results:
- Vocabulary pruned from raw token space to a high-signal subset using TF-IDF filtering and stop-word removal
- Lasso logistic regression (via `glmnet`) used for simultaneous feature selection and classification
- AUC evaluated on held-out test splits; model tuned via cross-validation on λ

---

## Repository Structure

```
├── Data & Code/
│   ├── buildVocab.html              # Vocabulary construction notebook (rendered)
│   ├── Report - Project 3.pdf       # Final project report
│   ├── buildVocab.Rmd               # Vocabulary construction source
│   ├── mymain.R                     # Main prediction script
│   ├── eval.R                       # Evaluation script
│   ├── gen5sets.R                   # 5-fold split generator
│   └── myvocab.txt                  # Final pruned vocabulary
└── README.md
```

---

## Methods

### Vocabulary Construction (`buildVocab`)
- Tokenized 25,000 training reviews using `tm` package
- Applied lowercasing, punctuation removal, stop-word filtering, and stemming
- Pruned by document frequency bounds to eliminate rare/ubiquitous tokens
- Final vocabulary: ≤ 2,000 terms selected for maximum discriminative signal

### Sentiment Classification (`mymain.R`)
- Constructed Document-Term Matrix (DTM) from pruned vocabulary
- Fitted Lasso logistic regression (`glmnet`, `family = "binomial"`) with 10-fold CV
- Optimal λ selected by cross-validated AUC
- Final model achieves competitive AUC on IMDB held-out test set

---

## Tech Stack

`R` · `glmnet` · `tm` · `SnowballC` · `pROC` · `Matrix`

# Classification of Legal Text: US Supreme Court Opinions

UE24CS352A Machine Learning, Mini-Project 5, Team C-26 (Section C), PES University

- Kshitij Satish Shetty (PES1UG24AM143)
- Kushal G (PES1UG24AM145)

## Problem

Given the text of a US Supreme Court opinion, predict its **issue area** (for example Criminal Procedure, Civil Rights, Federal Taxation), the legal topic assigned by expert coders in the Supreme Court Database. The assigned reference is the CS229 (2020) report by K. Iyer, which compares LDA + Logistic Regression, Doc2Vec + Logistic Regression and BERT.

## Dataset

Supreme Court Database as packaged by the textacy project: 8419 opinions (1946-2016) with full text and expert labels. The notebook downloads it automatically from the textacy-data GitHub release (about 111 MB), so nothing needs to be uploaded.

After cleaning (very short texts, unlabeled rows, exact duplicates and classes with fewer than 15 opinions removed) there are 8328 documents in 13 classes. Only the first 30000 characters of each opinion are used. The split is stratified: 5829 train / 1249 validation / 1250 test (seed 42).

## Approach

- Features: TF-IDF on single words, fitted on the training split only.
- Models compared: majority-class baseline, Logistic Regression, Linear SVM, Complement Naive Bayes, and the reference paper's models re-implemented on the same split (LDA topics + Logistic Regression, Doc2Vec + Logistic Regression).
- Model selection by macro-F1 on the validation set. The test set is used once at the end.
- Leakage checks: split before fitting, no identical text across splits, shuffled-label test; Doc2Vec and LDA are trained on the training split only.
- Robustness: 5-fold cross-validation, error analysis, a fine-grained issue-code experiment, and a similar-opinion search (Roe v. Wade) as in the reference paper.

## Results (test set)

| model | accuracy | macro_f1 | weighted_f1 |
|---|---|---|---|
| Majority baseline | 0.2304 | 0.0288 | 0.0863 |
| LinearSVC C=0.5 | 0.8176 | 0.7672 | 0.8156 |
| LDA(25 topics) + LogReg | 0.5816 | 0.4472 | 0.6118 |
| Doc2Vec + LogReg (C=1) | 0.6696 | 0.5843 | 0.6761 |

Selected model: **LinearSVC C=0.5**. 5-fold cross-validation of LinearSVC C=0.5 on train+validation: accuracy 0.808 +- 0.020, macro-F1 0.746 +- 0.027.

Fine-grained task (141 issue codes, 7434 opinions, 80/20 split): accuracy 0.707, top-5 accuracy 0.931, macro-F1 0.656 (majority class: 0.032).

Full per-class report, confusion matrix, top confusions, cross-validation scores and top words per class are in `results/`.

## Repository contents

| Path | What it is |
|---|---|
| `Legal_Text_Classification_MiniProject.ipynb` | Full pipeline: data, cleaning, split, features, models, evaluation, export |
| `app.py` | Streamlit demo |
| `model/` | Saved pipeline (`legal_clf.joblib`), label names, config, held-out sample opinions |
| `results/` | Metrics tables, classification report, confusion matrix, cross-validation, similar-opinion lists, top terms |
| `requirements.txt` | Pinned versions for running the app |

## How to run

### 1. Train (Google Colab, CPU is enough)

1. Open `Legal_Text_Classification_MiniProject.ipynb` in Google Colab (File > Upload notebook).
2. Choose Runtime > Run all (roughly 20-30 minutes; Doc2Vec is the slowest part). The first cell installs `streamlit` and `gensim`; if Colab asks to restart the runtime, restart and run all again. The last cell starts an optional tunnel for the demo, so it is fine to stop it.
3. The notebook writes `model/`, `results/`, `app.py`, `requirements.txt` and this README.

### 2. Run the Streamlit demo locally

```
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
python -m streamlit run app.py
```

Open http://localhost:8501. Pick a held-out sample or paste an opinion, then click **Classify**. The app shows the top three predicted issue areas and the words that pushed the prediction most. The virtual environment is not part of the repository (see `.gitignore`).

Note: the scores shown are a softmax of the SVM decision values, so they rank the classes but are not calibrated probabilities. The saved model was trained with the scikit-learn version pinned in `requirements.txt`, so use those versions.

## Limitations

The headline numbers come from a single split (cross-validation above shows the spread). The smallest classes have only a handful of test documents, so their per-class scores are noisy. Some top-weighted words are case names or numbers, so the model partly relies on case-specific vocabulary. Only the first part of each opinion is used, and no transformer model was trained (CPU-only; the reference paper's BERT run also ran out of memory). Our label set differs slightly from the paper's, so comparisons with its numbers are indicative.

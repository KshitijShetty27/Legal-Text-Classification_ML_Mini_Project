import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(page_title="Legal Text Classifier", layout="wide")


@st.cache_resource
def load():
    pipe = joblib.load("model/legal_clf.joblib")
    names = {int(k): v for k, v in json.load(open("model/label_names.json")).items()}
    cfg = json.load(open("model/config.json"))
    samples = pd.read_csv("model/samples.csv")
    return pipe, names, cfg, samples


pipe, names, cfg, samples = load()
tfidf, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]

st.title("US Supreme Court Opinion Classifier")
st.caption(f"TF-IDF + {cfg['best_model']}. Predicts the issue area of a court opinion.")

options = ["(paste my own text)"] + [
    f"Held-out sample {i + 1} (true label: {r.true_label})" for i, r in samples.iterrows()
]
choice = st.selectbox("Start from a held-out test opinion, or paste your own", options)
default = "" if choice == options[0] else samples.loc[options.index(choice) - 1, "text"]
text = st.text_area("Opinion text", value=default, height=300)

if st.button("Classify") and text.strip():
    X = tfidf.transform([text[: cfg["max_chars"]]])
    if hasattr(clf, "predict_proba"):
        scores, kind = clf.predict_proba(X)[0], "Probability"
    else:
        d = clf.decision_function(X)[0]
        e = np.exp(d - d.max())
        scores, kind = e / e.sum(), "Relative score"

    order = np.argsort(scores)[::-1][:3]
    st.subheader(f"Predicted issue area: {names[int(clf.classes_[order[0]])]}")
    st.table(pd.DataFrame({
        "Issue area": [names[int(clf.classes_[i])] for i in order],
        kind: [round(float(scores[i]), 3) for i in order],
    }))

    coef = getattr(clf, "coef_", None)
    if coef is not None:
        contrib = X.multiply(coef[order[0]]).toarray().ravel()
        top = np.argsort(contrib)[::-1][:10]
        terms = tfidf.get_feature_names_out()[top]
        st.write("Words in this text that pushed the prediction most:", ", ".join(terms))

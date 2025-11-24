# ridge_text_fast.py
# Lean Ridge baseline for ensembling (no CV, tuned for speed)

import os, joblib, numpy as np, pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from price_features_transformer import clean_text, build_numeric_table, smape

SEED = 42
np.random.seed(SEED)

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "dataset")
OUT  = os.path.join(BASE, "models/ridge_fast")
os.makedirs(OUT, exist_ok=True)

def main():
    df = pd.read_csv(os.path.join(DATA, "train.csv"))
    texts = df["catalog_content"].fillna("").map(clean_text).tolist()
    y = np.log1p(df["price"].astype(float).values)

    # 90/10 split once, no CV
    split = int(0.9 * len(df))
    X_tr_texts, X_val_texts = texts[:split], texts[split:]
    y_tr, y_val = y[:split], y[split:]

    # TF-IDF: lighter config
    tfw = TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=3,
                          max_features=100_000, sublinear_tf=True)
    tfc = TfidfVectorizer(analyzer="char", ngram_range=(3,4), min_df=5,
                          max_features=80_000, sublinear_tf=True)
    Xw_tr, Xw_val = tfw.fit_transform(X_tr_texts), tfw.transform(X_val_texts)
    Xc_tr, Xc_val = tfc.fit_transform(X_tr_texts), tfc.transform(X_val_texts)
    X_tr, X_val = hstack([Xw_tr, Xc_tr]), hstack([Xw_val, Xc_val])

    # Numeric features (optional small boost)
    num_tr = build_numeric_table(pd.Series(X_tr_texts))
    num_val = build_numeric_table(pd.Series(X_val_texts))

    # keep only numeric columns
    num_cols = [c for c in num_tr.columns if pd.api.types.is_numeric_dtype(num_tr[c])]
    if len(num_cols) == 0:
        # fallback: simple text length features
        num_tr = pd.DataFrame({
            "len_chars": pd.Series(X_tr_texts).str.len().astype(float),
            "len_words": pd.Series(X_tr_texts).str.split().apply(len).astype(float)
        })
        num_val = pd.DataFrame({
            "len_chars": pd.Series(X_val_texts).str.len().astype(float),
            "len_words": pd.Series(X_val_texts).str.split().apply(len).astype(float)
        })
    else:
        num_tr = num_tr[num_cols].fillna(0)
        num_val = num_val[num_cols].reindex(columns=num_cols).fillna(0)

    scaler = StandardScaler(with_mean=False)
    Xn_tr = scaler.fit_transform(num_tr)
    Xn_val = scaler.transform(num_val)


    X_tr = hstack([X_tr, Xn_tr]).tocsr()
    X_val = hstack([X_val, Xn_val]).tocsr()

    # Ridge — single alpha
    model = Ridge(alpha=3.0, random_state=SEED)
    model.fit(X_tr, y_tr)

    y_pred = np.expm1(model.predict(X_val)).clip(0.99, None)
    y_true = np.expm1(y_val)
    print(f"[FastRidge] SMAPE: {smape(y_true, y_pred):.2f}%")

    # Train on full data with same settings
    # Build numeric table from full text
    num_full = build_numeric_table(pd.Series(texts))

    # keep only numeric columns
    num_cols = [c for c in num_full.columns if pd.api.types.is_numeric_dtype(num_full[c])]
    if len(num_cols) == 0:
        # fallback: text length features
        num_full = pd.DataFrame({
            "len_chars": pd.Series(texts).str.len().astype(float),
            "len_words": pd.Series(texts).str.split().apply(len).astype(float)
        })
    else:
        num_full = num_full[num_cols].fillna(0)

    scaler = StandardScaler(with_mean=False)
    Xn = scaler.fit_transform(num_full)

    Xw = tfw.fit_transform(texts)
    Xc = tfc.fit_transform(texts)
    X_full = hstack([Xw, Xc, Xn]).tocsr()

    model.fit(X_full, y)
    joblib.dump(model, os.path.join(OUT, "ridge.pkl"))
    joblib.dump(tfw, os.path.join(OUT, "tfidf_word.pkl"))
    joblib.dump(tfc, os.path.join(OUT, "tfidf_char.pkl"))
    joblib.dump(scaler, os.path.join(OUT, "scaler.pkl"))
    print(f"[FastRidge] Model saved to {OUT}")

if __name__ == "__main__":
    main()

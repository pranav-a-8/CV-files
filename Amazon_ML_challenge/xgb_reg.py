# xgboost_reg.py
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from price_features_transformer import build_numeric_table, clean_text, smape
import xgboost as xgb

SEED = 42
np.random.seed(SEED)
# ---- CONFIGURATION ----
# Mode set to 'log_price' as requested
MODE = 'log_price'

def main():
    """
    Main function to train the price prediction model using XGBoost.
    1. Trains on a 90/10 split to find the best iteration and validate performance.
    2. Retrains on the full training data using the optimal parameters.
    3. Saves the final model to the 'models/XGBoost' directory.
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Make sure to update this path to your actual data directory
    DATA_DIR = os.path.join(BASE_DIR, "dataset")
    MODELS_DIR = os.path.join(BASE_DIR, "models/XGBoost")

    # Create models directory if it doesn't exist
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        print(f"Created directory: {MODELS_DIR}")

    print("Loading and preparing data...")
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))

    # Prepare text data and structured numeric features
    texts = df["catalog_content"].fillna("").map(clean_text)
    numeric_features = build_numeric_table(df["catalog_content"].fillna(""))

    # Define total price and pack size
    total_price = df["price"].astype(float).values
    pack_size = numeric_features["pack_size"].clip(1).values

    # Define target based on MODE
    if MODE == 'log_price':
        y = np.log1p(total_price)
    else:
        raise ValueError(f"This script is configured for 'log_price' mode only. Found: {MODE}")

    # ---------------- Step 1: Train on 90/10 split for validation ----------------
    print("\n--- Step 1: Training on 90/10 split for validation ---")
    # 90/10 split
    train_idx, val_idx = train_test_split(np.arange(len(df)), test_size=0.1, random_state=SEED)
    X_num_train = numeric_features.iloc[train_idx].copy()
    X_num_val   = numeric_features.iloc[val_idx].copy()
    pack_size_val = X_num_val["pack_size"].clip(1).values
    y_train = y[train_idx]
    y_val   = y[val_idx]

    # Identify categorical feature columns - encode them for XGBoost
    cat_features = ["unit", "canon_unit", "brand", "category_bucket", "sub_category", "flavor_profile"]
    
    # Label encode categorical features for XGBoost
    cat_mappings = {}
    for col in cat_features:
        X_num_train[col] = X_num_train[col].astype(str).fillna('missing')
        X_num_val[col] = X_num_val[col].astype(str).fillna('missing')
        
        # Create mapping
        unique_vals = pd.concat([X_num_train[col], X_num_val[col]]).unique()
        cat_mappings[col] = {val: idx for idx, val in enumerate(unique_vals)}
        
        # Apply mapping
        X_num_train[col] = X_num_train[col].map(cat_mappings[col])
        X_num_val[col] = X_num_val[col].map(cat_mappings[col])

    # Text features: TF-IDF + SVD
    print("Generating text features (TF-IDF + SVD)...")
    tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=2, max_features=200_000)
    X_text_train = tfidf.fit_transform(texts.iloc[train_idx].tolist())
    X_text_val   = tfidf.transform(texts.iloc[val_idx].tolist())

    svd = TruncatedSVD(n_components=128, random_state=SEED)
    X_svd_train = svd.fit_transform(X_text_train)
    X_svd_val   = svd.transform(X_text_val)

    svd_cols = [f"text_comp{i}" for i in range(X_svd_train.shape[1])]
    X_svd_train_df = pd.DataFrame(X_svd_train, columns=svd_cols)
    X_svd_val_df   = pd.DataFrame(X_svd_val, columns=svd_cols)

    X_train = pd.concat([X_num_train.reset_index(drop=True), X_svd_train_df.reset_index(drop=True)], axis=1)
    X_val   = pd.concat([X_num_val.reset_index(drop=True), X_svd_val_df.reset_index(drop=True)], axis=1)

    # Convert to DMatrix for XGBoost (GPU compatible)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Set up and train the XGBoost regressor for validation
    print("Training validation model on GPU...")
    
    params = {
        'objective': 'reg:absoluteerror',  # MAE loss
        'eval_metric': 'mae',
        'tree_method': 'hist',
        'device': 'cuda',  # GPU training
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'colsample_bylevel': 0.8,
        'min_child_weight': 3,
        'alpha': 0.1,  # L1 regularization
        'lambda': 1.0,  # L2 regularization
        'seed': SEED,
        'verbosity': 1
    }
    
    evals = [(dtrain, 'train'), (dval, 'valid')]
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=10000,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=100
    )

    # Predict on validation set and convert to total price scale
    print("Evaluating validation model...")
    raw_pred = model.predict(dval)
    # Convert log-price predictions back to price
    y_pred = np.expm1(raw_pred).clip(0.99, None)
    y_true = total_price[val_idx]

    val_smape = smape(y_true, y_pred)
    print(f"\nValidation SMAPE: {val_smape:.2f}%")
    print(f"Best iteration found: {model.best_iteration}")


    # ---------------- Step 2: Retrain on full train.csv and save model ----------------
    print("\n--- Step 2: Retraining on full dataset ---")

    # Prepare full dataset
    X_num_full = numeric_features.copy()
    
    # Label encode categorical features for full dataset
    for col in cat_features:
        X_num_full[col] = X_num_full[col].astype(str).fillna('missing')
        X_num_full[col] = X_num_full[col].map(cat_mappings[col]).fillna(-1).astype(int)

    print("Generating text features for the full dataset...")
    X_text_full = tfidf.fit_transform(texts.tolist())
    X_svd_full = svd.fit_transform(X_text_full)
    X_svd_full_df = pd.DataFrame(X_svd_full, columns=svd_cols)
    X_full = pd.concat([X_num_full.reset_index(drop=True), X_svd_full_df.reset_index(drop=True)], axis=1)

    y_full = y  # Already defined as np.log1p(total_price)

    # Convert to DMatrix
    dfull = xgb.DMatrix(X_full, label=y_full)

    # Train for the optimal number of iterations found in the validation step
    final_iterations = model.best_iteration
    print(f"Training final model on GPU for {final_iterations} iterations...")
    
    final_model = xgb.train(
        params,
        dfull,
        num_boost_round=final_iterations,
        evals=[(dfull, 'train')],
        verbose_eval=100
    )

    # Save the final model
    model_path = os.path.join(MODELS_DIR, 'xgboost_log_price_model.json')
    final_model.save_model(model_path)
    print(f"\nModel successfully trained and saved to: {model_path}")


if __name__ == "__main__":
    main()
# lgbm_reg.py
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from price_features_transformer import build_numeric_table, clean_text, smape
import lightgbm as lgb

SEED = 42
np.random.seed(SEED)
# ---- CONFIGURATION ----
# Mode set to 'log_price' as requested
MODE = 'log_price'

def main():
    """
    Main function to train the price prediction model using LightGBM.
    1. Trains on a 90/10 split to find the best iteration and validate performance.
    2. Retrains on the full training data using the optimal parameters.
    3. Saves the final model to the 'models/LGBM' directory.
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Make sure to update this path to your actual data directory
    DATA_DIR = os.path.join(BASE_DIR, "dataset")
    MODELS_DIR = os.path.join(BASE_DIR, "models/LGBM")

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

    # Identify categorical feature columns for the booster
    cat_features = ["unit", "canon_unit", "brand", "category_bucket", "sub_category", "flavor_profile"]
    for col in cat_features:
        X_num_train[col] = X_num_train[col].astype('category')
        X_num_val[col] = X_num_val[col].astype('category')

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

    # Prepare categorical feature names for LightGBM
    cat_feature_names = [col for col in cat_features if col in X_train.columns]

    # Set up and train the LightGBM regressor for validation
    print("Training validation model on GPU...")
    
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_feature_names, free_raw_data=False)
    val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_feature_names, reference=train_data, free_raw_data=False)
    
    params = {
        'objective': 'mae',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 63,  # 2^6 - 1, roughly equivalent to depth=6
        'max_depth': 6,
        'min_child_samples': 20,
        'subsample': 0.8,
        'subsample_freq': 1,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,  # L1 regularization
        'reg_lambda': 1.0,  # L2 regularization
        'random_state': SEED,
        'device': 'cpu',  # instead of 'gpu'
        'num_threads': 24,
        'gpu_use_dp': True,  # Double precision for numerical stability
        'verbose': -1
    }
    
    callbacks = [
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100)
    ]
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=10000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=callbacks
    )

    # Predict on validation set and convert to total price scale
    print("Evaluating validation model...")
    raw_pred = model.predict(X_val, num_iteration=model.best_iteration)
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
    for col in cat_features:
        X_num_full[col] = X_num_full[col].astype('category')

    print("Generating text features for the full dataset...")
    X_text_full = tfidf.fit_transform(texts.tolist())
    X_svd_full = svd.fit_transform(X_text_full)
    X_svd_full_df = pd.DataFrame(X_svd_full, columns=svd_cols)
    X_full = pd.concat([X_num_full.reset_index(drop=True), X_svd_full_df.reset_index(drop=True)], axis=1)

    cat_feature_names_full = [col for col in cat_features if col in X_full.columns]
    y_full = y  # Already defined as np.log1p(total_price)

    # Train for the optimal number of iterations found in the validation step
    final_iterations = model.best_iteration
    print(f"Training final model on GPU for {final_iterations} iterations...")
    
    full_train_data = lgb.Dataset(X_full, label=y_full, categorical_feature=cat_feature_names_full, free_raw_data=False)
    
    final_model = lgb.train(
        params,
        full_train_data,
        num_boost_round=final_iterations,
        valid_sets=[full_train_data],
        valid_names=['train'],
        callbacks=[lgb.log_evaluation(period=100)]
    )

    # Save the final model
    model_path = os.path.join(MODELS_DIR, 'lgbm_log_price_model.txt')
    final_model.save_model(model_path)
    print(f"\nModel successfully trained and saved to: {model_path}")


if __name__ == "__main__":
    main()
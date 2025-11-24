# ensemble_blend.py
# Optimized model blending with validation and test prediction

import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.sparse import hstack
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Import your feature extraction
from price_features_transformer import build_numeric_table, clean_text, smape, extract_features_row

# Import transformer components
from transformers import AutoTokenizer, AutoModel

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ===================== Transformer Model Definition (from T1) =====================
class TransformerModel(nn.Module):
    def __init__(self, name, num_num, num_units, num_cats, num_flavs, hidden=768, drop=0.2):
        super().__init__()
        self.enc = AutoModel.from_pretrained(name, trust_remote_code=False)
        d = self.enc.config.hidden_size
        self.unit_emb = nn.Embedding(num_units, 16)
        self.cat_emb = nn.Embedding(num_cats, 16)
        self.flav_emb = nn.Embedding(num_flavs, 16)
        self.mlp = nn.Sequential(
            nn.Linear(d + num_num + 16 + 16 + 16, hidden),
            nn.LayerNorm(hidden), 
            nn.SiLU(), 
            nn.Dropout(drop),
            nn.Linear(hidden, hidden//2),
            nn.LayerNorm(hidden//2), 
            nn.SiLU(), 
            nn.Dropout(drop),
            nn.Linear(hidden//2, hidden//4),
            nn.LayerNorm(hidden//4),
            nn.SiLU(),
            nn.Dropout(drop/2),
            nn.Linear(hidden//4, 1)
        )

    def mean_pool(self, last_hidden, mask):
        mask = mask.unsqueeze(-1).float()
        return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)

    def forward(self, ids, mask, num, unit, cat, flav):
        out = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        txt = self.mean_pool(out, mask)
        u = self.unit_emb(unit)
        c = self.cat_emb(cat)
        f = self.flav_emb(flav)
        x = torch.cat([txt, num, u, c, f], 1)
        return self.mlp(x).squeeze(-1)

# ===================== Elite Transformer Model Definition (from T3) =====================
class ResidualFusion(nn.Module):
    """Residual fusion layer with gating"""
    def __init__(self, text_dim, tabular_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.tab_proj = nn.Linear(tabular_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text_emb, tab_emb):
        text_h = self.text_proj(text_emb)
        tab_h = self.tab_proj(tab_emb)
        
        # Gated fusion
        concat = torch.cat([text_h, tab_h], dim=-1)
        gate = torch.sigmoid(self.gate(concat))
        
        fused = gate * text_h + (1 - gate) * tab_h
        fused = self.norm(fused)
        fused = self.dropout(fused)
        
        return fused

class EliteModel(nn.Module):
    def __init__(self, model_name, num_num, num_brands, num_units, 
                 num_cats, num_flavs, hidden=512, dropout=0.15):
        super().__init__()
        
        # Text encoder (MPNet doesn't support gradient checkpointing)
        self.enc = AutoModel.from_pretrained(model_name, trust_remote_code=False)
        d = self.enc.config.hidden_size
        
        # Embeddings with optimal dimensions
        self.brand_emb = nn.Embedding(num_brands, 32)
        self.unit_emb = nn.Embedding(num_units, 16)
        self.cat_emb = nn.Embedding(num_cats, 24)
        self.flav_emb = nn.Embedding(num_flavs, 16)
        
        # Tabular feature processor
        tabular_input_dim = num_num + 32 + 16 + 24 + 16
        self.tab_processor = nn.Sequential(
            nn.Linear(tabular_input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        
        # Multi-scale text fusion (CLS + mean pool)
        self.text_fusion = nn.Linear(d * 2, d)
        
        # Residual fusion layer
        self.fusion = ResidualFusion(d, hidden, hidden, dropout)
        
        # Prediction head with residual connections
        self.head = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.SiLU(),
                nn.Dropout(dropout),
            ),
            nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.LayerNorm(hidden // 2),
                nn.SiLU(),
                nn.Dropout(dropout / 2),
            ),
            nn.Linear(hidden // 2, 1)
        ])
        
        self.res_proj1 = nn.Linear(hidden, hidden)
        self.res_proj2 = nn.Linear(hidden, hidden // 2)

    def mean_pool(self, last_hidden, mask):
        mask = mask.unsqueeze(-1).float()
        return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)

    def forward(self, input_ids, attention_mask, num, brand, unit, cat, flav):
        # Text encoding with multi-scale pooling
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = out.last_hidden_state[:, 0, :]  # CLS token
        mean_emb = self.mean_pool(out.last_hidden_state, attention_mask)  # Mean pool
        
        # Combine multi-scale text features
        text_emb = self.text_fusion(torch.cat([cls_emb, mean_emb], dim=-1))
        
        # Tabular features
        b_emb = self.brand_emb(brand)
        u_emb = self.unit_emb(unit)
        c_emb = self.cat_emb(cat)
        f_emb = self.flav_emb(flav)
        
        tab_features = torch.cat([num, b_emb, u_emb, c_emb, f_emb], dim=-1)
        tab_emb = self.tab_processor(tab_features)
        
        # Fusion
        fused = self.fusion(text_emb, tab_emb)
        
        # Prediction head with residual connections
        x = self.head[0](fused)
        x = x + self.res_proj1(fused)  # Residual
        
        x = self.head[1](x)
        x = x + self.res_proj2(fused)  # Skip connection
        
        x = self.head[2](x)
        
        return x.squeeze(-1)

# ===================== Model Loaders =====================
def load_catboost():
    """Load CatBoost model"""
    from catboost import CatBoostRegressor
    model_path = os.path.join(BASE_DIR, "models/catboost/catboost_log_price_model.cbm")
    if not os.path.exists(model_path):
        print(f"[Warning] CatBoost model not found: {model_path}")
        return None
    model = CatBoostRegressor()
    model.load_model(model_path)
    return model

def load_lgbm():
    """Load LightGBM model"""
    import lightgbm as lgb
    model_path = os.path.join(BASE_DIR, "models/LGBM/lgbm_log_price_model.txt")
    if not os.path.exists(model_path):
        print(f"[Warning] LGBM model not found: {model_path}")
        return None
    return lgb.Booster(model_file=model_path)

def load_xgboost():
    """Load XGBoost model"""
    import xgboost as xgb
    model_path = os.path.join(BASE_DIR, "models/XGBoost/xgboost_log_price_model.json")
    if not os.path.exists(model_path):
        print(f"[Warning] XGBoost model not found: {model_path}")
        return None
    model = xgb.Booster()
    model.load_model(model_path)
    return model


def load_ridge():
    """Load Ridge model - gracefully handle version mismatch"""
    model_dir = os.path.join(BASE_DIR, "models/ridge_fast")
    model_path = os.path.join(model_dir, "ridge.pkl")
    if not os.path.exists(model_path):
        print(f"[Warning] Ridge model not found: {model_path}")
        return None
    
    try:
        ridge_dict = {
            'model': joblib.load(model_path),
            'tfidf_word': joblib.load(os.path.join(model_dir, "tfidf_word.pkl")),
            'tfidf_char': joblib.load(os.path.join(model_dir, "tfidf_char.pkl")),
            'scaler': joblib.load(os.path.join(model_dir, "scaler.pkl"))
        }
        
        # Validate that components are properly fitted
        if (not hasattr(ridge_dict['tfidf_word'], 'idf_') or 
            ridge_dict['tfidf_word'].idf_ is None or 
            len(ridge_dict['tfidf_word'].idf_) == 0):
            print(f"[Warning] Ridge TF-IDF word vectorizer corrupted (sklearn version mismatch)")
            return None
            
        if (not hasattr(ridge_dict['tfidf_char'], 'idf_') or 
            ridge_dict['tfidf_char'].idf_ is None or 
            len(ridge_dict['tfidf_char'].idf_) == 0):
            print(f"[Warning] Ridge TF-IDF char vectorizer corrupted (sklearn version mismatch)")
            return None
        
        return ridge_dict
    
    except Exception as e:
        print(f"[Warning] Failed to load Ridge model: {repr(e)}")
        return None


def load_transformer(model_path="models/T1/best_model.pt"):
    """Load Transformer model"""
    full_model_path = os.path.join(BASE_DIR, model_path)
    if not os.path.exists(full_model_path):
        print(f"[Warning] Transformer model not found: {full_model_path}")
        return None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(full_model_path, map_location=device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint['tokenizer'])
    
    # Reconstruct label encoders
    unit_le = LabelEncoder()
    unit_le.classes_ = np.array(checkpoint['unit_classes'])
    
    cat_le = LabelEncoder()
    cat_le.classes_ = np.array(checkpoint['cat_classes'])
    
    flav_le = LabelEncoder()
    flav_le.classes_ = np.array(checkpoint['flav_classes'])
    
    # Reconstruct scaler
    scaler = StandardScaler()
    scaler.mean_ = np.array(checkpoint['scaler_mean'])
    scaler.scale_ = np.array(checkpoint['scaler_scale'])
    
    # Build model
    model = TransformerModel(
        checkpoint['tokenizer'],
        num_num=len(checkpoint['numeric_cols']),
        num_units=len(unit_le.classes_),
        num_cats=len(cat_le.classes_),
        num_flavs=len(flav_le.classes_),
        hidden=checkpoint.get('hidden_dim', 768)
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    
    return {
        'model': model,
        'tokenizer': tokenizer,
        'unit_le': unit_le,
        'cat_le': cat_le,
        'flav_le': flav_le,
        'scaler': scaler,
        'numeric_cols': checkpoint['numeric_cols'],
        'max_length': checkpoint.get('max_length', 384),
        'device': device
    }

def load_elite_transformer():
    """Load Elite Transformer model from T3"""
    full_model_path = os.path.join(BASE_DIR, "models/artifacts_elite/best_model.pt")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(full_model_path, map_location=device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint['config']['model_name'], trust_remote_code=False)
    
    # Reconstruct label encoders
    brand_le = LabelEncoder()
    brand_le.classes_ = np.array(checkpoint['brand_classes'])
    
    unit_le = LabelEncoder()
    unit_le.classes_ = np.array(checkpoint['unit_classes'])
    
    cat_le = LabelEncoder()
    cat_le.classes_ = np.array(checkpoint['cat_classes'])
    
    flav_le = LabelEncoder()
    flav_le.classes_ = np.array(checkpoint['flav_classes'])
    
    # Reconstruct scaler
    scaler = StandardScaler()
    scaler.mean_ = np.array(checkpoint['scaler_mean'])
    scaler.scale_ = np.array(checkpoint['scaler_scale'])
    
    # Build model
    model = EliteModel(
        checkpoint['config']['model_name'],
        num_num=len(checkpoint['numeric_cols']),
        num_brands=len(brand_le.classes_),
        num_units=len(unit_le.classes_),
        num_cats=len(cat_le.classes_),
        num_flavs=len(flav_le.classes_),
        hidden=checkpoint['config']['hidden'],
        dropout=checkpoint['config']['dropout']
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    
    return {
        'model': model,
        'tokenizer': tokenizer,
        'brand_le': brand_le,
        'unit_le': unit_le,
        'cat_le': cat_le,
        'flav_le': flav_le,
        'scaler': scaler,
        'numeric_cols': checkpoint['numeric_cols'],
        'max_length': checkpoint['config']['max_length'],
        'device': device
    }

# ===================== Prediction Functions =====================
def predict_catboost(model, df, tfidf, svd):
    """Generate CatBoost predictions"""
    texts = df["catalog_content"].fillna("").map(clean_text)
    numeric_features = build_numeric_table(df["catalog_content"].fillna(""))
    
    cat_features = ["unit", "canon_unit", "brand", "category_bucket", "sub_category", "flavor_profile"]
    for col in cat_features:
        numeric_features[col] = numeric_features[col].astype(str).fillna('missing')
    
    X_text = tfidf.transform(texts.tolist())
    X_svd = svd.transform(X_text)
    svd_cols = [f"text_comp{i}" for i in range(X_svd.shape[1])]
    X_svd_df = pd.DataFrame(X_svd, columns=svd_cols)
    
    X = pd.concat([numeric_features.reset_index(drop=True), X_svd_df.reset_index(drop=True)], axis=1)
    
    cat_indices = [X.columns.get_loc(col) for col in cat_features if col in X.columns]
    
    raw_pred = model.predict(X)
    return np.expm1(raw_pred).clip(0.99, None)

def predict_lgbm(model, df, tfidf, svd):
    """Generate LGBM predictions"""
    texts = df["catalog_content"].fillna("").map(clean_text)
    numeric_features = build_numeric_table(df["catalog_content"].fillna(""))
    
    cat_features = ["unit", "canon_unit", "brand", "category_bucket", "sub_category", "flavor_profile"]
    for col in cat_features:
        numeric_features[col] = numeric_features[col].astype('category')
    
    X_text = tfidf.transform(texts.tolist())
    X_svd = svd.transform(X_text)
    svd_cols = [f"text_comp{i}" for i in range(X_svd.shape[1])]
    X_svd_df = pd.DataFrame(X_svd, columns=svd_cols)
    
    X = pd.concat([numeric_features.reset_index(drop=True), X_svd_df.reset_index(drop=True)], axis=1)
    
    raw_pred = model.predict(X, num_iteration=model.best_iteration)
    return np.expm1(raw_pred).clip(0.99, None)

def predict_xgboost(model, df, tfidf, svd, cat_mappings):
    """Generate XGBoost predictions"""
    import xgboost as xgb
    
    texts = df["catalog_content"].fillna("").map(clean_text)
    numeric_features = build_numeric_table(df["catalog_content"].fillna(""))
    
    cat_features = ["unit", "canon_unit", "brand", "category_bucket", "sub_category", "flavor_profile"]
    for col in cat_features:
        numeric_features[col] = numeric_features[col].astype(str).fillna('missing')
        numeric_features[col] = numeric_features[col].map(cat_mappings[col]).fillna(-1).astype(int)
    
    X_text = tfidf.transform(texts.tolist())
    X_svd = svd.transform(X_text)
    svd_cols = [f"text_comp{i}" for i in range(X_svd.shape[1])]
    X_svd_df = pd.DataFrame(X_svd, columns=svd_cols)
    
    X = pd.concat([numeric_features.reset_index(drop=True), X_svd_df.reset_index(drop=True)], axis=1)
    
    dmatrix = xgb.DMatrix(X)
    raw_pred = model.predict(dmatrix)
    return np.expm1(raw_pred).clip(0.99, None)

def predict_ridge(ridge_dict, df):
    """Generate Ridge predictions"""
    if ridge_dict is None:
        return None
    
    texts = df["catalog_content"].fillna("").map(clean_text).tolist()
    
    try:
        # Text features
        Xw = ridge_dict['tfidf_word'].transform(texts)
        Xc = ridge_dict['tfidf_char'].transform(texts)
        
        # Numeric features
        num = build_numeric_table(pd.Series(texts))
        num_cols = [c for c in num.columns if pd.api.types.is_numeric_dtype(num[c])]
        
        if len(num_cols) == 0:
            num = pd.DataFrame({
                "len_chars": pd.Series(texts).str.len().astype(float),
                "len_words": pd.Series(texts).str.split().apply(len).astype(float)
            })
        else:
            num = num[num_cols].fillna(0)
        
        Xn = ridge_dict['scaler'].transform(num)
        X = hstack([Xw, Xc, Xn]).tocsr()
        
        raw_pred = ridge_dict['model'].predict(X)
        return np.expm1(raw_pred).clip(0.99, None)
    
    except Exception as e:
        print(f"[Error] Ridge prediction failed: {repr(e)}")
        return None
def predict_transformer(trans_dict, df, batch_size=32):
    """Generate Transformer predictions"""
    device = trans_dict['device']
    model = trans_dict['model']
    tokenizer = trans_dict['tokenizer']
    
    texts = df["catalog_content"].fillna("").map(clean_text).tolist()
    
    # Extract features
    feat_list = [extract_features_row(t) for t in df['catalog_content'].fillna("")]
    feat_df = pd.DataFrame(feat_list)
    
    # Numeric features
    X_num = feat_df[trans_dict['numeric_cols']].fillna(0).values.astype(np.float32)
    X_num = trans_dict['scaler'].transform(X_num)
    
    # Categorical features
    units = feat_df['canon_unit'].fillna('<unk>').tolist()
    cats = feat_df['category_bucket'].fillna('other').tolist()
    flavs = feat_df['flavor_profile'].fillna('Other').tolist()
    
    unit_ids = []
    for u in units:
        try:
            unit_ids.append(trans_dict['unit_le'].transform([u])[0])
        except:
            unit_ids.append(trans_dict['unit_le'].transform(['<unk>'])[0])
    
    cat_ids = trans_dict['cat_le'].transform(cats)
    flav_ids = trans_dict['flav_le'].transform(flavs)
    
    # Batch prediction
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_num = X_num[i:i+batch_size]
            batch_unit = unit_ids[i:i+batch_size]
            batch_cat = cat_ids[i:i+batch_size]
            batch_flav = flav_ids[i:i+batch_size]
            
            # Tokenize
            enc = tokenizer(batch_texts, max_length=trans_dict['max_length'],
                          padding='max_length', truncation=True, return_tensors='pt')
            
            ids = enc['input_ids'].to(device)
            mask = enc['attention_mask'].to(device)
            num = torch.tensor(batch_num, dtype=torch.float32).to(device)
            unit = torch.tensor(batch_unit, dtype=torch.long).to(device)
            cat = torch.tensor(batch_cat, dtype=torch.long).to(device)
            flav = torch.tensor(batch_flav, dtype=torch.long).to(device)
            
            pred_log = model(ids, mask, num, unit, cat, flav)
            pred_raw = torch.expm1(pred_log).clamp(min=0.99)
            
            all_preds.append(pred_raw.cpu().numpy())
    
    return np.concatenate(all_preds)

def predict_elite_transformer(elite_dict, df, batch_size=32):
    """Generate Elite Transformer predictions"""
    device = elite_dict['device']
    model = elite_dict['model']
    tokenizer = elite_dict['tokenizer']
    
    texts = df["catalog_content"].fillna("").map(clean_text).tolist()
    
    # Extract features
    feat_list = [extract_features_row(t) for t in df['catalog_content'].fillna("")]
    feat_df = pd.DataFrame(feat_list)
    
    # Numeric features
    X_num = feat_df[elite_dict['numeric_cols']].fillna(0).values.astype(np.float32)
    X_num = elite_dict['scaler'].transform(X_num)
    
    # Categorical features
    brands = feat_df['brand'].fillna('<unk>').tolist()
    units = feat_df['canon_unit'].fillna('<unk>').tolist()
    cats = feat_df['category_bucket'].fillna('other').tolist()
    flavs = feat_df['flavor_profile'].fillna('Other').tolist()
    
    brand_ids = []
    for b in brands:
        try:
            brand_ids.append(elite_dict['brand_le'].transform([b])[0])
        except:
            brand_ids.append(elite_dict['brand_le'].transform(['<unk>'])[0])
    
    unit_ids = []
    for u in units:
        try:
            unit_ids.append(elite_dict['unit_le'].transform([u])[0])
        except:
            unit_ids.append(elite_dict['unit_le'].transform(['<unk>'])[0])
    
    cat_ids = []
    for c in cats:
        try:
            cat_ids.append(elite_dict['cat_le'].transform([c])[0])
        except:
            cat_ids.append(elite_dict['cat_le'].transform(['other'])[0])
    
    flav_ids = []
    for f in flavs:
        try:
            flav_ids.append(elite_dict['flav_le'].transform([f])[0])
        except:
            flav_ids.append(elite_dict['flav_le'].transform(['Other'])[0])
    
    # Batch prediction
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_num = X_num[i:i+batch_size]
            batch_brand = brand_ids[i:i+batch_size]
            batch_unit = unit_ids[i:i+batch_size]
            batch_cat = cat_ids[i:i+batch_size]
            batch_flav = flav_ids[i:i+batch_size]
            
            # Tokenize
            enc = tokenizer(batch_texts, max_length=elite_dict['max_length'],
                          padding='max_length', truncation=True, return_tensors='pt')
            
            ids = enc['input_ids'].to(device)
            mask = enc['attention_mask'].to(device)
            num = torch.tensor(batch_num, dtype=torch.float32).to(device)
            brand = torch.tensor(batch_brand, dtype=torch.long).to(device)
            unit = torch.tensor(batch_unit, dtype=torch.long).to(device)
            cat = torch.tensor(batch_cat, dtype=torch.long).to(device)
            flav = torch.tensor(batch_flav, dtype=torch.long).to(device)
            
            pred_log = model(ids, mask, num, brand, unit, cat, flav)
            pred_raw = torch.expm1(pred_log).clamp(min=0.99)
            
            all_preds.append(pred_raw.cpu().numpy())
    
    return np.concatenate(all_preds)

# ===================== Ensemble Optimization =====================
def optimize_weights(predictions_dict, y_true):
    """Find optimal weights using Nelder-Mead to minimize SMAPE"""
    model_names = list(predictions_dict.keys())
    n_models = len(model_names)
    
    def objective(weights):
        weights = np.array(weights) / weights.sum()  # Normalize
        ensemble = sum(w * predictions_dict[name] for w, name in zip(weights, model_names))
        return smape(y_true, ensemble)
    
    # Initial guess: equal weights
    x0 = np.ones(n_models) / n_models
    
    # Optimize
    result = minimize(objective, x0, method='Nelder-Mead',
                     options={'maxiter': 1000, 'xatol': 1e-6})
    
    optimal_weights = result.x / result.x.sum()
    best_smape = result.fun
    
    return dict(zip(model_names, optimal_weights)), best_smape


# ===================== Main Ensemble Pipeline =====================
def main():
    global BASE_DIR
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join("dataset")  # Changed to relative path for consistency
    
    print("="*60)
    print("ENSEMBLE BLENDING PIPELINE")
    print("="*60)
    
    # -------- Load all models --------
    print("\n[1/5] Loading models...")
    models = {}
    
    catboost_model = load_catboost()
    if catboost_model: models['catboost'] = catboost_model
    
    lgbm_model = load_lgbm()
    if lgbm_model: models['lgbm'] = lgbm_model
    
    xgb_model = load_xgboost()
    if xgb_model: models['xgboost'] = xgb_model
    
    ridge_model = load_ridge()
    if ridge_model: models['ridge'] = ridge_model
    
    trans_model = load_transformer()
    if trans_model: models['transformer'] = trans_model
    
    elite_model = load_elite_transformer()
    if elite_model: models['elite'] = elite_model
    
    print(f"Loaded {len(models)} models: {list(models.keys())}")
    
    if len(models) < 2:
        print("[Error] Need at least 2 models for ensembling!")
        return
    
    # -------- Load data --------
    print("\n[2/5] Loading train data...")
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    
    # Use same 90/10 split for consistency
    train_idx = int(0.9 * len(df))
    df_val = df.iloc[train_idx:].reset_index(drop=True)
    y_val = df_val["price"].astype(float).values
    
    print(f"Validation set: {len(df_val)} samples")
    
    # -------- Prepare shared components --------
    print("\n[3/5] Preparing text features...")
    texts = df["catalog_content"].fillna("").map(clean_text)
    
    # TF-IDF + SVD (for tree models)
    tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=2, max_features=200_000)
    X_text = tfidf.fit_transform(texts.tolist())
    svd = TruncatedSVD(n_components=128, random_state=SEED)
    svd.fit(X_text)
    
    # XGBoost categorical mappings
    xgb_cat_mappings = {}
    if 'xgboost' in models:
        cat_features = ["unit", "canon_unit", "brand", "category_bucket", "sub_category", "flavor_profile"]
        numeric_features = build_numeric_table(df["catalog_content"].fillna(""))
        for col in cat_features:
            unique_vals = numeric_features[col].astype(str).fillna('missing').unique()
            xgb_cat_mappings[col] = {val: idx for idx, val in enumerate(unique_vals)}
    
# -------- Generate validation predictions --------
    print("\n[4/5] Generating validation predictions...")
    val_preds = {}
    
    if 'catboost' in models:
        print("  - CatBoost...")
        val_preds['catboost'] = predict_catboost(models['catboost'], df_val, tfidf, svd)
    
    if 'lgbm' in models:
        print("  - LightGBM...")
        val_preds['lgbm'] = predict_lgbm(models['lgbm'], df_val, tfidf, svd)
    
    if 'xgboost' in models:
        print("  - XGBoost...")
        val_preds['xgboost'] = predict_xgboost(models['xgboost'], df_val, tfidf, svd, xgb_cat_mappings)
    
    if 'ridge' in models:
        print("  - Ridge...")
        ridge_pred = predict_ridge(models['ridge'], df_val)
        if ridge_pred is not None:
            val_preds['ridge'] = ridge_pred
        else:
            print("    [Skipped] Ridge model unavailable, removing from ensemble")
            del models['ridge']
    
    if 'transformer' in models:
        print("  - Transformer...")
        val_preds['transformer'] = predict_transformer(models['transformer'], df_val)
    
    if 'elite' in models:
        print("  - Elite Transformer...")
        val_preds['elite'] = predict_elite_transformer(models['elite'], df_val)
    
    # -------- Evaluate individual models --------
    print("\n" + "="*60)
    print("VALIDATION SCORES (Individual Models)")
    print("="*60)
    for name, preds in val_preds.items():
        score = smape(y_val, preds)
        print(f"{name:15s}: {score:6.2f}% SMAPE")
    
    # -------- Optimize ensemble weights --------
    print("\n[5/5] Optimizing ensemble weights...")
    optimal_weights, best_smape = optimize_weights(val_preds, y_val)
    
    print("\n" + "="*60)
    print("OPTIMAL ENSEMBLE WEIGHTS")
    print("="*60)
    for name, weight in sorted(optimal_weights.items(), key=lambda x: -x[1]):
        print(f"{name:15s}: {weight:6.4f} ({weight*100:5.1f}%)")
    print(f"\nEnsemble SMAPE: {best_smape:.2f}%")
    
    # -------- Generate test predictions --------
    test_path = os.path.join(DATA_DIR, "test.csv")
    if not os.path.exists(test_path):
        print(f"\n[Warning] Test file not found: {test_path}")
        print("Skipping test predictions.")
        return
    
    print("\n" + "="*60)
    print("GENERATING TEST PREDICTIONS")
    print("="*60)
    df_test = pd.read_csv(test_path)
    print(f"Test samples: {len(df_test)}")
    
    test_preds = {}
    
    if 'catboost' in models:
        print("  - CatBoost...")
        test_preds['catboost'] = predict_catboost(models['catboost'], df_test, tfidf, svd)
    
    if 'lgbm' in models:
        print("  - LightGBM...")
        test_preds['lgbm'] = predict_lgbm(models['lgbm'], df_test, tfidf, svd)
    
    if 'xgboost' in models:
        print("  - XGBoost...")
        test_preds['xgboost'] = predict_xgboost(models['xgboost'], df_test, tfidf, svd, xgb_cat_mappings)
    
    if 'ridge' in models:
        print("  - Ridge...")
        ridge_pred = predict_ridge(models['ridge'], df_test)
        if ridge_pred is not None:
            test_preds['ridge'] = ridge_pred
    
    if 'transformer' in models:
        print("  - Transformer...")
        test_preds['transformer'] = predict_transformer(models['transformer'], df_test)
    
    if 'elite' in models:
        print("  - Elite Transformer...")
        test_preds['elite'] = predict_elite_transformer(models['elite'], df_test)
    
    # Blend test predictions
    ensemble_test = sum(optimal_weights[name] * test_preds[name] for name in test_preds.keys())
    
    # Save submission
    submission = pd.DataFrame({
        'sample_id': df_test['sample_id'],
        'price': ensemble_test
    })
    
    output_path = os.path.join(BASE_DIR, "submission_ensemble.csv")
    submission.to_csv(output_path, index=False)
    
    print(f"\n[Done] Submission saved to: {output_path}")
    print(f"Price range: [{submission['price'].min():.2f}, {submission['price'].max():.2f}]")
    print(f"Mean price: {submission['price'].mean():.2f}")

if __name__ == "__main__":
    main()
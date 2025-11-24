# price_features_v2.py - Fixed version without data leakage
import re
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

# --------------------- Units & canonicalization ---------------------
UNIT_SCALES: Dict[str, Tuple[str, float]] = {
    # mass units -> grams
    'lb': ('g', 453.59237), 'lbs': ('g', 453.59237),
    'pound': ('g', 453.59237), 'pounds': ('g', 453.59237),
    'kg': ('g', 1000.0), 'kilogram': ('g', 1000.0), 'kilograms': ('g', 1000.0),
    'g': ('g', 1.0), 'gram': ('g', 1.0), 'grams': ('g', 1.0),
    'mg': ('g', 1e-3), 'milligram': ('g', 1e-3), 'milligrams': ('g', 1e-3),
    'oz': ('g', 28.349523125), 'ounce': ('g', 28.349523125), 'ounces': ('g', 28.349523125),

    # volume units -> milliliters
    'l': ('ml', 1000.0), 'liter': ('ml', 1000.0), 'litre': ('ml', 1000.0),
    'liters': ('ml', 1000.0), 'litres': ('ml', 1000.0),
    'ml': ('ml', 1.0), 'milliliter': ('ml', 1.0), 'milliliters': ('ml', 1.0),
    'millilitre': ('ml', 1.0), 'millilitres': ('ml', 1.0),
    'fl oz': ('ml', 29.5735295625), 'floz': ('ml', 29.5735295625),

    # length units
    'in': ('cm', 2.54), 'inch': ('cm', 2.54), 'inches': ('cm', 2.54),
    'ft': ('cm', 30.48), 'foot': ('cm', 30.48), 'feet': ('cm', 30.48),
    'cm': ('cm', 1.0), 'mm': ('cm', 0.1),

    # count units
    'count': ('count', 1.0), 'ct': ('count', 1.0),
    'pcs': ('count', 1.0), 'piece': ('count', 1.0), 'pieces': ('count', 1.0),
}

def _norm_unit(u: str) -> str:
    u = u.lower().strip()
    u = re.sub(r'[.\s]+', ' ', u)
    u = u.replace('fluid ounce', 'fl oz').replace('fluid ounces', 'fl oz')
    u = u.replace('fl. oz', 'fl oz').replace('fl-oz', 'fl oz')
    u = u.replace('ounces', 'oz').replace('ounce', 'oz')
    u = u.replace('ct.', 'ct')
    return u

def canonicalize_unit(raw_unit: str) -> Tuple[str, float]:
    if not isinstance(raw_unit, str) or not raw_unit.strip():
        return '<unk>', 1.0
    u = _norm_unit(raw_unit)
    if u in UNIT_SCALES:
        return UNIT_SCALES[u]
    u2 = u.replace(' ', '')
    if u2 in UNIT_SCALES:
        return UNIT_SCALES[u2]
    if u.endswith('s') and u[:-1] in UNIT_SCALES:
        return UNIT_SCALES[u[:-1]]
    return '<unk>', 1.0

# --------------------- Patterns ---------------------
VALUE_RE = re.compile(r'Value:\s*([+-]?\d+(?:\.\d+)?)', re.IGNORECASE)
UNIT_RE  = re.compile(r'Unit:\s*([A-Za-z.\-\s]+)', re.IGNORECASE)

PACK_PATTERNS = [
    r'[Pp]ack\s*of\s*(\d+)',
    r'(\d+)\s*[Pp]ack\b',
    r'(\d+)\s*-\s*[Cc]ount',
    r'(\d+)\s*(?:ct|count)\b',
    r'[x×]\s*(\d+)\b',
    r'\b(\d+)\s*[x×]\s*\d+\s*(?:oz|fl\s*oz|g|ml)\b',
    r'\(.*?[Pp]ack\s*of\s*(\d+).*?\)',
    r'\bcase\s*of\s*(\d+)\b',
]
PACK_RE = re.compile('|'.join(PACK_PATTERNS))

PREMIUM_KWS = ['organic', 'premium', 'gourmet', 'artisan', 'natural', 'handcrafted', 'imported', 'luxury']
BULK_KWS    = ['pack', 'case', 'bulk', 'bundle', 'family size', 'wholesale']

CAT_RULES = [
    ('soup',     r'\bsoup\b|ramen|broth'),
    ('sauce',    r'\bsauce\b|ketchup|mustard|mayo|dressing|marinara|salsa'),
    ('cookies',  r'\bcookies?\b|biscuit'),
    ('candy',    r'candy|chocolate|gummy|toffee|mint'),
    ('snack',    r'\bchips?\b|pretzel|popcorn|cracker|snack'),
    ('spice',    r'\bspice\b|seasoning|masala|herb|salt|pepper'),
    ('beverage', r'coffee|tea|soda|drink|beverage|juice'),
    ('grains',   r'rice|pasta|noodle|flour|oats|cereal'),
    ('oil',      r'\boil\b|olive oil|canola|sunflower'),
    ('gift',     r'gift|basket|hamper'),
    ('dairy',    r'cheese|milk|butter|yogurt'),
]

PROTEIN_RE = re.compile(r'(\d+(?:\.\d+)?)\s*(?:g|grams?)\s*(?:of\s*)?protein', re.IGNORECASE)
FIBER_RE = re.compile(r'(\d+(?:\.\d+)?)\s*(?:g|grams?)\s*(?:of\s*)?fiber', re.IGNORECASE)
CALORIES_RE = re.compile(r'(\d+(?:\.\d+)?)\s*calories?', re.IGNORECASE)
SUGAR_RE = re.compile(r'(\d+(?:\.\d+)?)\s*(?:g|grams?)\s*(?:of\s*)?sugar', re.IGNORECASE)
ITEM_NAME_RE = re.compile(r'Item Name:\s*(.+?)(?=\n|Bullet Point|Product Description|$)', re.IGNORECASE | re.DOTALL)

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r'Value:.*?Unit:.*?(?:\n|$)', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_value_unit(text: str) -> Tuple[float, str]:
    if not isinstance(text, str):
        return 0.0, 'Unknown'
    vm = VALUE_RE.search(text)
    um = UNIT_RE.search(text)
    value = float(vm.group(1)) if vm else 0.0
    unit  = um.group(1).strip() if um else 'Unknown'
    return value, unit

def detect_pack_size(text: str) -> int:
    if not isinstance(text, str):
        return 1
    m = PACK_RE.search(text)
    if not m:
        return 1
    for g in m.groups():
        if g is not None:
            try:
                return max(1, int(g))
            except:
                continue
    return 1

def extract_brand(text: str) -> str:
    if not isinstance(text, str):
        return "<unk>"
    m = re.search(r'Item Name:\s*(.+)', text, flags=re.IGNORECASE)
    s = m.group(1) if m else text
    s = s.split('|')[0].split(' - ')[0].split(',')[0].strip()
    s = re.sub(r'\s+', ' ', s)
    if len(s) == 0:
        return "<unk>"
    first_tok = s.split(' ')[0]
    if len(first_tok) <= 2:
        return s[:40].strip() or "<unk>"
    return first_tok[:40]

def category_bucket(text: str) -> str:
    low = text.lower() if isinstance(text, str) else ""
    for cat, pattern in CAT_RULES:
        if re.search(pattern, low):
            return cat
    return 'other'

def extract_item_name(text: str) -> str:
    if not isinstance(text, str):
        return ""
    m = ITEM_NAME_RE.search(text)
    return m.group(1).strip() if m else ""

def extract_sub_category(item_name: str) -> str:
    if not item_name:
        return "Other"
    return item_name[:50] or "Other"

def extract_flavor_profile(item_name: str, text: str) -> str:
    flavors = re.search(r'(mild|original|creamy|blue cheese|sherry|basil|key lime|black raspberry|strawberry banana|cookie dough|other flavors)', text.lower())
    return flavors.group(1) if flavors else "Other"

def extract_features_row(text: str) -> Dict[str, Any]:
    """
    Extract features WITHOUT any price information (no data leakage).
    All features here are derived purely from the catalog content.
    """
    if not isinstance(text, str):
        text = ""
    
    value_raw, unit_raw = extract_value_unit(text)
    canon_unit, scale = canonicalize_unit(unit_raw)
    value_canon = value_raw * scale

    pack_size = detect_pack_size(text)
    value_per_pack = value_canon / max(pack_size, 1)
    low = text.lower()

    total_ml = value_canon if canon_unit == 'ml' else 0.0
    total_g  = value_canon if canon_unit == 'g'  else 0.0
    is_bulk = 1.0 if (pack_size >= 6 or total_ml >= 1500 or total_g >= 1000) else 0.0

    item_name = extract_item_name(text)
    sub_category = extract_sub_category(item_name)
    flavor_profile = extract_flavor_profile(item_name, text)

    protein_m = PROTEIN_RE.search(text)
    fiber_m = FIBER_RE.search(text)
    calories_m = CALORIES_RE.search(text)
    sugar_m = SUGAR_RE.search(text)
    protein_grams = float(protein_m.group(1)) if protein_m else 0.0
    fiber_grams = float(fiber_m.group(1)) if fiber_m else 0.0
    calories_per_serving = float(calories_m.group(1)) if calories_m else 0.0
    sugar_grams = float(sugar_m.group(1)) if sugar_m else 0.0

    is_high_protein = 1.0 if 'high protein' in low or protein_grams > 10 else 0.0
    is_low_calorie = 1.0 if 'low calorie' in low or (calories_per_serving > 0 and calories_per_serving < 50) else 0.0

    # === FIXED: Valid quantity-based features (no price information) ===
    # Log-scaled features
    log_value_canon = np.log1p(value_canon)
    log_pack_size = np.log1p(pack_size)
    log_value_per_pack = np.log1p(value_per_pack)
    log_total_g = np.log1p(total_g)
    log_total_ml = np.log1p(total_ml)
    
    # Ratio features
    value_to_pack_ratio = value_canon / max(pack_size, 1)
    
    # Unit type indicators
    is_weight_unit = 1.0 if canon_unit == 'g' else 0.0
    is_volume_unit = 1.0 if canon_unit == 'ml' else 0.0
    is_count_unit = 1.0 if canon_unit == 'count' else 0.0
    
    # Inverse features (for modeling price per unit without using price)
    inv_total_g = min(1.0 / max(total_g, 1e-3), 1e4) if total_g > 0 else 0.0
    inv_total_ml = min(1.0 / max(total_ml, 1e-3), 1e4) if total_ml > 0 else 0.0
    inv_value_canon = 1.0 / value_canon if value_canon > 0 else 0.0
    inv_pack_size = 1.0 / pack_size

    return {
        'value': float(value_raw),
        'unit': unit_raw,
        'value_canon': float(value_canon),
        'canon_unit': canon_unit,
        'pack_size': float(pack_size),
        'value_per_pack': float(value_per_pack),
        'total_ml': float(total_ml),
        'total_g': float(total_g),
        
        # Phase 1: Valid quantity features (NO PRICE LEAKAGE)
        'log_value_canon': float(log_value_canon),
        'log_pack_size': float(log_pack_size),
        'log_value_per_pack': float(log_value_per_pack),
        'log_total_g': float(log_total_g),
        'log_total_ml': float(log_total_ml),
        'value_to_pack_ratio': float(value_to_pack_ratio),
        'is_weight_unit': float(is_weight_unit),
        'is_volume_unit': float(is_volume_unit),
        'is_count_unit': float(is_count_unit),
        'inv_total_g': float(inv_total_g),
        'inv_total_ml': float(inv_total_ml),
        'inv_value_canon': float(inv_value_canon),
        'inv_pack_size': float(inv_pack_size),
        
        'text_length': len(text),
        'word_count': len(text.split()),
        'bullet_count': float(text.count('Bullet Point')),
        'premium_count': float(sum(kw in low for kw in PREMIUM_KWS)),
        'bulk_count': float(sum(kw in low for kw in BULK_KWS)),
        'has_number': float(1 if re.search(r'\d', text) else 0),
        'is_bulk': is_bulk,
        'brand': extract_brand(text),
        'category_bucket': category_bucket(text),
        'sub_category': sub_category,
        'flavor_profile': flavor_profile,
        
        # Quality flags
        'is_organic': 1.0 if 'organic' in low else 0.0,
        'is_gluten_free': 1.0 if 'gluten-free' in low or 'gluten free' in low else 0.0,
        'is_vegan': 1.0 if 'vegan' in low or 'plant-based' in low else 0.0,
        'is_keto_friendly': 1.0 if 'keto' in low or 'low carb' in low else 0.0,
        'is_non_gmo': 1.0 if 'non-gmo' in low or 'gmo free' in low else 0.0,
        'is_kosher': 1.0 if 'kosher' in low else 0.0,
        'is_dairy_free': 1.0 if 'dairy free' in low else 0.0,
        'is_nut_free': 1.0 if 'nut-free' in low else 0.0,
        'is_soy_free': 1.0 if 'soy free' in low else 0.0,
        'is_sugar_free': 1.0 if 'sugar free' in low or 'no sugar added' in low else 0.0,
        'is_low_fat': 1.0 if 'low fat' in low or '0 grams trans fat' in low else 0.0,
        'is_low_sodium': 1.0 if 'low sodium' in low or 'no salt added' in low else 0.0,
        'is_all_natural': 1.0 if 'all natural' in low or 'natural' in low else 0.0,
        'is_fair_trade': 1.0 if 'fair trade' in low else 0.0,
        'is_usda_certified': 1.0 if 'usda organic' in low else 0.0,
        
        # Nutritional features
        'protein_grams': protein_grams,
        'fiber_grams': fiber_grams,
        'calories_per_serving': calories_per_serving,
        'sugar_grams': sugar_grams,
        'has_vitamins': 1.0 if 'vitamin' in low or 'vitamins' in low else 0.0,
        'has_minerals': 1.0 if any(k in low for k in ['potassium', 'iron', 'magnesium', 'zinc']) else 0.0,
        'has_antioxidants': 1.0 if 'antioxidants' in low else 0.0,
        'is_high_protein': is_high_protein,
        'is_low_calorie': is_low_calorie,
        
        # Preparation features
        'is_ready_to_eat': 1.0 if 'ready to eat' in low else 0.0,
        'is_easy_to_prepare': 1.0 if 'easy to prepare' in low or 'instant' in low else 0.0,
        'is_versatile': 1.0 if 'versatile' in low or 'multiple uses' in low else 0.0,
        'is_snack': 1.0 if 'snack' in low or 'on-the-go' in low else 0.0,
        'is_beverage': 1.0 if any(k in low for k in ['drink', 'beverage', 'tea', 'coffee', 'juice']) else 0.0,
        'is_baking_ingredient': 1.0 if any(k in low for k in ['baking', 'flour', 'powder']) else 0.0,
        'has_no_preservatives': 1.0 if 'no preservatives' in low else 0.0,
        'is_shelf_stable': 1.0 if any(k in low for k in ['shelf stable', 'canned']) else 0.0,
        
        # Metadata
        'has_ingredients_list': 1.0 if 'ingredients:' in low else 0.0,
        'has_product_description': 1.0 if 'product description:' in low else 0.0,
    }

def build_numeric_table(catalog_series: pd.Series) -> pd.DataFrame:
    feats = [extract_features_row(t) for t in catalog_series]
    return pd.DataFrame(feats)

def smape(y_true, y_pred) -> float:
    """Symmetric Mean Absolute Percentage Error"""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    y_true_arr = np.abs(y_true_arr)
    y_pred_arr = np.abs(y_pred_arr)
    denom = (y_true_arr + y_pred_arr) / 2.0
    return float(np.mean(np.abs(y_pred_arr - y_true_arr) / (denom + 1e-8)) * 100.0)
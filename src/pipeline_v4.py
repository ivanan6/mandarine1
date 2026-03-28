"""
Insurance Pricing Pipeline v4 — Fast & Optimized
=================================================
- 3 folds, lr 0.05, 3000 rounds (~40 min)
- Out-of-fold target encoding (no leakage)
- Feature selection (drop weak features)
- Per-insurer tuned
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from pathlib import Path
import warnings
import time
import gc
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SUBMISSIONS = ROOT / "submissions"
SUBMISSIONS.mkdir(exist_ok=True)

INSURERS = list("ABCDEFGHIJK")
PRICE_COLS = [f"Insurer_{x}_price" for x in INSURERS]
DEDUCTIBLE_COLS = [f"Insurer_{x}_deductible" for x in INSURERS]

# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
print("Loading data...")
train_raw = pd.read_parquet(DATA / "block1_train.parquet", dtype_backend='numpy_nullable')
test2_raw = pd.read_parquet(DATA / "block2_test.parquet", dtype_backend='numpy_nullable')
test3_raw = pd.read_parquet(DATA / "block3_test (1).parquet", dtype_backend='numpy_nullable')
print(f"Train: {train_raw.shape}, Test2: {test2_raw.shape}, Test3: {test3_raw.shape}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

# Columns to drop early (>60% missing or useless)
DROP_EARLY = [
    'vehicle_number_plate', 'usage', 'vehicle_ownership_duration',
    'second_driver_birthdate', 'second_driver_claim_free_years',
    'postal_code_houses_owned_by_rental_association_ratio',
    'payment_frequency', 'coverage',
    # High missing postal code features (>60%)
    'postal_code_social_benefit_recipients_ratio',
    'postal_code_rental_houses_ratio',
    'postal_code_multi_family_houses_ratio',
    'postal_code_houses_built_before_1945_ratio',
    'postal_code_houses_built_45_to_65_ratio',
    'postal_code_houses_built_65_to_75_ratio',
    'postal_code_houses_built_75_to_85_ratio',
    'postal_code_houses_built_85_to_95_ratio',
    'postal_code_houses_built_95_to_05_ratio',
    'postal_code_houses_built_05_to_15_ratio',
    # Electric power columns (77%+ missing)
    'vehicle_net_max_power_electric',
    'vehicle_nominal_continuous_max_power',
]

# Categoricals to frequency-encode
FREQ_ENCODE = ['vehicle_maker', 'vehicle_model', 'postal_code',
               'province', 'municipality', 'vehicle_fuel_type',
               'vehicle_primary_color', 'vehicle_odometer_verdict_code',
               'postal_code_urban_category']

# Categoricals to target-encode (high cardinality, big impact)
TARGET_ENCODE = ['vehicle_maker', 'vehicle_model', 'postal_code']


def engineer_features(df):
    df = df.copy()

    # ── Dates → days ──
    date_cols = ['contractor_birthdate', 'vehicle_first_registration_date',
                 'vehicle_country_first_registration_date', 'vehicle_last_registration_date',
                 'vehicle_inspection_report_date', 'vehicle_inspection_expiry_date']
    ref = pd.Timestamp('2025-01-01')
    for col in date_cols:
        if col not in df.columns:
            continue
        parsed = pd.to_datetime(df[col], format='%d/%m/%Y', errors='coerce')
        if parsed.notna().sum() < len(df) * 0.1:
            parsed = pd.to_datetime(df[col], errors='coerce')
        df[col + '_days'] = (ref - parsed).dt.days
        df.drop(columns=[col], inplace=True)

    # ── Strings → numeric ──
    cat_keep = set(FREQ_ENCODE) | {
        'coverage', 'is_driver_owner', 'vehicle_is_imported',
        'vehicle_is_imported_within_last_12_months',
        'vehicle_can_be_registered', 'vehicle_has_open_recall',
        'vehicle_is_marked_for_export', 'vehicle_is_taxi',
        'payment_frequency', 'usage'}
    for col in df.columns:
        if col in cat_keep or col.startswith('Insurer_') or col == 'quote_id':
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # ── Driver ──
    if 'contractor_birthdate_days' in df.columns:
        df['driver_age'] = df['contractor_birthdate_days'] / 365.25
        df['driver_age_sq'] = df['driver_age'] ** 2
        df['driver_age_18_25'] = ((df['driver_age'] >= 18) & (df['driver_age'] < 25)).astype(float)
        df['driver_age_26_35'] = ((df['driver_age'] >= 25) & (df['driver_age'] < 35)).astype(float)
        df['driver_age_36_50'] = ((df['driver_age'] >= 35) & (df['driver_age'] < 50)).astype(float)
        df['driver_age_51_65'] = ((df['driver_age'] >= 50) & (df['driver_age'] < 65)).astype(float)
        df['driver_age_65plus'] = (df['driver_age'] >= 65).astype(float)

    # ── CFY ──
    if 'claim_free_years' in df.columns:
        cfy = pd.to_numeric(df['claim_free_years'], errors='coerce')
        df['claim_free_years'] = cfy
        df['cfy_negative'] = (cfy < 0).astype(float)
        df['cfy_0_2'] = ((cfy >= 0) & (cfy <= 2)).astype(float)
        df['cfy_3_5'] = ((cfy >= 3) & (cfy <= 5)).astype(float)
        df['cfy_6_10'] = ((cfy >= 6) & (cfy <= 10)).astype(float)
        df['cfy_10plus'] = (cfy > 10).astype(float)
        df['cfy_sq'] = cfy ** 2

    # ── Vehicle ──
    if 'vehicle_power' in df.columns and 'vehicle_net_weight' in df.columns:
        df['power_to_weight'] = df['vehicle_power'] / df['vehicle_net_weight'].replace(0, np.nan)
    if 'vehicle_value_new' in df.columns:
        df['log_vehicle_value'] = np.log1p(df['vehicle_value_new'])
        df['vehicle_value_sq'] = df['vehicle_value_new'] ** 2
    if 'vehicle_value_new' in df.columns and 'vehicle_age' in df.columns:
        df['value_per_age'] = df['vehicle_value_new'] / df['vehicle_age'].replace(0, np.nan)
    if 'vehicle_engine_size' in df.columns and 'vehicle_number_of_cylinders' in df.columns:
        df['displacement_per_cyl'] = df['vehicle_engine_size'] / df['vehicle_number_of_cylinders'].replace(0, np.nan)
    if 'vehicle_length' in df.columns and 'vehicle_width' in df.columns:
        df['vehicle_area'] = df['vehicle_length'] * df['vehicle_width']

    # ── Inspection ──
    if 'vehicle_inspection_expiry_date_days' in df.columns and 'vehicle_inspection_report_date_days' in df.columns:
        df['inspection_validity'] = df['vehicle_inspection_expiry_date_days'] - df['vehicle_inspection_report_date_days']

    # ── Coverage ──
    if 'coverage' in df.columns:
        df['coverage_ordinal'] = df['coverage'].map({'mtpl': 0, 'limited_casco': 1, 'casco': 2})
        df['is_mtpl'] = (df['coverage'] == 'mtpl').astype(float)
        df['is_limited_casco'] = (df['coverage'] == 'limited_casco').astype(float)
        df['is_casco'] = (df['coverage'] == 'casco').astype(float)

    # ── Booleans ──
    bool_map = {'True': 1, 'False': 0, 'true': 1, 'false': 0,
                True: 1, False: 0, '1': 1, '0': 0, 'yes': 1, 'no': 0}
    for col in ['is_driver_owner', 'vehicle_is_imported',
                'vehicle_is_imported_within_last_12_months',
                'vehicle_can_be_registered', 'vehicle_has_open_recall',
                'vehicle_is_marked_for_export', 'vehicle_is_taxi']:
        if col in df.columns:
            df[col] = df[col].map(bool_map)

    # ── Payment frequency ──
    if 'payment_frequency' in df.columns:
        df['payment_freq_ordinal'] = df['payment_frequency'].map(
            {'yearly': 0, 'half_yearly': 1, 'quarterly': 2, 'monthly': 3})

    # ── Frequency encoding ──
    for col in FREQ_ENCODE:
        if col in df.columns:
            freq = df[col].value_counts(normalize=True)
            df[col + '_freq'] = df[col].map(freq).astype(float)

    # ── KEY INTERACTIONS ──
    da = df.get('driver_age')
    co = df.get('coverage_ordinal')
    cfy = df.get('claim_free_years')
    vv = df.get('vehicle_value_new')
    vaage = df.get('vehicle_age')
    mc = df.get('municipality_crimes_per_1000')

    if da is not None and co is not None:
        df['age_x_coverage'] = da * co
    if cfy is not None and co is not None:
        df['cfy_x_coverage'] = cfy * co
    if vaage is not None and co is not None:
        df['veh_age_x_coverage'] = vaage * co
    if vv is not None and co is not None:
        df['value_x_coverage'] = vv * co
    if da is not None and vv is not None:
        df['age_x_value'] = da * vv / 1000
    if cfy is not None and vv is not None:
        df['cfy_x_value'] = cfy * vv / 1000
    if da is not None and cfy is not None:
        df['age_x_cfy'] = da * cfy
    if mc is not None and vv is not None:
        df['theft_risk'] = mc * vv / 1000

    # ── Drop ──
    to_drop = [c for c in DROP_EARLY if c in df.columns]
    to_drop += [c for c in FREQ_ENCODE if c in df.columns]
    df.drop(columns=to_drop, inplace=True)

    # ── Final numeric ──
    for col in df.columns:
        if col.startswith('Insurer_') or col == 'quote_id':
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


print("\nFeature engineering...")
t1 = time.time()

# Keep original categoricals for target encoding before FE drops them
train_cats = train_raw[['quote_id'] + TARGET_ENCODE].copy()
test2_cats = test2_raw[['quote_id'] + TARGET_ENCODE].copy()
test3_cats = test3_raw[['quote_id'] + TARGET_ENCODE].copy()
# Convert arrow strings
for col in TARGET_ENCODE:
    train_cats[col] = train_cats[col].astype(str).replace('<NA>', 'missing').fillna('missing')
    test2_cats[col] = test2_cats[col].astype(str).replace('<NA>', 'missing').fillna('missing')
    test3_cats[col] = test3_cats[col].astype(str).replace('<NA>', 'missing').fillna('missing')

train_fe = engineer_features(train_raw)
test2_fe = engineer_features(test2_raw)
test3_fe = engineer_features(test3_raw)
del train_raw, test2_raw, test3_raw
gc.collect()
print(f"  Done in {time.time()-t1:.1f}s")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. OUT-OF-FOLD TARGET ENCODING (no leakage!)
# ═══════════════════════════════════════════════════════════════════════════════
print("\nOut-of-fold target encoding...")
t_te = time.time()

N_TE_FOLDS = 5
te_kf = KFold(n_splits=N_TE_FOLDS, shuffle=True, random_state=123)
SMOOTHING = 50

te_features_per_insurer = {}

for ins in INSURERS:
    price_col = f"Insurer_{ins}_price"
    mask = train_fe[price_col].notna()
    prices = train_fe[price_col].values
    global_mean = train_fe.loc[mask, price_col].mean()

    te_cols = []
    for cat_col in TARGET_ENCODE:
        col_name = f'{cat_col}_te_{ins}'
        te_cols.append(col_name)

        # OOF encoding for train
        train_fe[col_name] = np.nan
        cat_values = train_cats[cat_col].values

        for fold_idx, (tr_idx, va_idx) in enumerate(te_kf.split(train_fe)):
            # Compute encoding only from training fold, only from quoted rows
            fold_mask = np.zeros(len(train_fe), dtype=bool)
            fold_mask[tr_idx] = True
            fold_mask = fold_mask & mask.values

            fold_cats = cat_values[fold_mask]
            fold_prices = prices[fold_mask]

            # Group by category
            df_tmp = pd.DataFrame({'cat': fold_cats, 'price': fold_prices})
            agg = df_tmp.groupby('cat')['price'].agg(['mean', 'count'])
            smooth_map = (agg['count'] * agg['mean'] + SMOOTHING * global_mean) / (agg['count'] + SMOOTHING)

            # Apply to validation fold
            va_cats = cat_values[va_idx]
            train_fe.iloc[va_idx, train_fe.columns.get_loc(col_name)] = (
                pd.Series(va_cats).map(smooth_map).fillna(global_mean).values
            )

        # For test: use ALL train data (quoted rows only)
        all_cats = cat_values[mask.values]
        all_prices = prices[mask.values]
        df_tmp = pd.DataFrame({'cat': all_cats, 'price': all_prices})
        agg = df_tmp.groupby('cat')['price'].agg(['mean', 'count'])
        smooth_map = (agg['count'] * agg['mean'] + SMOOTHING * global_mean) / (agg['count'] + SMOOTHING)

        test2_fe[col_name] = test2_cats[cat_col].map(smooth_map).fillna(global_mean).astype(float)
        test3_fe[col_name] = test3_cats[cat_col].map(smooth_map).fillna(global_mean).astype(float)

    te_features_per_insurer[ins] = te_cols

del train_cats, test2_cats, test3_cats
gc.collect()
print(f"  Done in {time.time()-t_te:.1f}s")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. FEATURE LISTS
# ═══════════════════════════════════════════════════════════════════════════════
base_features = [c for c in train_fe.columns
                 if c not in PRICE_COLS + DEDUCTIBLE_COLS + ['quote_id']
                 and not c.endswith(tuple(f'_te_{ins}' for ins in INSURERS))]

print(f"Base features: {len(base_features)}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. PER-INSURER CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
INSURER_CONFIG = {
    'A': {'num_leaves': 255, 'min_child_samples': 100},
    'B': {'num_leaves': 127, 'min_child_samples': 50},
    'C': {'num_leaves': 127, 'min_child_samples': 50},
    'D': {'num_leaves': 127, 'min_child_samples': 50},
    'E': {'num_leaves': 255, 'min_child_samples': 80},
    'F': {'num_leaves': 127, 'min_child_samples': 50},
    'G': {'num_leaves': 127, 'min_child_samples': 50},
    'H': {'num_leaves': 127, 'min_child_samples': 30},
    'I': {'num_leaves': 255, 'min_child_samples': 80},
    'J': {'num_leaves': 127, 'min_child_samples': 50},
    'K': {'num_leaves': 255, 'min_child_samples': 80},
}

# ═══════════════════════════════════════════════════════════════════════════════
# 6. TRAIN
# ═══════════════════════════════════════════════════════════════════════════════
def train_insurer(ins, train_df, test2_df, test3_df, base_feats, n_folds=3):
    t_ins = time.time()
    price_col = f"Insurer_{ins}_price"
    ded_col = f"Insurer_{ins}_deductible"

    print(f"\n  INSURER {ins}", end="")

    # Features
    ins_features = base_feats.copy()
    if ded_col in train_df.columns:
        ins_features.append(ded_col)
    ins_features.extend(te_features_per_insurer.get(ins, []))

    # Data
    mask = train_df[price_col].notna()
    X = train_df.loc[mask, ins_features].to_numpy(dtype=np.float32, na_value=np.nan)
    y = train_df.loc[mask, price_col].to_numpy(dtype=np.float64, na_value=np.nan)
    print(f": {len(y):,} rows, {len(ins_features)} features")

    cfg = INSURER_CONFIG[ins]
    params = {
        'objective': 'regression_l1',
        'metric': 'mae',
        'boosting_type': 'gbdt',

        'device': 'gpu',
        'max_bin': 127,

        'num_leaves': cfg['num_leaves'],
        'learning_rate': 0.05,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'min_child_samples': cfg['min_child_samples'],
        'reg_alpha': 0.3,
        'reg_lambda': 0.3,

        'verbose': -1
    }

    models = []
    maes = []
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
        dtrain = lgb.Dataset(X[tr_idx], label=y[tr_idx], feature_name=ins_features, free_raw_data=False)
        dval = lgb.Dataset(X[va_idx], label=y[va_idx], reference=dtrain, free_raw_data=False)

        model = lgb.train(
            params, dtrain, num_boost_round=3000, valid_sets=[dval],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        preds = model.predict(X[va_idx])
        mae = np.mean(np.abs(y[va_idx] - preds))
        maes.append(mae)
        models.append(model)
        print(f"    Fold {fold}: MAE={mae:.2f}, iters={model.best_iteration}")

    avg_mae = np.mean(maes)
    n_quoted = len(y)
    print(f"  CV MAE: {avg_mae:.2f} ({time.time()-t_ins:.0f}s)")

    # Predict
    X_t2 = test2_df[ins_features].to_numpy(dtype=np.float32, na_value=np.nan)
    X_t3 = test3_df[ins_features].to_numpy(dtype=np.float32, na_value=np.nan)
    pred2 = np.mean([m.predict(X_t2) for m in models], axis=0)
    pred3 = np.mean([m.predict(X_t3) for m in models], axis=0)

    return avg_mae, n_quoted, np.round(np.maximum(pred2, 1.0), 2), np.round(np.maximum(pred3, 1.0), 2)


print("\n" + "="*60)
print("TRAINING (v4 — fast, OOF target encoding)")
print("="*60)
t_all = time.time()

results2 = pd.DataFrame({'quote_id': test2_fe['quote_id']})
results3 = pd.DataFrame({'quote_id': test3_fe['quote_id']})
all_maes = {}
all_n = {}

for ins in INSURERS:
    mae, n, p2, p3 = train_insurer(ins, train_fe, test2_fe, test3_fe, base_features)
    all_maes[ins] = mae
    all_n[ins] = n
    results2[f"Insurer_{ins}_price"] = p2
    results3[f"Insurer_{ins}_price"] = p3

print(f"\nTraining done in {time.time()-t_all:.0f}s")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. SUMMARY & SAVE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("RESULTS (v4)")
print("="*60)
for ins in INSURERS:
    print(f"  Insurer {ins}: MAE = {all_maes[ins]:.2f}  (n={all_n[ins]:,})")

total_n = sum(all_n.values())
pooled = sum(all_maes[ins] * all_n[ins] for ins in INSURERS) / total_n
print(f"\n  POOLED MAE: {pooled:.2f}")

out2 = SUBMISSIONS / "block2_v4.csv"
out3 = SUBMISSIONS / "block3_v4.csv"
results2.to_csv(out2, sep=';', decimal='.', index=False)
results3.to_csv(out3, sep=';', decimal='.', index=False)

print(f"\n  Saved: {out2}")
print(f"  Saved: {out3}")
print(f"  Total time: {time.time()-t0:.0f}s")
print("  Done!")

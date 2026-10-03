import os, json, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone

warnings.filterwarnings('ignore')
BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
MODELS = os.path.join(BASE, 'models')
os.makedirs(MODELS, exist_ok=True)


def encoder():
    try:
        return OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown='ignore', sparse=True)


def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, default=float)


def band_series(series, labels=('Low', 'Normal', 'High')):
    q1, q2 = series.quantile([0.33, 0.66]).values
    return pd.cut(series, bins=[-np.inf, q1, q2, np.inf], labels=labels, include_lowest=True)


def train_crop_recommendation():
    """Farmer-friendly crop model.
    The original dataset contains N/P/K/pH. Farmers do not need to type those values.
    We convert the source data into broad observable categories and train models on those.
    """
    df = pd.read_csv(os.path.join(DATA, 'crop_recommendation.csv'))
    y = df['label'].astype(str).str.strip().str.lower()

    # Broad categories derived from the source distribution, not entered by the farmer.
    fertility_score = sum(df[c].rank(pct=True) for c in ['N', 'P', 'K']) / 3
    df['Soil_Fertility'] = pd.cut(fertility_score, [-np.inf, 0.33, 0.66, np.inf], labels=['Weak', 'Normal', 'Rich'])
    rainfall_level = band_series(df['rainfall'])
    humidity_level = band_series(df['humidity'])
    df['Rainfall_Level'] = rainfall_level
    # Water availability is a combined proxy using humidity + rainfall.
    water_score = 0.55 * df['humidity'].rank(pct=True) + 0.45 * df['rainfall'].rank(pct=True)
    df['Water_Availability'] = pd.cut(water_score, [-np.inf, 0.33, 0.66, np.inf], labels=['Low', 'Medium', 'Good'])
    df['Temperature_Level'] = band_series(df['temperature'])

    features = ['Soil_Fertility', 'Water_Availability', 'Rainfall_Level', 'Temperature_Level']
    X = df[features]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    pre = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), features)], remainder='drop')
    raw_models = {
        'Random Forest': RandomForestClassifier(n_estimators=350, random_state=42, n_jobs=-1, class_weight='balanced'),
        'Extra Trees': ExtraTreesClassifier(n_estimators=350, random_state=42, n_jobs=-1, class_weight='balanced'),
        'Logistic Regression': LogisticRegression(max_iter=2000, C=2.0)
    }
    results, fitted = {}, {}
    for name, model in raw_models.items():
        pipe = Pipeline([('pre', clone(pre)), ('model', model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        results[name] = {'accuracy': accuracy_score(y_test, pred), 'f1_weighted': f1_score(y_test, pred, average='weighted')}
        fitted[name] = pipe

    profiles = {}
    profile_df = df.copy()
    profile_df['crop'] = y.values
    for crop, group in profile_df.groupby('crop'):
        profiles[crop] = {
            'Soil_Fertility': str(group['Soil_Fertility'].mode().iloc[0]),
            'Water_Availability': str(group['Water_Availability'].mode().iloc[0]),
            'Rainfall_Level': str(group['Rainfall_Level'].mode().iloc[0]),
            'Temperature_Level': str(group['Temperature_Level'].mode().iloc[0])
        }

    selected = ['Random Forest', 'Extra Trees']
    joblib.dump(
        {'models': {k: fitted[k] for k in selected}, 'features': features, 'selected': selected,
         'classes': sorted(y.unique()), 'profiles': profiles},
        os.path.join(MODELS, 'crop_farmer_models.joblib'), compress=3
    )
    save_json({'module': 'Farmer Crop Recommendation', 'models': results, 'selected': selected, 'best': max(results, key=lambda k: results[k]['accuracy'])},
              os.path.join(MODELS, 'crop_farmer_metrics.json'))
    return results


def train_local_history():
    """Historical-location model from the large Indian crop production dataset."""
    df = pd.read_csv(os.path.join(DATA, 'indian_crop_production.csv'))
    df = df.dropna(subset=['State_Name', 'District_Name', 'Season', 'Crop', 'Area', 'Production', 'yield']).copy()
    for c in ['State_Name', 'District_Name', 'Season', 'Crop']:
        df[c] = df[c].astype(str).str.strip()

    # Historical profile: for each location + season, the crop with the highest observed
    # mean yield becomes the historical recommendation target.
    g = df.groupby(['State_Name', 'District_Name', 'Season', 'Crop'], as_index=False).agg(
        total_area=('Area', 'sum'), mean_yield=('yield', 'mean'), records=('Crop', 'size')
    )
    group_cols = ['State_Name', 'District_Name', 'Season']
    g['group_area_total'] = g.groupby(group_cols)['total_area'].transform('sum').replace(0, np.nan)
    g['area_share'] = (g['total_area'] / g['group_area_total']).fillna(0)
    profile = g.sort_values(['State_Name', 'District_Name', 'Season', 'mean_yield', 'total_area'],
                            ascending=[True, True, True, False, False])
    top = profile.groupby(group_cols, as_index=False).first()

    features = ['State_Name', 'District_Name', 'Season']
    X, y = top[features], top['Crop']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=42)
    pre = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), features)])
    raw_models = {
        'Extra Trees (Local History)': ExtraTreesClassifier(n_estimators=25, random_state=42, n_jobs=-1, class_weight='balanced'),
        'Logistic Regression (Local History)': LogisticRegression(max_iter=1200, C=2.0)
    }
    results, fitted = {}, {}
    for name, model in raw_models.items():
        pipe = Pipeline([('pre', clone(pre)), ('model', model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        results[name] = {'accuracy': accuracy_score(y_test, pred), 'f1_weighted': f1_score(y_test, pred, average='weighted')}
        fitted[name] = pipe

    # Compact lookup for exact historical matches and dropdowns.
    lookup = {}
    for _, row in profile.groupby(group_cols, sort=False).head(5).iterrows():
        key = f"{row['State_Name']}|||{row['District_Name']}|||{row['Season']}"
        lookup.setdefault(key, []).append({
            'crop': row['Crop'], 'mean_yield': float(row['mean_yield']),
            'area_share': float(row['area_share'])
        })

    options = {state: sorted(group['District_Name'].unique().tolist()) for state, group in df.groupby('State_Name')}
    selected = list(raw_models.keys())
    joblib.dump({'models': fitted, 'features': features, 'selected': selected, 'classes': sorted(y.unique()),
                 'lookup': lookup}, os.path.join(MODELS, 'crop_local_history_models.joblib'), compress=3)
    save_json({'module': 'Local History Crop Recommendation', 'models': results, 'selected': selected,
               'best': max(results, key=lambda k: results[k]['accuracy'])}, os.path.join(MODELS, 'crop_local_history_metrics.json'))
    save_json({'states': sorted(options), 'districts': options, 'seasons': sorted(df['Season'].unique().tolist())},
              os.path.join(MODELS, 'farmer_options.json'))
    return results


def train_yield():
    """Yield prediction with simple farmer inputs only."""
    df = pd.read_csv(os.path.join(DATA, 'crop_yield.csv')).rename(columns={'Yield': 'target'})
    df = df[df['target'] >= 0].copy()
    for c in ['Season', 'State', 'Crop']:
        df[c] = df[c].astype(str).str.strip()
    df['Rainfall_Level'] = band_series(df['Annual_Rainfall'])
    df['Temperature_Level'] = band_series(df['Avg_Temperature'])
    # Area is deliberately excluded from the yield model. It is used only after prediction to
    # estimate harvest quantity, preventing the model from learning a field-size artefact.
    features = ['Crop', 'Season', 'State', 'Rainfall_Level', 'Temperature_Level']
    X, y = df[features], np.log1p(df['target'])
    cat = features
    pre = ColumnTransformer([('cat', encoder(), cat)], remainder='drop')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=42)
    raw_models = {
        'Random Forest': RandomForestRegressor(n_estimators=90, random_state=42, n_jobs=-1, min_samples_leaf=3),
        'Extra Trees': ExtraTreesRegressor(n_estimators=90, random_state=42, n_jobs=-1, min_samples_leaf=3),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42, n_estimators=90, max_depth=3, learning_rate=.05, loss='huber')
    }
    results, fitted = {}, {}
    for name, model in raw_models.items():
        pipe = Pipeline([('pre', clone(pre)), ('model', model)])
        pipe.fit(X_train, y_train)
        pred = np.maximum(0, np.expm1(pipe.predict(X_test)))
        true = np.expm1(y_test)
        results[name] = {'mae': mean_absolute_error(true, pred), 'rmse': mean_squared_error(true, pred) ** .5, 'r2': r2_score(true, pred)}
        fitted[name] = pipe

    # Local baseline for explainability.
    baseline = df.groupby(['State', 'Crop', 'Season'], as_index=False)['target'].median().rename(columns={'target': 'median_yield'})
    crop_baseline = df.groupby('Crop', as_index=False)['target'].median().rename(columns={'target': 'median_yield'})
    joblib.dump({'models': fitted, 'features': features, 'selected': ['Random Forest', 'Extra Trees', 'Gradient Boosting'],
                 'target_transform': 'log1p', 'rainfall_thresholds': list(df['Annual_Rainfall'].quantile([.33, .66]).values),
                 'temperature_thresholds': list(df['Avg_Temperature'].quantile([.33, .66]).values)},
                os.path.join(MODELS, 'yield_prediction.joblib'), compress=3)
    baseline.to_pickle(os.path.join(MODELS, 'yield_baseline.pkl'))
    crop_baseline.to_pickle(os.path.join(MODELS, 'crop_baseline.pkl'))
    save_json({'module': 'Farmer-Friendly Yield Prediction', 'models': results, 'selected': list(raw_models),
               'best': max(results, key=lambda k: results[k]['r2'])}, os.path.join(MODELS, 'yield_metrics.json'))
    return results


def train_price():
    df = pd.read_csv(os.path.join(DATA, 'crop_price.csv'))
    df['month'] = pd.to_datetime(df['month'])
    df = df.sort_values(['commodity_name', 'month']).copy()
    g = df.groupby('commodity_name')['avg_modal_price']
    for lag in [1, 3, 6, 12]:
        df[f'lag_{lag}'] = g.shift(lag)
    df['rolling_3'] = df.groupby('commodity_name')['avg_modal_price'].transform(lambda s: s.shift(1).rolling(3).mean())
    df['year'] = df['month'].dt.year
    df['month_num'] = df['month'].dt.month
    df['trend'] = (df['month'] - pd.Timestamp('2010-01-01')).dt.days / 30.4375
    df['sin_month'] = np.sin(2 * np.pi * df['month_num'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month_num'] / 12)
    df = df.dropna(subset=['lag_12', 'rolling_3']).copy()
    features = ['commodity_name', 'year', 'month_num', 'trend', 'sin_month', 'cos_month', 'lag_1', 'lag_3', 'lag_6', 'lag_12', 'rolling_3']
    cutoff = df['month'].quantile(.80)
    train, test = df[df['month'] < cutoff], df[df['month'] >= cutoff]
    pre = ColumnTransformer([('cat', encoder(), ['commodity_name']), ('num', SimpleImputer(strategy='median'), [c for c in features if c != 'commodity_name'])])
    raw_models = {
        'Random Forest': RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1, min_samples_leaf=2),
        'Extra Trees': ExtraTreesRegressor(n_estimators=120, random_state=42, n_jobs=-1, min_samples_leaf=2),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42, n_estimators=120, max_depth=3, learning_rate=.04, loss='huber')
    }
    results, fitted = {}, {}
    for name, model in raw_models.items():
        pipe = Pipeline([('pre', clone(pre)), ('model', model)])
        pipe.fit(train[features], train['avg_modal_price'])
        pred = np.maximum(0, pipe.predict(test[features]))
        results[name] = {'mae': mean_absolute_error(test['avg_modal_price'], pred), 'rmse': mean_squared_error(test['avg_modal_price'], pred) ** .5, 'r2': r2_score(test['avg_modal_price'], pred)}
        fitted[name] = pipe
    selected = ['Random Forest', 'Extra Trees', 'Gradient Boosting']
    joblib.dump({'models': fitted, 'features': features, 'selected': selected, 'latest_month': df['month'].max().strftime('%Y-%m-%d')},
                os.path.join(MODELS, 'price_prediction.joblib'), compress=3)
    df.to_pickle(os.path.join(MODELS, 'price_features.pkl'))
    save_json({'module': 'Crop Price Forecast', 'models': results, 'selected': selected,
               'best': max(results, key=lambda k: results[k]['r2']), 'latest_month': df['month'].max().strftime('%Y-%m-%d')},
              os.path.join(MODELS, 'price_metrics.json'))
    return results


if __name__ == '__main__':
    print('1/4 Training farmer-friendly crop model...')
    print(train_crop_recommendation())
    print('2/4 Training local-history crop model (large dataset)...')
    print(train_local_history())
    print('3/4 Training farmer-friendly yield models...')
    print(train_yield())
    print('4/4 Training crop price models...')
    print(train_price())
    print('\nAll farmer-friendly models saved in models/')

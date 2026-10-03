import os, json, sqlite3
from functools import wraps
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
from werkzeug.security import generate_password_hash, check_password_hash

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'smart-agri-college-project-change-me')
DB = os.path.join(BASE, 'instance', 'users.db')
MODELS = os.path.join(BASE, 'models')
DATA = os.path.join(BASE, 'data')


def db():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    con = db()
    con.execute('CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, password TEXT NOT NULL, created_at TEXT NOT NULL)')
    cur = con.execute('SELECT 1 FROM users WHERE username=?', ('student',))
    if not cur.fetchone():
        con.execute('INSERT INTO users(username,password,created_at) VALUES(?,?,?)', ('student', generate_password_hash('student123'), datetime.now().isoformat()))
    con.commit(); con.close()


def login_required(fn):
    @wraps(fn)
    def wrap(*a, **kw):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return fn(*a, **kw)
    return wrap


def load_json(name, default=None):
    p = os.path.join(MODELS, name)
    if not os.path.exists(p):
        return default
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def model_path(name):
    p = os.path.join(MODELS, name)
    if not os.path.exists(p):
        raise FileNotFoundError(f'Model file not found: {name}. Run python train_models.py first.')
    return p


def pct_scores(items):
    vals = np.array([x['confidence'] for x in items], dtype=float)
    if len(vals) == 0:
        return items
    total = vals.sum()
    if total > 0:
        for x in items:
            x['confidence'] = round(float(x['confidence'] / total * 100), 2)
    return items


def top_probabilities(model, X, topn=5):
    probs = model.predict_proba(X)[0]
    classes = model.classes_
    series = pd.Series(probs, index=classes).sort_values(ascending=False).head(topn)
    return [{'crop': str(k).title(), 'raw_crop': str(k), 'confidence': round(float(v) * 100, 2)} for k, v in series.items()]


def safe_float(v):
    return float(v) if v not in (None, '') else 0.0


def farmer_score(value, mapping):
    return mapping.get(value, 60)


def advisory_for_crop(inputs, crop, profiles):
    profile = profiles.get(crop.lower(), {})
    checks = {
        'Soil health': (inputs['Soil_Fertility'], profile.get('Soil_Fertility')),
        'Water availability': (inputs['Water_Availability'], profile.get('Water_Availability')),
        'Rainfall': (inputs['Rainfall_Level'], profile.get('Rainfall_Level')),
        'Temperature': (inputs['Temperature_Level'], profile.get('Temperature_Level')),
    }
    match = sum(1 for a, b in checks.values() if a == b)
    fit = round(match / len(checks) * 100, 1)
    advice = []
    if inputs['Water_Availability'] == 'Low':
        advice.append('Water is limited. Plan irrigation carefully and prefer crops that tolerate lower water availability.')
    if inputs['Rainfall_Level'] == 'Low':
        advice.append('Rainfall is low. Keep a backup irrigation plan and protect the soil from drying.')
    if inputs['Temperature_Level'] == 'Hot':
        advice.append('Weather is hot. Check crop heat tolerance and keep moisture available around the root zone.')
    if inputs['Soil_Fertility'] == 'Weak':
        advice.append('Soil fertility looks weak from your selected field condition. Consider locally recommended soil-improvement practices.')
    if not advice:
        advice.append('The selected conditions are reasonably aligned with the recommended crop profile.')
    return fit, advice, profile


@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('landing.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        u = request.form.get('username', '').strip(); p = request.form.get('password', '')
        con = db(); row = con.execute('SELECT * FROM users WHERE username=?', (u,)).fetchone(); con.close()
        if row and check_password_hash(row['password'], p):
            session['user_id'] = row['id']; session['username'] = u
            return redirect(url_for('dashboard'))
        flash('Invalid username or password.', 'error')
    return render_template('auth.html', mode='login')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        u = request.form.get('username', '').strip(); p = request.form.get('password', '')
        if len(u) < 3 or len(p) < 6:
            flash('Username must be 3+ characters and password 6+ characters.', 'error')
            return render_template('auth.html', mode='register')
        try:
            con = db(); con.execute('INSERT INTO users(username,password,created_at) VALUES(?,?,?)', (u, generate_password_hash(p), datetime.now().isoformat())); con.commit(); con.close()
            flash('Account created. Please log in.', 'success'); return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already exists.', 'error')
    return render_template('auth.html', mode='register')


@app.route('/logout')
def logout():
    session.clear(); return redirect(url_for('index'))


@app.route('/dashboard')
@login_required
def dashboard():
    required = ['crop_farmer_models.joblib', 'crop_local_history_models.joblib', 'yield_prediction.joblib', 'price_prediction.joblib']
    if not all(os.path.exists(os.path.join(MODELS, x)) for x in required):
        flash('Models are not trained yet. Run: python train_models.py', 'error')
    options = load_json('farmer_options.json', {'states': [], 'districts': {}, 'seasons': [], 'yield_crops': []})
    if not options.get('yield_crops'):
        yp = os.path.join(DATA, 'crop_yield.csv')
        if os.path.exists(yp):
            options['yield_crops'] = sorted(pd.read_csv(yp, usecols=['Crop'])['Crop'].dropna().astype(str).str.strip().unique().tolist())
    price = load_json('price_metrics.json', {}) or {}
    commodities = []
    price_path = os.path.join(DATA, 'crop_price.csv')
    if os.path.exists(price_path):
        commodities = sorted(pd.read_csv(price_path)['commodity_name'].dropna().astype(str).unique().tolist())
    return render_template('dashboard.html', user=session.get('username'), options=options, metrics={
        'crop': load_json('crop_farmer_metrics.json'),
        'history': load_json('crop_local_history_metrics.json'),
        'yield': load_json('yield_metrics.json'),
        'price': price
    }, commodities=commodities)


@app.post('/api/crop-recommend')
@login_required
def crop_recommend():
    weather_pack = joblib.load(model_path('crop_farmer_models.joblib'))
    local_pack = joblib.load(model_path('crop_local_history_models.joblib'))
    inputs = {
        'Soil_Fertility': request.form['Soil_Fertility'],
        'Water_Availability': request.form['Water_Availability'],
        'Rainfall_Level': request.form['Rainfall_Level'],
        'Temperature_Level': request.form['Temperature_Level'],
    }
    Xw = pd.DataFrame([inputs], columns=weather_pack['features'])
    weather_model = weather_pack['models']['Random Forest']
    weather_top = top_probabilities(weather_model, Xw)

    loc = pd.DataFrame([{
        'State_Name': request.form['State_Name'],
        'District_Name': request.form['District_Name'],
        'Season': request.form['Season']
    }])
    local_model = local_pack['models']['Extra Trees (Local History)']
    local_top = top_probabilities(local_model, loc)

    # Hybrid recommendation: equal-weight probabilities from the weather/soil model and local-history model.
    w = pd.Series(weather_model.predict_proba(Xw)[0], index=weather_model.classes_)
    h = pd.Series(local_model.predict_proba(loc)[0], index=local_model.classes_)
    combined = pd.concat([w, h], axis=1).fillna(0).mean(axis=1).sort_values(ascending=False)
    combined_top = [{'crop': str(k).title(), 'raw_crop': str(k), 'confidence': round(float(v) * 100, 2)} for k, v in combined.head(5).items()]

    key = f"{request.form['State_Name']}|||{request.form['District_Name']}|||{request.form['Season']}"
    history_rows = local_pack.get('lookup', {}).get(key, [])[:5]
    history_exact = [{**x, 'crop': str(x['crop']).title(), 'mean_yield': round(float(x['mean_yield']), 3), 'area_share': round(float(x['area_share']) * 100, 1)} for x in history_rows]

    best_raw = combined.index[0]
    fit, advice, profile = advisory_for_crop(inputs, best_raw, weather_pack.get('profiles', {}))
    analysis = [
        {'label': 'Soil health', 'score': farmer_score(inputs['Soil_Fertility'], {'Weak': 35, 'Normal': 68, 'Rich': 92}), 'value': inputs['Soil_Fertility'], 'icon': '🌿'},
        {'label': 'Water', 'score': farmer_score(inputs['Water_Availability'], {'Low': 35, 'Medium': 68, 'Good': 94}), 'value': inputs['Water_Availability'], 'icon': '💧'},
        {'label': 'Rainfall', 'score': farmer_score(inputs['Rainfall_Level'], {'Low': 35, 'Normal': 72, 'High': 90}), 'value': inputs['Rainfall_Level'], 'icon': '🌧️'},
        {'label': 'Temperature', 'score': farmer_score(inputs['Temperature_Level'], {'Low': 55, 'Normal': 92, 'High': 65}), 'value': inputs['Temperature_Level'], 'icon': '🌡️'},
        {'label': 'Crop profile match', 'score': fit, 'value': f'{fit:.0f}% match', 'icon': '🌾'},
    ]
    return jsonify({'ok': True, 'best': combined_top[0], 'combined': combined_top, 'weather_model': weather_top,
                    'local_model': local_top, 'history_exact': history_exact, 'analysis': analysis,
                    'advice': advice, 'profile': profile, 'location': loc.iloc[0].to_dict()})


@app.post('/api/yield-predict')
@login_required
def yield_predict():
    pack = joblib.load(model_path('yield_prediction.joblib'))
    vals = {
        'Crop': request.form['Crop'],
        'Season': request.form['Season'],
        'State': request.form['State'],
        'Rainfall_Level': request.form['Rainfall_Level'],
        'Temperature_Level': request.form['Temperature_Level']
    }
    X = pd.DataFrame([vals], columns=pack['features'])
    preds = []
    for name, model in pack['models'].items():
        value = max(0.0, float(np.expm1(model.predict(X)[0])))
        preds.append({'name': name, 'value': round(value, 4)})
    ensemble = float(np.mean([x['value'] for x in preds]))
    area_acres = safe_float(request.form.get('Area_Acres', 0))
    area_hectares = area_acres * 0.404686
    harvest = ensemble * area_hectares if area_hectares else 0

    baseline_df = pd.read_pickle(os.path.join(MODELS, 'yield_baseline.pkl'))
    exact = baseline_df[(baseline_df['State'] == vals['State']) & (baseline_df['Crop'] == vals['Crop']) & (baseline_df['Season'] == vals['Season'])]
    state_crop = baseline_df[(baseline_df['State'] == vals['State']) & (baseline_df['Crop'] == vals['Crop'])]
    crop_base = pd.read_pickle(os.path.join(MODELS, 'crop_baseline.pkl'))
    crop_only = crop_base[crop_base['Crop'] == vals['Crop']]
    if len(exact):
        baseline = float(exact['median_yield'].iloc[0]); baseline_source = 'local state + crop + season history'
    elif len(state_crop):
        baseline = float(state_crop['median_yield'].median()); baseline_source = 'local state + crop history'
    elif len(crop_only):
        baseline = float(crop_only['median_yield'].iloc[0]); baseline_source = 'national crop history'
    else:
        baseline = None; baseline_source = 'not available'

    analysis = [
        {'label': 'Land size', 'score': min(100, max(15, area_acres / 5 * 10 if area_acres else 15)), 'value': f'{area_acres:g} acres', 'icon': '📐'},
        {'label': 'Rainfall level', 'score': farmer_score(vals['Rainfall_Level'], {'Low': 35, 'Normal': 75, 'High': 88}), 'value': vals['Rainfall_Level'], 'icon': '🌧️'},
        {'label': 'Temperature level', 'score': farmer_score(vals['Temperature_Level'], {'Low': 58, 'Normal': 92, 'High': 67}), 'value': vals['Temperature_Level'], 'icon': '🌡️'},
        {'label': 'AI agreement', 'score': max(0, 100 - abs(preds[0]['value'] - preds[1]['value']) / max(1, ensemble) * 100), 'value': 'Higher = models agree more', 'icon': '🤝'},
    ]
    return jsonify({'ok': True, 'prediction': round(ensemble, 4), 'models': preds, 'area_acres': area_acres,
                    'area_hectares': round(area_hectares, 3), 'estimated_harvest': round(harvest, 4),
                    'baseline': round(baseline, 4) if baseline is not None else None,
                    'baseline_source': baseline_source, 'analysis': analysis,
                    'explanation': 'Yield prediction uses crop, season, state and broad rainfall/temperature levels. Land size is used only to convert predicted yield into an estimated harvest amount.'})


@app.post('/api/price-forecast')
@login_required
def price_forecast():
    pack = joblib.load(model_path('price_prediction.joblib'))
    hist = pd.read_pickle(os.path.join(MODELS, 'price_features.pkl'))
    commodity = request.form['commodity']
    h = hist[hist['commodity_name'] == commodity].sort_values('month')
    if len(h) < 12:
        return jsonify({'ok': False, 'error': 'Not enough history for this commodity.'}), 400
    last = h.iloc[-1]
    last_date = pd.Timestamp(last['month'])
    target = last_date + pd.offsets.MonthBegin(1)
    vals = {'commodity_name': commodity, 'year': target.year, 'month_num': target.month,
            'trend': (target - pd.Timestamp('2010-01-01')).days / 30.4375,
            'sin_month': np.sin(2 * np.pi * target.month / 12), 'cos_month': np.cos(2 * np.pi * target.month / 12),
            'lag_1': float(last['avg_modal_price']), 'lag_3': float(h.iloc[-3]['avg_modal_price']),
            'lag_6': float(h.iloc[-6]['avg_modal_price']), 'lag_12': float(h.iloc[-12]['avg_modal_price']),
            'rolling_3': float(h.iloc[-3:]['avg_modal_price'].mean())}
    X = pd.DataFrame([vals], columns=pack['features'])
    preds = []
    for name, model in pack['models'].items():
        value = max(0.0, float(model.predict(X)[0]))
        preds.append({'name': name, 'value': round(value, 2)})
    ensemble = float(np.mean([x['value'] for x in preds]))
    history = h.tail(12)[['month', 'avg_modal_price']].copy()
    history['month'] = history['month'].dt.strftime('%b %y')
    chart = [{'month': r['month'], 'price': round(float(r['avg_modal_price']), 2)} for _, r in history.iterrows()]
    change_pct = ((ensemble - float(last['avg_modal_price'])) / max(0.01, float(last['avg_modal_price']))) * 100
    analysis = [
        {'label': 'Recent price level', 'score': 80, 'value': f"₹{float(last['avg_modal_price']):,.0f}", 'icon': '💰'},
        {'label': '3-month movement', 'score': min(100, max(0, abs((float(last['avg_modal_price']) - float(h.iloc[-3]['avg_modal_price'])) / max(1, float(h.iloc[-3]['avg_modal_price'])) * 100) * 3)), 'value': f"{((float(last['avg_modal_price']) - float(h.iloc[-3]['avg_modal_price'])) / max(1, float(h.iloc[-3]['avg_modal_price'])) * 100):+.1f}%", 'icon': '📊'},
        {'label': 'Model agreement', 'score': max(0, 100 - np.std([x['value'] for x in preds]) / max(1, ensemble) * 300), 'value': 'Closer bars = stronger agreement', 'icon': '🤝'},
        {'label': 'Forecast change', 'score': min(100, max(0, 50 + change_pct)), 'value': f'{change_pct:+.1f}%', 'icon': '🔮'},
    ]
    return jsonify({'ok': True, 'commodity': commodity, 'last_month': last_date.strftime('%b %Y'),
                    'last_price': round(float(last['avg_modal_price']), 2), 'forecast_month': target.strftime('%b %Y'),
                    'prediction': round(ensemble, 2), 'models': preds, 'history': chart,
                    'analysis': analysis, 'note': 'Forecast is based on the latest month contained in the supplied dataset, not live market data.'})


@app.get('/api/metrics')
@login_required
def metrics():
    return jsonify({'crop': load_json('crop_farmer_metrics.json'), 'history': load_json('crop_local_history_metrics.json'),
                    'yield': load_json('yield_metrics.json'), 'price': load_json('price_metrics.json')})


init_db()
if __name__ == '__main__':
    app.run(debug=True)

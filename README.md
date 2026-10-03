# SmartAgri AI — Farmer-Friendly Agricultural Prediction Web App

A college mini-project that combines multiple machine-learning models with a simple farmer-facing dashboard.

## What changed in Farmer Mode

The application no longer asks the farmer to type laboratory-style **N, P, K or pH values**.

### Crop Recommendation inputs
- State — choose from a list
- District — choose from a list that updates automatically after selecting the state
- Season — Kharif, Rabi, Summer, etc.
- Soil health — Weak / Normal / Rich, explained in simple language
- Water availability — Low / Medium / Good
- Rainfall — Low / Normal / High
- Temperature — Cool / Normal / Hot
- Land size — acres

### Yield Prediction inputs
- Crop
- State
- Season
- Land size in acres
- Rainfall level
- Temperature level

### Price Forecast inputs
- Crop/commodity name only

## Models

### Crop recommendation
1. Random Forest on simple soil/weather categories derived from the supplied crop recommendation dataset.
2. Extra Trees using historical state + district + season profiles from the supplied large Indian crop production dataset.
3. Logistic Regression is also trained as an additional comparison model.
4. The dashboard shows two independent model viewpoints and a combined recommendation.

### Yield prediction
1. Random Forest Regressor
2. Extra Trees Regressor
3. Gradient Boosting Regressor

The yield model intentionally does **not** use `Production` as an input. Land size is also not used as a predictor of yield; it is used after the prediction to calculate an estimated harvest for the farmer.

### Crop price forecasting
1. Random Forest Regressor
2. Extra Trees Regressor
3. Gradient Boosting Regressor

The price split is chronological. Historical lag and rolling features are calculated within each commodity so future values do not leak into the training features.

## Dashboard

Each prediction opens a detailed analysis dashboard containing:
- combined result
- two model predictions shown separately
- model comparison bars
- farmer-input analysis bars
- local historical reference where available
- practical advisory notes
- recent price trend chart for price forecasting
- transparent note explaining that the forecast is based on the supplied datasets

## Run the project

### 1. Open the project folder
Extract `SmartAgri_AI_Farmer_Mode.zip` and open the `smart_agri_ai` folder in VS Code.

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

### 3. Activate it

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks scripts for the current terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 4. Install packages

```powershell
pip install -r requirements.txt
```

### 5. Train models

The optimized training script is designed to be practical on a normal college laptop.

```powershell
python train_models.py
```

The trained files are created inside `models/`.

### 6. Start Flask

```powershell
python app.py
```

Open:

`http://127.0.0.1:5000`

## Demo login

Username: `student`

Password: `student123`

You can register another account from the application.

## Example farmer workflow

1. Open **Choose Crop**.
2. Select the state and district.
3. Choose the growing season.
4. Select soil health based on the previous crop's performance.
5. Select water availability.
6. Select expected rainfall and general temperature.
7. Enter the field size in acres.
8. Click **Find suitable crops**.
9. Read the two model predictions, combined recommendation, input graph and local history.
10. Use **Estimate Yield** for the selected crop.
11. Use **Check Price** to see the data-based next-month price forecast.

## Important project limitation

This is an academic decision-support system trained only on the supplied historical datasets. It is not connected to live weather, live mandi prices or a live agricultural advisory service. Local weather conditions, soil tests and agricultural-department advice should still be considered before making real planting or financial decisions.

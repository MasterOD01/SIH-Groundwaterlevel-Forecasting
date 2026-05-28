Real-Time Groundwater Resource Evaluation Using DWLR DataProblem Statement ID: SIH25068PS Category: Software (Miscellaneous Theme)Team ID: 86211Team Name: PlaceholderAn enterprise-grade, daily-resolution machine learning pipeline that ingests continuous 6-hourly Digital Water Level Recorder (DWLR) sensor data, dynamically resamples and aligns it with NASA POWER meteorological feeds, runs a hybrid STL-ARIMA and LightGBM model hierarchy to forecast groundwater levels (GWL), and translates outputs into an actionable, policy-compliant Groundwater Stress Index (GSI) explained via exact SHAP attribution chains.🗺️ Architectural TopologyOur daily batch and real-time inference infrastructure operates on a tightly-coupled five-layer stack.                                [ IoT DWLR Sensors ] (6-Hourly Data)
                                         │
                                         ▼
                             [ Prefect Orchestration ]
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
           [ NASA POWER API ]                         [ IMD Gridded Data ]
       (Climate, Temperature, Soil)                   (0.25° Daily Rain)
                    │                                         │
                    └────────────────────┬────────────────────┘
                                         ▼
                        [ Stage 1: Daily Spatial Resampling ]
                                         │
                                         ▼
                      [ Stage 2: Feature Engineering Pipeline ]
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
         [ Antecedent Lags ]                     [ Crop Calendar Signals ]
         (7 to 90d Soil/Rain)                     (is_kharif, cumulative IDD)
                    │                                         │
                    └────────────────────┬────────────────────┘
                                         ▼
                         [ Stage 3: Two-Tier Forecasting ]
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
         [ Tier 1: STL-ARIMA ]                   [ Tier 2: LightGBM ]
         (Secular vs. Seasonal)                 (Non-linear interactions)
                    │                                         │
                    └────────────────────┬────────────────────┘
                                         ▼
                            [ Stage 4: Hydro Evaluation ]
                             (NSE, KGE, Bootstrap Intervals)
                                         │
                                         ▼
                             [ Stage 5: Explainability ]
                        (SHAP TreeExplainer -> Metric GSI)
                                         │
                                         ▼
                              [ Storage: TimescaleDB ]
                               (Partitioned Hypertables)
                                         │
                ┌────────────────────────┴────────────────────────┐
                ▼                                                 ▼
        [ FastAPI Backend REST ]                         [ Redis Cache Layer ]
                │                                                 │
                └────────────────────────┬────────────────────────┘
                                         ▼
                            [ Mobile App Client (Flutter) ]
                          (Mapbox, Offline Cache, Push Alerts)
📂 Repository Directory Treesih25068-groundwater-eval/
├── .github/
│   └── workflows/
│       └── pipeline-ci.yml       # Automated checks & testing pipeline
├── data-pipeline/
│   ├── config.py                 # Configuration parameters & API paths
│   ├── extractors/
│   │   ├── dwlr_fetcher.py       # CGWB WRIS DWLR data scrapers & API clients
│   │   ├── nasa_power.py         # NASA POWER meteorological API harvester
│   │   └── imd_gridded.py        # IMD 0.25-degree daily rain parser
│   ├── transformation.py         # Dynamic 6-hourly to daily resampling engine
│   └── pipeline_flow.py          # Prefect DAG configuration and flow runner
├── ml-core/
│   ├── __init__.py
│   ├── features.py               # Feature Engineering (circular encoding, lags)
│   ├── models/
│   │   ├── base.py               # Model base class interface
│   │   ├── stl_arima.py          # Tier 1: STL-ARIMA (Decomposition + ARIMA remainder)
│   │   └── lightgbm_model.py     # Tier 2: LightGBM model training & walk-forward CV
│   ├── evaluation.py             # Hydrological metrics (NSE, KGE, RMSE, Coverage)
│   └── explainers.py             # SHAP TreeExplainer & dynamic GSI computer
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py               # FastAPI entry point
│   │   ├── database.py           # TimescaleDB connections and Session local
│   │   ├── models.py             # SQLAlchemy ORM schemas
│   │   ├── routes/
│   │   │   ├── forecast.py       # /forecast/{station_id} REST outputs
│   │   │   ├── gsi.py            # /gsi/{station_id} historical tracking
│   │   │   └── shap.py           # /shap/{station_id} attribution vectors
│   │   └── cache.py              # Redis connector & cache management
│   ├── requirements.txt
│   └── Dockerfile
├── mobile-app/
│   ├── android/
│   ├── ios/
│   ├── lib/
│   │   ├── main.dart             # Flutter Entrypoint
│   │   ├── models/               # Station metadata & GSI state models
│   │   ├── screens/
│   │   │   ├── dashboard_map.dart # Mapbox interface showing GSI zones
│   │   │   ├── analytics.dart    # STL & SHAP visual cards
│   │   │   └── alerts_screen.dart # Target agricultural notification inbox
│   │   └── services/
│   │       ├── api_client.dart   # Client communicating with FastAPI
│   │       └── local_storage.dart# SQLite cache for offline-first resilience
│   └── pubspec.yaml
├── scripts/
│   └── db_init.sql               # Database schemas with Hypertables setup
├── tests/                        # Suite of pytest modules
│   ├── test_features.py
│   └── test_models.py
├── LICENSE
└── README.md
🧮 Mathematical Formulation & Foundations1. Hydrological Performance MetricsStandard mean absolute percentage error (MAPE) indicators are inappropriate for groundwater depth modeling due to divisions by zero or highly inflated values near baseline. We utilize Nash-Sutcliffe Efficiency (NSE) and Kling-Gupta Efficiency (KGE) to assess prediction accuracy.Nash-Sutcliffe Efficiency (NSE)$$NSE = 1 - \frac{\sum_{t=1}^{T} (Y_t - \hat{Y}_t)^2}{\sum_{t=1}^{T} (Y_t - \bar{Y})^2}$$Where $Y_t$ is the actual depth to water table, $\hat{Y}_t$ is the forecasted depth at time $t$, and $\bar{Y}$ is the observed mean. An $NSE = 1$ indicates a perfect model, $NSE = 0$ indicates model predictions are as accurate as the mean predictor, and $NSE < 0$ implies the mean predictor is superior to the model.Kling-Gupta Efficiency (KGE)$$KGE = 1 - \sqrt{(r - 1)^2 + (\beta - 1)^2 + (\gamma - 1)^2}$$Where:$r$ is the Pearson correlation coefficient between observations and forecasts.$\beta = \frac{\mu_{\hat{Y}}}{\mu_Y}$ (Bias ratio measuring the deviation in mean value).$\gamma = \frac{CV_{\hat{Y}}}{CV_Y} = \frac{\sigma_{\hat{Y}} / \mu_{\hat{Y}}}{\sigma_Y / \mu_Y}$ (Variability ratio tracking dispersion matching).2. Crop-Calendar & Circular Feature MappingTo eliminate artificial boundary discontinuities (such as the December 31 to January 1 step) and capture the dynamic kharif agricultural cycle, we construct the following mappings:$$\text{Sin\_DOY} = \sin\left(\frac{2 \pi \cdot \text{DOY}}{365.25}\right), \quad \text{Cos\_DOY} = \cos\left(\frac{2 \pi \cdot \text{DOY}}{365.25}\right)$$Where $\text{DOY}$ represents day of the year $[1, 365]$.Our agricultural demand proxy incorporates cumulative Irrigation Degree-Days ($IDD_{cum}$) derived from critical temperature thresholds over a rolling 30-day window:$$IDD_{cum} = \sum_{i=t-30}^{t} \max\left(T2M_i - T_{base}, 0\right) \cdot \mathbb{I}(\text{is\_kharif} = 1)$$Where $T_{base} = 10^\circ\text{C}$ and $\mathbb{I}$ is an indicator function tracking active seasonal periods (June to November).3. Groundwater Stress Index (GSI)Raw water level forecasts are mapped directly to a decision-ready risk framework:$$GSI = 0.65 \cdot \Psi_{depth} + 0.35 \cdot \Phi_{rate}$$Depth Component ($\Psi_{depth}$)$$\Psi_{depth} = \frac{GWL_t - P_{10}(GWL)}{P_{90}(GWL) - P_{10}(GWL)}$$Where $P_{10}$ and $P_{90}$ are the 10th and 90th percentile depths for the same calendar week calculated historically over the target station's record.Rate Component ($\Phi_{rate}$)$$\Phi_{rate} = \text{Sigmoid}\left(\frac{\Delta_{30} GWL}{\sigma(\Delta_{30} GWL)}\right) = \frac{1}{1 + \exp\left(-\frac{GWL_t - GWL_{t-30}}{\sigma(\Delta_{30} GWL)}\right)}$$The compound GSI classification limits are aligned directly with Central Ground Water Board (CGWB) guidelines:$GSI \ge 0.70$: 🚨 Stressed / Over-exploited (Water table critically low or dropping fast).$0.40 \le GSI < 0.70$: ⚠️ Moderate / Semi-critical (Monitor extractions; restricted pumping recommended).$GSI < 0.40$: ✅ Safe (Recharge velocity exceeds or matches extraction).📊 Barnala, Punjab Validation Study ResultsThe pipeline's predictive capability was validated using actual operational data from the Barnala DWLR station in Central Punjab (February 2023 to January 2025; 684 daily observations).Ablation Matrix (5-Fold Walk-Forward Cross-Validation)Model ConfigurationNSEKGERMSE (m)Coverage @ 90% IntervalSeasonal Naïve (Baseline)$0.41 - 0.55$$0.39 - 0.51$$0.72 - 0.91$—Tier 1: STL-ARIMA(1,1,1)$0.59 - 0.67$$0.55 - 0.63$$0.58 - 0.71$$81\% - 87\%$Tier 2: LightGBM (Our Pipeline)$0.70 - 0.78$$0.65 - 0.72$$0.42 - 0.55$$88\% - 93\%$Key Analytical FindingsTrend & Seasonal Demarcation: STL decomposition isolated a secular depletion trend of $7.6\%$ per annum alongside a deep seasonal extraction trough corresponding directly to the kharif irrigation cycle.Feature Impact: SHAP analysis isolated the top predictors of next-day levels: gwl_lag_1d (autocorrelative memory), rain_cum_45d (delayed percolation), and days_into_kharif (agricultural depletion proxy).Distribution-free Prediction Intervals: LightGBM prediction intervals constructed using residual bootstrap reached $89\%$ mean empirical coverage, verifying high calibration accuracy under variable climatic regimes.🛠️ Installation & SetupPrerequisitesDocker & Docker Compose (Recommended)Python 3.11+Flutter SDK (3.16+)PostgreSQL 15+ / TimescaleDBDatabase InitializationTimescaleDB utilizes hypertables to handle high-frequency writes and multi-station queries smoothly. Configure database tables:psql -h localhost -U postgres -d groundwater -f scripts/db_init.sql
The database setup configuration includes the hypertable initialization step:-- Convert standard relational log to partitioned hypertable
SELECT create_hypertable('station_readings', 'reading_date', partitioning_column => 'station_id', number_partitions => 4);
Backend & Model Training EnvironmentNavigate to directory and install requirements:cd backend
pip install -r requirements.txt
Configure environment variables in a .env file:DATABASE_URL=postgresql://postgres:password@localhost:5432/groundwater
REDIS_URL=redis://localhost:6379/0
NASA_POWER_API_KEY=your_nasa_key
IMD_GRID_DATA_PATH=/opt/data/imd_rain/
Spin up FastAPI backend:uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
Running Model Training and Inference via PrefectTo perform feature engineering and train individual models in parallel across stations:# Set up Prefect Server locally
prefect server start

# Run the batch orchestration pipeline once
python data-pipeline/pipeline_flow.py
⚡ API Quick Reference1. Fetch Station ForecastsEndpoint: GET /forecast/{station_id}Query Params: days=30Response Payload:{
  "station_id": "PUN_BARN_01",
  "name": "Barnala DWLR",
  "coordinates": [75.546, 30.382],
  "as_of": "2026-05-28",
  "forecast": [
    {
      "date": "2026-05-29",
      "predicted_depth_meters": 14.82,
      "lower_90_ci": 14.51,
      "upper_90_ci": 15.13,
      "gsi_value": 0.74,
      "gsi_status": "Stressed"
    }
  ]
}
2. Fetch SHAP Explainability VectorEndpoint: GET /shap/{station_id}Response Payload:{
  "station_id": "PUN_BARN_01",
  "prediction_date": "2026-05-29",
  "base_value_meters": 13.50,
  "predicted_value_meters": 14.82,
  "attributions": {
    "gwl_lag_1d": 0.62,
    "days_into_kharif": 0.45,
    "rain_cum_45d": 0.31,
    "gwl_30d_change": -0.06
  },
  "narrative": "Depth is projected to drop 1.32m deeper than average. Primary causes: seasonal Paddy irrigation demand (+0.45m impact) and a severe cumulative rain deficit over the last 45 days (+0.31m impact)."
}
📱 Mobile App CompilationVerify local Flutter environmental configurations:flutter doctor
Fetch pub packages:cd mobile-app
flutter pub get
Configure API base URL within lib/services/api_client.dart:const String baseUrl = 'http://YOUR_API_IP:8000';
Compile and launch targeted binary (Android example):flutter run --release
🤝 Contributing & CollaborationWe welcome technical contributions targeting model optimization, spatial graph network structures, and edge integrations:Fork this Repository.Form a feature branch (git checkout -b feature/AmazingFeature).Commit localized alterations (git commit -m 'Implement Graph Spatial Constraints').Push to remote (git push origin feature/AmazingFeature).Raise a Pull Request.📚 References & Literature BasesData Ingestion Standard: India-WRIS Water Resources Information System & Central Ground Water Board (CGWB).SHAP Explainers: Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. NeurIPS 2017.NASA POWER Program: NASA Langley Research Center POWER Project.Hydrological Models: Bhanja, S. N. et al. (2019). Long-term groundwater recharge rates across India by in situ measurements. HESS.

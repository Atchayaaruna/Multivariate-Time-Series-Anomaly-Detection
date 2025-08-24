# MLP Windowed Autoencoder for Anomaly Detection

## 📌 Overview
This project implements an **MLP Windowed Autoencoder** to detect anomalies in multivariate time-series data such as industrial process data. It combines:
- **Sliding Window Technique** for temporal context
- **MLP Autoencoder** for reconstruction error-based anomaly scoring
- **Percentile-based scoring (0–100)**
- **Feature contribution analysis** to explain anomalies

---

## ✅ Success Criteria
- Code runs without errors on provided dataset
- Produces columns:
  - `anomaly_score_0_100`
  - `top_feature_1 ... top_feature_k`
- Training period anomaly scores:
  - Mean < 10
  - Max < 25
- Handles edge cases:
  - All normal data → Low scores (0–20)
  - Training anomalies → Warn but continue
  - Require ≥72 hours of data (warn if less)
  - Single feature dataset → Supported
  - Perfect predictions → Add small noise
- Code follows **PEP 8**, modular, documented

---

Technologies Used

Python 3

TensorFlow/Keras

Pandas, NumPy

Scikit-learn

Plotly, Streamlit

## 📂 Project Structure
ANOMALYDETECTIONPROJECT/
├── artifacts/ # Saved model and scaler
│ ├── autoencoder_model.h5 # Trained MLP Autoencoder
│ ├── metadata.json # Metadata (window size, features)
│ └── scaler.pkl # MinMax Scaler
├── data/
│ ├── raw_dataset.csv # Place your dataset
│ └── output_with_anomalies.csv # Output with anomaly scores
├── scripts/
│ └── mlp_windowed_autoencoder_pipeline.py # Main training & scoring pipeline
├── dashboard.py # Streamlit dashboard
├── README.md # Project documentation
└── requirements.txt # Python dependencies

1) Create a project folder(Anomalydetectionproject)

2) Create and Activate Virtual Environment
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate

3) Install Dependencies
pip install -r requirements.txt

4) How to run:
python scripts/mlp_windowed_autoencoder_pipeline.py \
  --input data/81ce1f00-c3f4-4baa-9b57-006fad1875adTEP_Train_Test.csv \
  --output data/output_with_anomalies.csv \
  --window 60 \
  --epochs 50 \
  --batch_size 64 \
  --save_model

5) Launch the dashboard:
streamlit run dashboard.py
Access the app at: http://localhost:8501


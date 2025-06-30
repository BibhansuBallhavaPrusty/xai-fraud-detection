# Explainable AI for Fraud Detection

This project is an end-to-end machine learning pipeline for detecting fraudulent transactions using XGBoost and SHAP, with a Streamlit dashboard.

##  Tech Stack
- Python
- Scikit-learn
- XGBoost
- SHAP
- Streamlit
- Docker

##  Run Locally
```bash
git clone https://github.com/your-username/xai-fraud-detection.git
cd xai-fraud-detection
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

##  Run with Docker
```bash
docker-compose build
docker-compose up
```

⚠️ Note: Dataset file is not included due to size. Please download it from:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Save as: data/creditcard.csv

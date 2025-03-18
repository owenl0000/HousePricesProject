# House Price Prediction App

This is a Streamlit-based web application that allows users to input property details and receive a predicted house price based on a trained machine learning model.

## Deployment Link
https://housepricespredictionproject-kaggle-owenl0000.streamlit.app/


## Features
- User-friendly interface for inputting property features
- Supports categorical and numerical inputs
- Automatic computation for engineered features 
- Machine learning model integration for real-time predictions
- Deployed online for easy access

## Technologies Used
- **Frontend & Deployment:** Streamlit
- **Backend:** Python
- **Machine Learning:** Scikit-learn, XGBoost, LightGBM
- **Data Processing:** Pandas, NumPy
- **Model Deployment:** Pickle for loading trained models

## Setup Instructions
### 1. Clone the Repository
```bash
git clone https://github.com/owenl0000/HousePricesProject.git
cd HousePricesProject
```

### 2. Install Dependencies
Create a virtual environment (optional but recommended) and install the required packages:
```bash
pip install -r requirements.txt
```

### 3. Run the App Locally
```bash
streamlit run app.py
```

### 4. Deploying the App
The app is deployed on Streamlit Cloud. You can deploy it by:
1. Creating a Streamlit account and linking your GitHub repository.
2. Ensuring `requirements.txt` includes all necessary dependencies.
3. Setting up the repository as a Streamlit app.


## Model Details
The app uses a stacking ensemble model trained with multiple regressors, including:
- Linear Regression
- XGBoost Regressor
- Gradient Boosting Regressor
- Random Forest Regressor
- Ridge Regression
- Stacking Regressor
- Voting Regressor

But due to limited computing power for deployment, the deployed version uses only XGBoost Regressor

To switch between XGBoost and Stacking Regressor for local machine switch the pkl files from stacking_model.pkl and xgboost_model.pkl




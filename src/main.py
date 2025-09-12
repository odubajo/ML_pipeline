from fastapi import FastAPI
import pickle
import pandas as pd 
from data_model import Water

app = FastAPI(
    title="Water Potability Prediction",
    description="predicting water potability",
)

with open("/Users/qoyyum/Desktop/ML_pipeline/model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/")
def index():
    return "Welcome to the Water Potability Prediction API"

@app.post("/predict")
def model_predict(water : Water):
    sample = pd.DataFrame({
        'ph': [water.ph],
        'Hardness': [water.Hardness],
        'Solids': [water.Solids],
        'Chloramines': [water.Chloramines],
        'Sulfate': [water.Sulfate],
        'Conductivity': [water.Conductivity],
        'Organic_carbon': [water.Organic_carbon],
        'Trihalomethanes': [water.Trihalomethanes],
        'Turbidity': [water.Turbidity]
    })

    predicted_value = model.predict(sample)

    if predicted_value == 1:
        return "water is potable"
    else:
        return "water is not potable"
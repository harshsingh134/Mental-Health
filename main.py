import warnings
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sklearn.exceptions import InconsistentVersionWarning

# Suppress minor sklearn patch-version warning when unpickling
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Mental_Health_Model.pkl"
model = joblib.load(MODEL_PATH)

top_countries = ['Other', 'India', 'USA', 'Canada', 'Australia', 'UK', 'Germany', 'Mexico', 'Turkey', 'France']
top_countries_map = {c.lower(): c for c in top_countries}
country_aliases = {
    "united states": "USA",
    "united states of america": "USA",
    "us": "USA",
    "usa": "USA",
    "united kingdom": "UK",
    "uk": "UK",
    "great britain": "UK",
    "britain": "UK",
    "england": "UK",
}

app = FastAPI(title="Student Mental Health Signal API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Model for incoming prediction payload
class StudentData(BaseModel):
    age                     : int = Field(..., ge=10, le=100)
    gender                  : Literal['Male', 'Female']
    country                 : str
    academic_level          : Literal['Undergraduate', 'Graduate', 'High School']
    most_used_platform      : Literal['Facebook', 'LinkedIn', 'Instagram', 'Snapchat', 'Twitter', 'YouTube', 'TikTok', 'LINE', 'KakaoTalk', 'VKontakte', 'WhatsApp', 'WeChat']
    purpose_of_use          : Literal['Networking', 'Education', 'Entertainment', 'News']
    avg_daily_usage_hours   : float = Field(..., ge=0, le=24)
    daily_unlocks           : int   = Field(..., ge=0)
    study_hours             : float = Field(..., ge=0, le=24)
    physical_activity_hours : float = Field(..., ge=0, le=24)
    sleep_hours_per_night   : float = Field(..., ge=0, le=24)
    stress_level            : Literal['Medium', 'Low', 'Very High', 'High']


# Pydantic Model for prediction output
class PredictionResponse(BaseModel):
    predicted_mental_health_score: float


@app.get('/')
def greet(request: Request):
    accept = request.headers.get("accept", "")
    index_file = BASE_DIR / "index.html"
    if "text/html" in accept and index_file.exists():
        return FileResponse(index_file)
    return {"message": "Welcome to Sheryians AI School Guys"}


@app.get('/style.css')
def get_style():
    style_file = BASE_DIR / "style.css"
    if style_file.exists():
        return FileResponse(style_file, media_type="text/css")
    return {"error": "Not found"}


@app.get('/script.js')
def get_script():
    script_file = BASE_DIR / "script.js"
    if script_file.exists():
        return FileResponse(script_file, media_type="application/javascript")
    return {"error": "Not found"}


@app.post('/predict', response_model=PredictionResponse)
def predict(data: StudentData):
    country_clean = data.country.strip().lower()
    if country_clean in country_aliases:
        country_group = country_aliases[country_clean]
    elif country_clean in top_countries_map:
        country_group = top_countries_map[country_clean]
    else:
        country_group = "Other"

    input_row = pd.DataFrame([{
        'Study_Hours'               : data.study_hours,
        'Age'                       : data.age,
        'Avg_Daily_Usage_Hours'     : data.avg_daily_usage_hours,
        'Daily_Unlocks'             : data.daily_unlocks,
        'Physical_Activity_Hours'   : data.physical_activity_hours,
        'Sleep_Hours_Per_Night'     : data.sleep_hours_per_night,
        'Stress_Level'              : data.stress_level,
        'Gender'                    : data.gender,
        'Academic_Level'            : data.academic_level,
        'Most_Used_Platform'        : data.most_used_platform,
        'Purpose_Of_Use'            : data.purpose_of_use,
        'Grouped_country'           : country_group,
    }])

    raw_prediction = float(model.predict(input_row)[0])
    clamped_score = max(0.0, min(10.0, raw_prediction))
    return PredictionResponse(predicted_mental_health_score=round(clamped_score, 2))
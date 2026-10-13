# Mansik-Santulan-Score (Mental Health Signal)

A machine learning web application that predicts a student's mental health wellness signal (scale 0–10) based on their social media habits, screen time, study rhythm, physical activity, sleep, and stress levels.

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.10+ installed and the virtual environment set up:
```bash
# If needed, create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\activate   # On Windows
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Backend Server
Start the FastAPI server using Uvicorn:
```bash
uvicorn main:app --reload
```
By default, the server runs on [http://127.0.0.1:8000](http://127.0.0.1:8000).

### 4. Open the Web Application
You can interact with the application in two ways:
- **Direct in browser:** Navigate to [http://127.0.0.1:8000/](http://127.0.0.1:8000/) (FastAPI serves the full interactive UI).
- **Standalone:** Double-click or open `index.html` in your browser. The frontend automatically discovers and connects to your local FastAPI backend.
- **Interactive API Docs:** Visit [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to explore OpenAPI / Swagger documentation.

---

## 🛠️ API Endpoints

### `GET /`
- When accessed from a web browser (`Accept: text/html`), returns the interactive UI.
- When accessed programmatically (`Accept: application/json`), returns:
  ```json
  { "message": "Welcome to Sheryians AI School Guys" }
  ```

### `POST /predict`
Accepts student habit and lifestyle telemetry, predicts mental wellness score, and returns the score.

#### Example Request
```json
{
  "age": 21,
  "gender": "Male",
  "country": "India",
  "academic_level": "Undergraduate",
  "most_used_platform": "YouTube",
  "purpose_of_use": "Education",
  "avg_daily_usage_hours": 3.5,
  "daily_unlocks": 50,
  "study_hours": 4.0,
  "physical_activity_hours": 1.5,
  "sleep_hours_per_night": 7.0,
  "stress_level": "Medium"
}
```

#### Example Response
```json
{
  "predicted_mental_health_score": 7.42
}
```

---

## 🧠 Machine Learning Model
- **Algorithm:** Random Forest Regressor Pipeline with `ColumnTransformer` preprocessing.
- **Features Handled:**
  - `Study_Hours` (Log1p transformed + Standard Scaler)
  - `Age`, `Avg_Daily_Usage_Hours`, `Daily_Unlocks`, `Physical_Activity_Hours`, `Sleep_Hours_Per_Night` (Standard Scaler)
  - `Stress_Level` (Ordinal Encoded: Low, Medium, High, Very High)
  - `Gender`, `Academic_Level`, `Most_Used_Platform`, `Purpose_Of_Use`, `Grouped_country` (One-Hot Encoded)
- **Target:** Mental Health Score (continuous score ~0–10).

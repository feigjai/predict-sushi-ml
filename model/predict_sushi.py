import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import joblib

# --- 1. Utility Functions for Holidays ---
def compute_school_holidays(year):
    # Example for WA 2025, update as needed for future
    return [
        (f'{year}-04-12', f'{year}-04-27'),
        (f'{year}-07-05', f'{year}-07-20'),
        (f'{year}-09-27', f'{year}-10-12'),
        (f'{year}-12-19', f'{year+1}-02-01')
    ]

def compute_public_holidays(year):
    return [
        f'{year}-01-01', f'{year}-01-27', f'{year}-03-03', f'{year}-04-18', f'{year}-04-20',
        f'{year}-04-21', f'{year}-04-25', f'{year}-06-02', f'{year}-09-29', f'{year}-12-25', f'{year}-12-26'
    ]

def is_school_holiday(date, school_hol_ranges):
    return any(start <= date <= end for (start, end) in school_hol_ranges)

def get_day_type(date, public_hol_list):
    # Returns 'PublicHoliday', 'DayBefore', 'DayAfter', or 'Other'
    for d in public_hol_list:
        d = datetime.strptime(d, "%Y-%m-%d").date()
        if date == d: return 'PublicHoliday'
        if date == d - timedelta(days=1): return 'DayBefore'
        if date == d + timedelta(days=1): return 'DayAfter'
    return 'Other'

# --- 2. Fetch today's forecast from OpenWeatherMap ---
def get_weather_forecast(city="Perth,AU", api_key="YOUR_OPENWEATHERMAP_API_KEY"):
    url = (f"https://api.openweathermap.org/data/2.5/forecast"
           f"?q={city}&cnt=1&units=metric&appid={api_key}")
    r = requests.get(url)
    data = r.json()
    if "list" in data and len(data["list"]) > 0:
        forecast = data["list"][0]
        max_temp = forecast["main"]["temp_max"]
        # You may want to sum/avg over the whole day's periods
        rain = forecast.get("rain", {}).get("3h", 0)
        # For more accurate features, extend with actual day's intervals
        clouds = forecast["clouds"]["all"]
        return max_temp, rain, clouds
    else:
        raise Exception("Failed to fetch weather forecast")

# --- 3. Main Prediction Routine ---
def predict_todays_sushi():
    # 1. Load data & model for moving averages
    sales = pd.read_csv('Square-item-import-data-Square-Sales-202425.csv')
    sales['Date'] = pd.to_datetime(sales['Date'], errors='coerce').dt.date
    sales = sales[sales['Category'].str.contains('sushi', case=False, na=False)]
    sushi_daily = sales.groupby('Date')['Qty'].sum().reset_index().sort_values('Date')
    
    # 2. Holidays
    today = datetime.now().date()
    sch_hols = [(datetime.strptime(start, "%Y-%m-%d").date(),
                 datetime.strptime(end, "%Y-%m-%d").date())
                for start, end in compute_school_holidays(today.year)]
    pub_hols = compute_public_holidays(today.year)
    is_school = is_school_holiday(today, sch_hols)
    day_type = get_day_type(today, pub_hols)
    
    # 3. Weather
    API_KEY = "YOUR_OPENWEATHERMAP_API_KEY"  # <- Put your real API key here
    max_temp, rain, cloud = get_weather_forecast(api_key=API_KEY)

    # 4. Features
    prev_1 = sushi_daily['Qty'].iloc[-1] if len(sushi_daily) else 0
    avg_7 = sushi_daily['Qty'].tail(7).mean() if len(sushi_daily) >=7 else sushi_daily['Qty'].mean()
    month = today.month
    dow = today.weekday()
    season = {12:0, 1:0, 2:0, 3:1, 4:1, 5:1, 6:2, 7:2, 8:2, 9:3, 10:3, 11:3}[month]
    temp_band = 3 if max_temp > 38 else 2 if max_temp > 28 else 0 if max_temp < 20 else 1
    rain_band = int(rain >= 0.2)
    sunny = int(cloud < 3)
    cloudy = int(cloud >= 6)
    features = {
        'TempBand': temp_band, 'RainBand': rain_band, 'Sunny': sunny, 'Cloudy': cloudy,
        'Month': month, 'DayOfWeek': dow, 'Season': season,
        'PublicHoliday': int(day_type == 'PublicHoliday'),
        'DayBeforePH': int(day_type == 'DayBefore'),
        'DayAfterPH': int(day_type == 'DayAfter'),
        'SchoolHoliday': int(is_school),
        'prev_1': prev_1, 'avg_7': avg_7
    }
    X_today = pd.DataFrame([features])
    
    # 5. Load model and predict
    model = joblib.load('sushi_rf_predictor.joblib')
    pred = model.predict(X_today)[0]
    
    # 6. Print / report result
    print(f\"\\n--- Sushi Opening Stock Prediction ({today}) ---\")
    print(f\"Weather: Max Temp={max_temp:.1f}°C, Rain={rain:.1f}mm, Clouds={cloud}\")
    print(f\"School Holiday: {'Yes' if is_school else 'No'} | Public Holiday type: {day_type}\")
    print(f\"Recommended opening sushi stock: {int(round(pred))} units\\n\")
    # Could email, push to Google Sheets, or text message here!
    return int(round(pred))

if __name__ == '__main__':
    predict_todays_sushi()

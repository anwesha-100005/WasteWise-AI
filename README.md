# 🌿 WasteWise AI
**IBM SkillsBuild AI Internship Project · SDG 12: Responsible Consumption and Production**

> Predicts food wastage for events using a Random Forest model and delivers AI-powered (Groq LLM) recommendations to reduce waste.

---

## 📁 Project Structure

```
wastewise-ai/
├── backend/
│   ├── app.py                   ← Flask API (prediction + Groq recommendations)
│   ├── requirements.txt         ← Python dependencies
│   ├── Procfile                 ← Render/Heroku start command
│   ├── rf_model.pkl             ← Trained Random Forest model
│   ├── reference_columns.pkl    ← Feature column names
│   └── recommendation_data.pkl  ← Data-driven recommendation triggers
├── frontend/
│   └── index.html               ← Single-page prediction UI
├── notebook/
│   └── FoodWastage.ipynb        ← Full Colab notebook (EDA → Model → Recommendations)
└── README.md
```

---

## 🚀 Deployment

| Layer    | Platform | URL (after deploy)                        |
|----------|----------|-------------------------------------------|
| Backend  | Render   | `https://wastewise-backend.onrender.com`  |
| Frontend | Vercel   | `https://wastewise-ai.vercel.app`         |

See **DEPLOYMENT_GUIDE.md** for step-by-step instructions.

---

## 🔑 Environment Variables

| Variable       | Where to set | Description               |
|----------------|--------------|---------------------------|
| `GROQ_API_KEY` | Render → Environment | Free key from [console.groq.com](https://console.groq.com) |

---

## 🔗 API Endpoints

| Method | Endpoint   | Description                            |
|--------|------------|----------------------------------------|
| GET    | `/`        | Health check                           |
| POST   | `/predict` | Returns predicted wastage + recommendations |

### Sample POST /predict request
```json
{
  "Type of Food": "Meat",
  "Number of Guests": 300,
  "Event Type": "Wedding",
  "Quantity of Food": 400,
  "Storage Conditions": "Refrigerated",
  "Purchase History": "Bulk",
  "Seasonality": "Summer",
  "Preparation Method": "Buffet",
  "Geographical Location": "Urban",
  "Pricing": 25
}
```

---

*Built with Python · Flask · scikit-learn · Groq LLaMA 3.3 · IBM SkillsBuild 2025*

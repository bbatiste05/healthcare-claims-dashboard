# 🏥 Healthcare Claims Cost Analyzer

An interactive Streamlit dashboard that analyzes mock healthcare claims data to identify cost drivers, patient risks, and potential fraud.

### 🔗 [Live App on Streamlit](https://healthcare-claims-dashboardgit-7xsa69mraafreb2w77scbc.streamlit.app/)  
### 💻 [Source Code on GitHub](https://github.com/bbatiste05/healthcare-claims-dashboard)

---

## 💡 Features
- 📊 **Top ICD and CPT Cost Analysis** — spot high-cost diagnoses and procedures
- 🧍 **Risk Scoring Dashboard** — flag high-risk patients using ICD-10 codes
- 💰 **Cost Anomaly Detection** — detect providers with outlier billing behavior using Z-scores
- 🕵️ **Fraud Detection Tool** — identify providers making excessive claims per patient
- 📂 **Custom File Upload** — load your own claims dataset to test live dashboards
- 📈 **Interactive Visuals** — built with Plotly for zoomable, insightful charts
- 🛠 Built with Python, Pandas, and Streamlit

---

📂 How to Use This App
This Streamlit dashboard requires CSV file uploads to function — no sample data is loaded by default.

🏥 Claims Dashboard Tabs (Tabs 1–4)

Upload a claims dataset via the sidebar
Expected columns:
provider_id
charge_amount
cpt (optional for CPT audit tab)
These power:
🧍 Risk Scoring
💰 Cost Anomalies
🕵️ Fraud Detection
💥 CPT Charge Audit
🤖 No-Show Predictor (Tab 5)

Upload a separate CSV within the main window of the "No-Show Predictor" tab
Expected columns:
no_show (values should be "Yes" or "No")
gender, age, wait_days, diabetic, hypertensive, sms_received
The model performs:
Logistic Regression
Confusion Matrix + Report
Feature Importance
Per-patient Prediction Scoring
✅ 2. Future Enhancements for the No-Show Model

Ideas you can gradually integrate:

🔘 Add patient filtering by age/gender
🌳 Use RandomForestClassifier or XGBoost for comparison
🧪 Add SHAP/Explainability for feature interpretation
📈 Create time-series views by appointment_day
✅ 3. GitHub & Streamlit Sharing Tips

✅ Include Screenshots: Add visuals from each tab
✅ Add Live Demo: Link to your Streamlit Cloud app
✅ Post to Streamlit Community: Write a short project showcase
✅ Consider GitHub Topics:
#healthcare
#streamlit
#machine-learning
#claims-analysis


## 📁 Project Structure
.
├── app.py # Main Streamlit app with tabbed navigation
├── load_data.py # Handles file loading and cleaning
├── mock_claims.csv # Sample dataset (replaceable)
├── requirements.txt # Dependencies (Streamlit, Pandas, Plotly)
├── dashboards/ # Modular dashboards
│ ├── risk_scoring.py
│ ├── cost_anomalies.py
│ └── fraud_detection.py
└── README.md

## 🖼️ Screenshots


Run Locally
# Clone the repo
git clone https://github.com/bbatiste05/healthcare-claims-dashboard.git
cd healthcare-claims-dashboard

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

🧰 Tech Stack

Tool	Purpose
Python 3	Data manipulation and backend logic
Pandas	Data cleaning and grouping
Plotly	Interactive visualizations
Streamlit	Dashboard UI + app deployment
📌 Author

Brandon Batiste
Healthcare Technology + Data Specialist
🔗 linkedin.com/in/brandonbatiste

📄 License

This project is open-source under the MIT License.


---




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




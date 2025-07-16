import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def run(df=None):
    st.subheader("🤖 No-Show Prediction Model")

    # If no data provided, let user upload
    if df is None:
        st.info("Upload a CSV file or use the sample no-show dataset.")
        uploaded_file = st.file_uploader("Upload No-Show CSV", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_csv("no_show.csv")
            st.info("Using default no_show.csv")

    st.write("### Preview of Data")
    st.dataframe(df.head())

    # Step 1: Normalize column names
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace("-", "_").str.replace("\r", "").str.replace("\n", "")



    # Check actual column names early
    if 'no_show' not in df.columns:
        st.error("❌ 'no_show' column not found. Here's what we found instead:")
        st.write(df.columns.tolist())
        st.stop()


    # Step 2: Validate required column
    if 'no_show' not in df.columns:
        st.error("❌ Required column 'no_show' not found. Please check the header in your CSV.")
        st.stop()

    # Step 3: Encode target and categorical features
    df['no_show'] = df['no_show'].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})
    df['gender'] = LabelEncoder().fit_transform(df['gender'])

    feature_cols = ['gender', 'age', 'wait_days', 'diabetic', 'hypertensive', 'sms_received']
    target_col = 'no_show'

    X = df[feature_cols]
    y = df[target_col]

    # Step 4: Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Step 5: Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    st.write("### 🔍 Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

    st.write("### 📊 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Show', 'No-Show'], yticklabels=['Show', 'No-Show'], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.write("### 🔬 Feature Importance (Model Coefficients)")
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.coef_[0]
    }).sort_values(by='Importance', key=abs, ascending=False)

    fig2 = px.bar(importance_df, x='Feature', y='Importance', title="Logistic Regression Feature Importance")
    st.plotly_chart(fig2, use_container_width=True)

    st.write("### 🧾 Patient-Level Predictions")
    results_df = X_test.copy()
    results_df['Actual'] = y_test.values
    results_df['Predicted'] = y_pred
    results_df['Predicted_Prob'] = y_prob
    st.dataframe(results_df.sort_values("Predicted_Prob", ascending=False).reset_index(drop=True))


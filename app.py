import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt


# Load saved objects
model = joblib.load("kraljic_model.joblib")
feature_names = joblib.load("kraljic_features.joblib")
label_encoder = joblib.load("kraljic_label_encoder.joblib")

st.set_page_config(
    page_title="Kraljic Matrix Decision Support App",
    layout="wide"
)

st.title("📦 Kraljic Category Prediction System")
st.write("Procurement decision support tool using machine learning.")


# Sidebar
st.sidebar.header("About")
st.sidebar.write("""
This application predicts the Kraljic procurement category of a product
based on supplier risk, profit impact and operational indicators.
""")

mode = st.sidebar.radio(
    "Select Mode",
    ["Single Prediction", "Batch Prediction"]
)


# SINGLE PREDICTION 
if mode == "Single Prediction":

    st.subheader("🧾 Product & Supplier Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        supplier_region = st.selectbox(
            "Supplier Region",
            ["Asia", "Europe", "North America", "South America", "Africa", "Global"]
        )
        single_source = st.selectbox("Single Source Risk", ["No", "Yes"])

    with col2:
        lead_time = st.number_input("Lead Time (Days)", 1, 365, 45)
        order_volume = st.number_input("Order Volume (Units)", 1, 100000, 5000)

    with col3:
        cost = st.number_input("Cost per Unit", 1.0, 100000.0, 500.0)
        env_impact = st.slider("Environmental Impact Score", 1, 5, 3)

    st.subheader("📊 Risk & Impact Scores")

    col4, col5 = st.columns(2)

    with col4:
        supply_risk = st.slider("Supply Risk Score", 1, 5, 3)

    with col5:
        profit_impact = st.slider("Profit Impact Score", 1, 5, 3)


  # Encoding
    single_source_val = 1 if single_source == "Yes" else 0

    region_cols = {
        "Supplier_Region_Africa": 0,
        "Supplier_Region_Asia": 0,
        "Supplier_Region_Europe": 0,
        "Supplier_Region_Global": 0,
        "Supplier_Region_North America": 0,
        "Supplier_Region_South America": 0
    }

    key = f"Supplier_Region_{supplier_region}"
    if key in region_cols:
        region_cols[key] = 1

    input_dict = {
        "Lead_Time_Days": lead_time,
        "Order_Volume_Units": order_volume,
        "Cost_per_Unit": cost,
        "Supply_Risk_Score": supply_risk,
        "Profit_Impact_Score": profit_impact,
        "Environmental_Impact": env_impact,
        "Single_Source_Risk": single_source_val
    }

    input_dict.update(region_cols)

    input_vector = np.array([[input_dict.get(col, 0) for col in feature_names]])

    warnings = []
    if lead_time > 120:
        warnings.append("Very high lead time.")
    if supply_risk >= 4:
        warnings.append("High supply risk detected.")
    if profit_impact >= 4:
        warnings.append("High profit impact item.")

    for w in warnings:
        st.warning(w)

    
    # Prediction
    
    if st.button("🔍 Predict Kraljic Category"):

        pred = model.predict(input_vector)[0]
        proba = model.predict_proba(input_vector)[0]

        label = label_encoder.inverse_transform([pred])[0]

        st.subheader("✅ Prediction Result")
        st.metric("Predicted Category", label)

        
        # Probability chart
      
        st.subheader("📈 Class Probability Distribution")

        fig, ax = plt.subplots()
        ax.bar(label_encoder.classes_, proba)
        ax.set_ylabel("Probability")
        ax.set_xlabel("Category")
        st.pyplot(fig)

       
        
        
        # Summary
     
        st.subheader("📄 Input Summary")

        st.write({
            "Region": supplier_region,
            "Single Source": single_source,
            "Lead Time": lead_time,
            "Order Volume": order_volume,
            "Cost": cost,
            "Supply Risk": supply_risk,
            "Profit Impact": profit_impact,
            "Environmental Impact": env_impact
        })

        
        # Downloadable report
       
        report = f"""
KRALJIC CATEGORY PREDICTION REPORT

Supplier Region: {supplier_region}
Single Source Risk: {single_source}
Lead Time: {lead_time}
Order Volume: {order_volume}
Cost per Unit: {cost}
Supply Risk Score: {supply_risk}
Profit Impact Score: {profit_impact}
Environmental Impact: {env_impact}

Predicted Category: {label}
"""

        st.download_button(
            "📥 Download Report",
            report,
            file_name="kraljic_prediction_report.txt"
        )

   
    # Scenario testing (UNIQUE FEATURE)
    
    st.subheader("🔁 What-If Scenario Test")
    st.write("Increase supply risk and profit impact by 1 and observe change.")

    if st.button("Run What-If Scenario"):

        new_supply = min(5, supply_risk + 1)
        new_profit = min(5, profit_impact + 1)

        input_dict["Supply_Risk_Score"] = new_supply
        input_dict["Profit_Impact_Score"] = new_profit

        new_vector = np.array([[input_dict.get(col, 0) for col in feature_names]])
        new_pred = model.predict(new_vector)[0]
        new_label = label_encoder.inverse_transform([new_pred])[0]

        st.info(f"What-if predicted category: {new_label}")

    if st.button("🔄 Reset"):
        st.rerun()


#  BATCH PREDICTION
else:

    st.subheader("📂 Batch Prediction (Upload CSV)")

    uploaded = st.file_uploader("Upload prepared CSV file", type=["csv"])

    st.write("""
The uploaded file must already contain the same processed feature columns
used during model training.
""")

    if uploaded is not None:

        data = pd.read_csv(uploaded)

        missing_cols = set(feature_names) - set(data.columns)

        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
        else:
            X = data[feature_names]

            preds = model.predict(X)
            labels = label_encoder.inverse_transform(preds)

            data["Predicted_Kraljic_Category"] = labels

            st.success("Batch prediction completed.")
            st.dataframe(data.head(), use_container_width=True)

            csv = data.to_csv(index=False).encode("utf-8")

            st.download_button(
                "📥 Download Results CSV",
                csv,
                "kraljic_batch_predictions.csv",
                "text/csv"
            )
import streamlit as st
import pandas as pd
import joblib
import os

# --- 1. LOAD THE MODEL 

@st.cache_resource
def load_my_model():
   
    model_path = 'Olist_Delivery_Status_Predictor.pkl'
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model file '{model_path}' not found! Please check GitHub.")
        return None
    
    try:
        
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"🚨 Unpickling Error: {e}")
        st.info("Ensure scikit-learn==1.7.2 is in your requirements.txt")
        return None

model = load_my_model()

# --- 2. ENCODING MAP ---
payment_map = {
    "Credit Card": 0,
    "Boleto": 1,
    "Voucher": 2,
    "Debit Card": 3
}

# --- 3. PAGE CONFIGURATION ---
st.set_page_config(page_title="Olist Delivery Predictor", layout="wide", page_icon="🚚")

st.title("🚚 Olist Delivery Status Predictor")
st.markdown("### Olist Logistics Analyzer - Predict the probability of delivery")
st.write("Adjust the parameters in the sidebar to analyze the risk of a delivery being late.")

# --- 4. SIDEBAR INPUTS (THE ELITE 8) ---
st.sidebar.header("Logistics Input Parameters")

def user_input_features():
    # Sidebar inputs with unique keys to prevent errors
    pay_inst = st.sidebar.number_input("Payment Installments", 1, 24, 1, key="k1")
    
    pay_type_label = st.sidebar.selectbox("Payment Method", list(payment_map.keys()), key="k2")
    pay_type = float(payment_map[pay_type_label]) 
    
    photo_qty = st.sidebar.slider("Product Photos Qty", 1, 20, 1, key="k3")
    est_dur = st.sidebar.number_input("Estimated Delivery Duration (Days)", 1, 100, 15, key="k4")
    lng = st.sidebar.number_input("Customer Longitude", format="%.6f", value=-46.6333, key="k5")
    pay_seq = st.sidebar.number_input("Payment Sequential Steps", 1, 10, 1, key="k6")
    p_len = st.sidebar.number_input("Product Length (cm)", 1, 300, 30, key="k7")
    lat = st.sidebar.number_input("Customer Latitude", format="%.6f", value=-23.5505, key="k8")


    data = {
        'payment_installments': pay_inst,
        'payment_type': pay_type,
        'product_photos_qty': photo_qty,
        'estimated_delivery_duration': est_dur,
        'geolocation_lng': lng,
        'payment_sequential': pay_seq,
        'product_length_cm': p_len,
        'geolocation_lat': lat
    }
    return pd.DataFrame([data])

# --- 5. PREDICTION LOGIC ---
input_df = user_input_features()

# Display summary table
st.subheader("Current Order Data")
st.dataframe(input_df)

if st.button("Analyze Delivery Risk"):
    if model is not None:
        # Perform prediction
        prediction = model.predict(input_df)
        proba = model.predict_proba(input_df)

        st.markdown("---")
        
        # We index [0][1] to get the probability of the 'LATE' class (1)
        late_probability = proba[0][1] * 100
        on_time_probability = proba[0][0] * 100

        if prediction[0] == 1:
            st.error(f"### 🚨 Prediction: LATE DELIVERY RISK")
            st.metric("Risk Level", f"{late_probability:.2f}%")
            st.write(f"**Interpretation:** The model predicts a **{late_probability:.2f}% probability** that this delivery will be late.")
        else:
            st.success(f"### ✅ Prediction: ON-TIME DELIVERY")
            st.metric("Confidence Level", f"{on_time_probability:.2f}%")
            st.write(f"**Interpretation:** The model predicts a **{on_time_probability:.2f}% probability** that this delivery will be on-time.")
    else:
        st.warning("Model is not loaded. Please check the error message at the top.")

# --- 6. FOOTER / MODEL INFO ---
st.sidebar.markdown("---")
st.sidebar.info(f"**Model Type:** Random Forest\n\n**Accuracy:** 92.14%\n\n**Framework:** Scikit-Learn 1.7.2")
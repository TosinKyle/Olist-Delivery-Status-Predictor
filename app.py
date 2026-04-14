import streamlit as st
import pandas as pd
import pickle

# --- 1. LOAD THE MODEL ---

try:
    with open('Olist_Delivery_Status_Predictor.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("❌ Model file not found! Please check the file name and directory.")

# --- 2. ENCODING MAP ---
# This translates the UI text into the numbers your model was trained on
payment_map = {
    "Credit Card": 0,
    "Boleto": 1,
    "Voucher": 2,
    "Debit Card": 3
}

# --- 3. PAGE CONFIGURATION ---
st.set_page_config(page_title="Olist Delivery Predictor", layout="wide", page_icon="🚚")

st.title("🚚 Olist Delivery Status Predictor")
st.markdown("### Olist Logistics Analyzer - Predict the probability of early or late delivery")
st.write("Fill in the order parameters in the sidebar to predict if a delivery will be **On-Time** or **Late**.")

# --- 4. SIDEBAR INPUTS (THE ELITE 8) ---
st.sidebar.header("Logistics Input Parameters")

def user_input_features():
    # We use unique 'key' arguments to prevent StreamlitDuplicateElementId errors
    
    pay_inst = st.sidebar.number_input("Payment Installments", 1, 24, 1, key="k1")
    
    pay_type_label = st.sidebar.selectbox("Payment Method", list(payment_map.keys()), key="k2")
    pay_type = float(payment_map[pay_type_label]) # Convert to float for sklearn
    
    photo_qty = st.sidebar.slider("Product Photos Qty", 1, 20, 1, key="k3")
    
    est_dur = st.sidebar.number_input("Estimated Delivery Duration (Days)", 1, 100, 15, key="k4")
    
    lng = st.sidebar.number_input("Customer Longitude", format="%.6f", value=-46.6333, key="k5")
    
    pay_seq = st.sidebar.number_input("Payment Sequential Steps", 1, 10, 1, key="k6")
    
    p_len = st.sidebar.number_input("Product Length (cm)", 1, 300, 30, key="k7")
    
    lat = st.sidebar.number_input("Customer Latitude", format="%.6f", value=-23.5505, key="k8")

    # Create Dictionary in the EXACT order your model expects
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

# Display the inputs to the user
st.subheader("Order Data Summary")
st.dataframe(input_df)

if st.button("Analyze Delivery Risk"):
    # Get prediction and probabilities
    prediction = model.predict(input_df)
    proba = model.predict_proba(input_df)

    st.markdown("---")
    
    if prediction[0] == 1:
        st.error("### 🚨 Prediction: LATE DELIVERY RISK")
        st.metric("Risk Confidence", f"{proba[0][1]*100:.2f}%")
        st.write("Action Recommended: Review logistics route or contact seller for early dispatch.")
    else:
        st.success("### ✅ Prediction: ON-TIME DELIVERY")
        st.metric("Confidence", f"{proba[0][0]*100:.2f}%")
        st.write("This order is likely to follow the standard delivery timeline.")

# --- 6. MODEL INFO ---
st.sidebar.markdown("---")
st.sidebar.info(f"**Model:** Random Forest\**Accuracy:** 92.14%\**AUC:** 0.7422")
import streamlit as st
import pandas as pd
import joblib

# Center layout
st.set_page_config(page_title="House Price Prediction", layout="centered")

# Load components
model = joblib.load("Hp_Model.pkl")
scaler = joblib.load("Scaler.pkl")
selector = joblib.load("Features_selected.pkl")

# Extract final features
all_features = selector.feature_names_in_
selected_features = all_features[selector.get_support()]

# Center Content Wrapper
with st.container():
    st.markdown(
        "<h1 style='text-align: center;'>🏠 House Price Classification</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center; font-size:18px;'>"
        "Enter feature values below to predict whether the price is HIGH or LOW"
        "</p>",
        unsafe_allow_html=True,
    )

    # Place inputs in a centered column
    col = st.columns([1,2,1])[1]  # middle column wider
    user_data = {}

    with col:
        for feature in selected_features:
            user_data[feature] = st.number_input(f"{feature}", value=0.0, step=0.1)

        input_df = pd.DataFrame([user_data], columns=selected_features)
        input_scaled = scaler.transform(input_df)

        btn = st.button("🔮 Predict", use_container_width=True)

        if btn:
            prediction = model.predict(input_scaled)[0]
            prob = model.predict_proba(input_scaled)[0][prediction] * 100
            
            if prediction == 1:
                st.success(f"🔥 Predicted: HIGH Price ({prob:.2f}% confidence)")
            else:
                st.info(f"📉 Predicted: LOW Price ({prob:.2f}% confidence)")

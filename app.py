import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Load ML assets
with open("glass_model.pkl", "rb") as file_model:
    model = pickle.load(file_model)

with open("scaler.pkl", "rb") as file_scaler:
    scaler = pickle.load(file_scaler)

# -------------------------
# App Header and Welcome
# -------------------------
st.set_page_config(page_title="Glass Type Prediction", layout="centered")
st.title("🧪 Glass Type Prediction App")
st.markdown("""
Welcome! 👋 This tool helps you identify the **type of glass** based on its chemical ingredients.
Just plug in the values and let the model do the rest! 🔬
""")

# -------------------------
# Glass type dictionary
# -------------------------
glass_types = {
    1: "🏢 Building Windows (Float)",
    2: "🏠 Building Windows (Non-Float)",
    3: "🚗 Vehicle Windows (Float)",
    4: "🚙 Vehicle Windows (Non-Float)",
    5: "🍾 Containers",
    6: "🍽️ Tableware",
    7: "💡 Headlamps"
}

# -------------------------
# Session defaults
# -------------------------
default_inputs = {
    "RI": 0.0, "Na": 0.0, "Mg": 0.0, "Al": 0.0,
    "Si": 0.0, "K": 0.0, "Ca": 0.0, "Ba": 0.0, "Fe": 0.0,
    "reset_form": False
}
for k, v in default_inputs.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -------------------------
# Reset form if triggered
# -------------------------
if st.session_state.reset_form:
    for k in default_inputs:
        if k != "reset_form":
            st.session_state[k] = 0.0
    st.session_state.reset_form = False

# -------------------------
# Sample button
# -------------------------
if st.button("🎓 Try with Sample Glass Data"):
    sample_data = {
        "RI": 1.516, "Na": 12.85, "Mg": 3.55, "Al": 1.27,
        "Si": 73.15, "K": 0.45, "Ca": 8.10, "Ba": 0.0, "Fe": 0.05
    }
    for k, v in sample_data.items():
        st.session_state[k] = v

# -------------------------
# Input Form
# -------------------------
with st.form("input_form"):
    st.subheader("🔍 Input Glass Composition")
    RI = st.number_input("Refractive Index (RI)", format="%.6f", key="RI")
    Na = st.number_input("Sodium (Na)", format="%.2f", key="Na")
    Mg = st.number_input("Magnesium (Mg)", format="%.2f", key="Mg")
    Al = st.number_input("Aluminum (Al)", format="%.2f", key="Al")
    Si = st.number_input("Silicon (Si)", format="%.2f", key="Si")
    K  = st.number_input("Potassium (K)", format="%.2f", key="K")
    Ca = st.number_input("Calcium (Ca)", format="%.2f", key="Ca")
    Ba = st.number_input("Barium (Ba)", format="%.2f", key="Ba")
    Fe = st.number_input("Iron (Fe)", format="%.2f", key="Fe")

    left, right = st.columns(2)
    with left:
        submit = st.form_submit_button("🔮 Analyze Glass")
    with right:
        reset = st.form_submit_button("🧹 Clear Form")

# -------------------------
# Handle Reset Button
# -------------------------
if reset:
    st.session_state.reset_form = True
    st.rerun()

# -------------------------
# Handle Prediction
# -------------------------
if submit:
    user_data = pd.DataFrame([[
        st.session_state.RI, st.session_state.Na, st.session_state.Mg,
        st.session_state.Al, st.session_state.Si, st.session_state.K,
        st.session_state.Ca, st.session_state.Ba, st.session_state.Fe
    ]], columns=["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"])

    scaled_input = scaler.transform(user_data)
    predicted_class = model.predict(scaled_input)[0]
    predicted_prob = model.predict_proba(scaled_input)[0]

    predicted_label = glass_types.get(predicted_class, "Unknown Type")
    st.success(f"🔍 **Prediction Result:** {predicted_label} (Type {predicted_class})")

    # Show probabilities
    st.markdown("### 📊 Model Confidence:")
    for class_id, prob in zip(model.classes_, predicted_prob):
        label = glass_types.get(class_id, "Unknown")
        st.write(f"**{label}**: {prob:.2%}")

    st.caption("Results based on your input chemical composition.")

# -------------------------
# Feature Importance
# -------------------------
with st.expander("📌 How does the model decide?"):
    st.write("Feature importance shows which elements matter most in the prediction.")
    importance_scores = model.feature_importances_
    features = ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]

    fig, ax = plt.subplots()
    ax.barh(features, importance_scores, color='mediumslateblue')
    ax.set_title("🔬 Feature Importance (Random Forest)")
    ax.set_xlabel("Score")
    st.pyplot(fig)

# -------------------------
# Model Info
# -------------------------
with st.expander("📚 About This App"):
    st.write("""
    - 🔍 **Model**: Random Forest Classifier  
    - 📈 **Accuracy**: ~89% on UCI Glass Dataset  
    - 🔬 **Trained on**: Chemical attributes of various glass types  
    - 🧠 **Goal**: Educational use for material classification
    """)

# Sidebar credits and GitHub link
with st.sidebar:
    st.header("Resources")
    st.markdown("[GitHub Repo](https://github.com/OrleneJane/BAM_3024_Assignment_07_glass-type-prediction-app_Bilog-Orlene-Jane)")
    st.markdown("Created by Orlene Jane Bilog")
    st.markdown("Powered by **Streamlit** + **scikit-learn**")
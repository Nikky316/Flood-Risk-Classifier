import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# =========================
# PAGE CONFIGURATION
# =========================

st.set_page_config(
    page_title="Flood Risk Classifier",
    page_icon="🌊",
    layout="wide"
)

# =========================
# LOAD TRAINED MODEL
# =========================

model = joblib.load("models/flood_model.pkl")

# =========================
# SIDEBAR
# =========================

st.sidebar.title("📊 Model Information")
st.sidebar.markdown("""
### 🤖 Machine Learning Model

**Random Forest Classifier**
""")
st.sidebar.info("Dataset Size: 50,000 Records")
st.sidebar.write("20 Environmental & Infrastructure Factors")
st.sidebar.markdown("---")
st.sidebar.caption(
    "This application is intended for educational and demonstration purposes. Predictions are based on the trained machine learning model and should not replace official flood warnings or environmental assessments."
)

# =========================
# TITLE
# =========================

st.title("🌊 Flood Risk Classifier")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Records", "50,000")

with col2:
    st.metric("Features", "20")

with col3:
    st.metric("Model", "Random Forest")


st.markdown("""
### Intelligent Flood Risk Assessment System

This machine learning application evaluates environmental and infrastructure factors
to classify flood risk as Low, Medium, or High.
""")

st.markdown("""
This application predicts flood risk using environmental and infrastructure-related factors.

**Risk Categories**
- 🟢 Low Risk
- 🟡 Medium Risk
- 🔴 High Risk
""")

# =========================
# FEATURE IMPORTANCE
# =========================

st.subheader("📈 Feature Importance")

importance_df = pd.DataFrame({
    "Feature": model.feature_names_in_,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=True
)

fig, ax = plt.subplots(figsize=(10,7))

ax.barh(
    importance_df["Feature"],
    importance_df["Importance"],
    color="steelblue"
)

ax.set_xlabel("Importance Score")
ax.set_title("Most Important Features")

st.pyplot(fig)

st.divider()

# =========================
# INPUT SECTION
# =========================

st.subheader("📝 Enter Flood Indicators")

st.caption(
    "Adjust the environmental and infrastructure factors below, then click Predict Flood Risk."
)

col1, col2 = st.columns(2)

with col1:
    MonsoonIntensity = st.slider("🌧 Monsoon Intensity", 0, 10, 5)
    TopographyDrainage = st.slider("🏞 Topography Drainage", 0, 10, 5)
    RiverManagement = st.slider("🌊 River Management", 0, 10, 5)
    Deforestation = st.slider("🌳 Deforestation", 0, 10, 5)
    Urbanization = st.slider("🏙 Urbanization", 0, 10, 5)
    ClimateChange = st.slider("🌡 Climate Change", 0, 10, 5)
    DamsQuality = st.slider("🏗 Dams Quality", 0, 10, 5)
    Siltation = st.slider("🪨 Siltation", 0, 10, 5)
    AgriculturalPractices = st.slider("🚜 Agricultural Practices", 0, 10, 5)
    Encroachments = st.slider("🏠 Encroachments", 0, 10, 5)

with col2:
    IneffectiveDisasterPreparedness = st.slider(
        "🚨 Disaster Preparedness", 0, 10, 5
    )

    DrainageSystems = st.slider(
        "🚰 Drainage Systems", 0, 10, 5
    )

    CoastalVulnerability = st.slider(
        "🌊 Coastal Vulnerability", 0, 10, 5
    )

    Landslides = st.slider(
        "⛰ Landslides", 0, 10, 5
    )

    Watersheds = st.slider(
        "💧 Watersheds", 0, 10, 5
    )

    DeterioratingInfrastructure = st.slider(
        "🏗 Infrastructure Quality", 0, 10, 5
    )

    PopulationScore = st.slider(
        "👥 Population Score", 0, 10, 5
    )

    WetlandLoss = st.slider(
        "🌿 Wetland Loss", 0, 10, 5
    )

    InadequatePlanning = st.slider(
        "📋 Inadequate Planning", 0, 10, 5
    )

    PoliticalFactors = st.slider(
        "🏛 Political Factors", 0, 10, 5
    )

# =========================
# PREDICTION BUTTON
# =========================

if st.button("🔍 Predict Flood Risk"):

    from datetime import datetime

    input_data = pd.DataFrame([[
        MonsoonIntensity,
        TopographyDrainage,
        RiverManagement,
        Deforestation,
        Urbanization,
        ClimateChange,
        DamsQuality,
        Siltation,
        AgriculturalPractices,
        Encroachments,
        IneffectiveDisasterPreparedness,
        DrainageSystems,
        CoastalVulnerability,
        Landslides,
        Watersheds,
        DeterioratingInfrastructure,
        PopulationScore,
        WetlandLoss,
        InadequatePlanning,
        PoliticalFactors
    ]], columns=model.feature_names_in_)

    st.subheader("Input Summary")
    st.dataframe(input_data)

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0]
    confidence = max(probability) * 100

    st.divider()
    st.subheader("📌 Prediction Result")

    st.metric(
        "Prediction Confidence",
        f"{confidence:.2f}%"
    ) 

    risk_levels = ["Low", "Medium", "High"]

    prob_df = pd.DataFrame({
        "Risk": risk_levels,
        "Probability": probability
    })

    st.bar_chart(
        prob_df.set_index("Risk")
)

    if prediction == 0:
        st.success("🟢 LOW FLOOD RISK")
        st.progress(25)
        st.info("""
        ### Environmental Assessment

        Current environmental conditions indicate a **low likelihood of flooding**. Existing drainage capacity and environmental resilience appear sufficient under the provided conditions.

        ### Recommended Actions

        - Continue routine inspection and maintenance of drainage infrastructure.
        - Preserve wetlands and vegetation that naturally absorb excess rainfall.
        - Monitor seasonal weather forecasts for unusual rainfall patterns.
        - Encourage responsible land use to minimize future flood vulnerability.
        """)

    elif prediction == 1:
        st.warning("🟡 MEDIUM FLOOD RISK")
        st.progress(60)
        st.info("""
        ### Environmental Assessment

        The selected conditions suggest **moderate flood susceptibility**. Increased rainfall, land-use changes, or reduced drainage efficiency could elevate flood risk.
        
        ### Recommended Actions

        - Clear blocked drains, culverts, and waterways before heavy rainfall.
        - Strengthen community flood preparedness and emergency response plans.
        - Improve stormwater drainage and water retention systems.
        - Reduce deforestation and promote sustainable urban planning.
        - Closely monitor weather forecasts during the rainy season.
        """)

    else:
        st.error("🔴 HIGH FLOOD RISK")
        st.progress(95)
        st.info("""
        ### Environmental Assessment

        The analysis indicates **high flood risk**. Environmental and infrastructure conditions suggest a significant likelihood of flooding if heavy rainfall occurs.

        ### Recommended Actions

        - Activate local emergency preparedness and evacuation procedures.
        - Relocate residents and valuable assets from flood-prone areas where necessary.
        - Continuously monitor river levels and official weather advisories.
        - Deploy temporary flood barriers and inspect critical infrastructure.
        - Prioritize restoration of drainage systems and reinforce embankments.
        - Implement long-term mitigation measures such as reforestation, wetland conservation, and improved urban drainage planning.
    """)
        
    st.caption(
    f"Prediction generated on {datetime.now().strftime('%d %B %Y %H:%M')}"
)


# =========================
# ABOUT PROJECT
# =========================

st.divider()

with st.expander("ℹ️ About This Project"):
    st.markdown("""
### Flood Risk Classifier

The Flood Risk Classifier uses machine learning to predict flood risk based on environmental and infrastructure indicators.

#### Technology Stack
- Python
- Pandas
- Scikit-Learn
- Streamlit
- Matplotlib

#### Machine Learning Model
- Random Forest Classifier

#### Dataset
- 50,000 records
- 20 environmental and infrastructure features

#### Purpose
To support flood risk assessment by providing predictive insights that help communities and decision-makers prepare for potential flood events.
""")

# True Footer

st.markdown("---")

st.markdown(
    """
    <div style='text-align:center; color:gray; padding:10px;'>
        © 2026 | Nike Nsikak-Nelson
    </div>
    """,
    unsafe_allow_html=True
)
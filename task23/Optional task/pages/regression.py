import streamlit as st

@st.cache_resource
def get_regression_data_and_models():
    from train_models import prepare_diabetes_data, train_and_eval_regression
    data = prepare_diabetes_data()
    results = train_and_eval_regression(data)
    return data, results

data, results = get_regression_data_and_models()
reg_model_info = results['Linear Regression']
model = reg_model_info['model']

st.header("📈 Regression Module - Diabetes Disease Progression")
st.caption("Dataset: Diabetes Dataset (Scikit-Learn)")

col_input, col_viz = st.columns([1, 1])

with col_input:
    st.subheader("⚙️ Feature Inputs")
    input_values = []
    
    df_features = data['df'].drop(columns=['target'])
    means = df_features.mean()
    
    for feat in data['feature_names']:
        val = st.slider(
            f"Feature: {feat}",
            min_value=float(df_features[feat].min()),
            max_value=float(df_features[feat].max()),
            value=float(means[feat]),
            format="%.4f"
        )
        input_values.append(val)

    predict_btn = st.button("Predict Disease Progression")

with col_viz:
    st.subheader("🎯 Regression Predictions & Evaluation")
    
    if predict_btn:
        scaled_input = data['scaler'].transform([input_values])
        prediction = model.predict(scaled_input)[0]
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>Predicted Progression Level</h3>
            <h1 style="color: #4F46E5;">{prediction:.2f}</h1>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Model Summary")
    
    r1, r2, r3 = st.columns(3)
    r1.metric("R² Score", f"{reg_model_info['r2_score']:.4f}")
    r2.metric("Mean Squared Error", f"{reg_model_info['mse']:.2f}")
    r3.metric("Root MSE", f"{reg_model_info['rmse']:.2f}")
    
    # Plot Actual vs Predicted
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(data['y_test'], reg_model_info['y_pred'], alpha=0.6, color='#4F46E5')
    ax.plot([data['y_test'].min(), data['y_test'].max()], [data['y_test'].min(), data['y_test'].max()], 'r--', lw=2)
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title('Actual vs Predicted Progression')
    st.pyplot(fig)
import streamlit as st

@st.cache_resource
def get_classification_data_and_models():
    from train_models import prepare_breast_cancer_data, train_and_eval_classification
    data = prepare_breast_cancer_data()
    results = train_and_eval_classification(data)
    return data, results

data, results = get_classification_data_and_models()

st.header("🧪 Classification Module - Breast Cancer Diagnosis")
st.caption("Dataset: Breast Cancer Wisconsin (Diagnostic)")

selected_model_name = st.selectbox("Select Model Architecture", list(results.keys()))
model_data = results[selected_model_name]
model = model_data['model']

st.markdown("---")

col_inputs, col_outputs = st.columns([1, 1])

with col_inputs:
    st.subheader("📋 Input Sample Metrics")
    input_vals = []
    
    # Showcase top key features for easy user input while taking defaults from dataset mean
    df_features = data['df'].drop(columns=['target'])
    means = df_features.mean()
    stops = df_features.std()
    
    # Group inputs cleanly using expanders
    with st.expander("Mean Attributes", expanded=True):
        for feat in data['feature_names'][:10]:
            val = st.number_input(
                f"{feat}",
                value=float(means[feat]),
                format="%.4f"
            )
            input_vals.append(val)
            
    with st.expander("Error Attributes"):
        for feat in data['feature_names'][10:20]:
            val = st.number_input(
                f"{feat}",
                value=float(means[feat]),
                format="%.4f"
            )
            input_vals.append(val)

    with st.expander("Worst Attributes"):
        for feat in data['feature_names'][20:]:
            val = st.number_input(
                f"{feat}",
                value=float(means[feat]),
                format="%.4f"
            )
            input_vals.append(val)

    predict_btn = st.button("Run Classification Prediction")

with col_outputs:
    st.subheader("🎯 Prediction & Evaluation Results")
    
    if predict_btn:
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        
        # Scale user inputs
        scaled_input = data['scaler'].transform([input_vals])
        prediction = model.predict(scaled_input)[0]
        probabilities = model.predict_proba(scaled_input)[0]
        
        target_class = data['target_names'][prediction]
        confidence = probabilities[prediction] * 100
        
        if prediction == 1:
            st.success(f"**Diagnosis:** {target_class.upper()} ({confidence:.2f}% Confidence)")
        else:
            st.error(f"**Diagnosis:** {target_class.upper()} ({confidence:.2f}% Confidence)")
            
        st.markdown("#### Prediction Probabilities")
        prob_df = pd.DataFrame({
            'Class': data['target_names'],
            'Probability': probabilities
        })
        st.bar_chart(prob_df.set_index('Class'))
    
    st.markdown("---")
    st.markdown(f"### Model Performance: `{selected_model_name}`")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{model_data['accuracy']*100:.2f}%")
    m2.metric("Precision", f"{model_data['precision']*100:.2f}%")
    m3.metric("Recall", f"{model_data['recall']*100:.2f}%")
    m4.metric("F1-Score", f"{model_data['f1_score']*100:.2f}%")
    
    # Display Confusion Matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.heatmap(model_data['confusion_matrix'], annot=True, fmt='d', cmap='Blues', 
                xticklabels=data['target_names'], yticklabels=data['target_names'], ax=ax)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    st.pyplot(fig)
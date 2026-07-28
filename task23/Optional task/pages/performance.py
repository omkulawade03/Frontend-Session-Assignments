import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

@st.cache_resource
def load_all_metrics():
    from train_models import (
        prepare_breast_cancer_data, train_and_eval_classification,
        prepare_diabetes_data, train_and_eval_regression
    )
    
    c_data = prepare_breast_cancer_data()
    c_results = train_and_eval_classification(c_data)
    
    r_data = prepare_diabetes_data()
    r_results = train_and_eval_regression(r_data)
    
    return c_results, r_results

c_results, r_results = load_all_metrics()

st.header("📊 Model Performance Comparison")

# --- CLASSIFICATION SECTION ---
st.subheader("🧪 Classification Metrics Comparison")
clf_rows = []
for name, metrics in c_results.items():
    clf_rows.append({
        'Model': name,
        'Accuracy': f"{metrics['accuracy']*100:.2f}%",
        'Precision': f"{metrics['precision']*100:.2f}%",
        'Recall': f"{metrics['recall']*100:.2f}%",
        'F1-Score': f"{metrics['f1_score']*100:.2f}%"
    })

clf_df = pd.DataFrame(clf_rows)

col_clf_table, col_clf_dl = st.columns([3, 1])
with col_clf_table:
    st.dataframe(clf_df, use_container_width=True)

with col_clf_dl:
    clf_csv = clf_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Classification Metrics (CSV)",
        data=clf_csv,
        file_name="classification_metrics.csv",
        mime="text/csv",
        key="download-clf-csv"
    )

best_clf = max(c_results.items(), key=lambda x: x[1]['accuracy'])
st.success(f"🏆 **Top Performing Classification Model:** {best_clf[0]} (Accuracy: {best_clf[1]['accuracy']*100:.2f}%)")

st.markdown("---")

# --- REGRESSION SECTION ---
st.subheader("📈 Regression Metrics")
reg_rows = []
for name, metrics in r_results.items():
    reg_rows.append({
        'Model': name,
        'R² Score': f"{metrics['r2_score']:.4f}",
        'MSE': f"{metrics['mse']:.2f}",
        'RMSE': f"{metrics['rmse']:.2f}"
    })

reg_df = pd.DataFrame(reg_rows)

col_reg_table, col_reg_dl = st.columns([3, 1])
with col_reg_table:
    st.dataframe(reg_df, use_container_width=True)

with col_reg_dl:
    reg_csv = reg_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Regression Metrics (CSV)",
        data=reg_csv,
        file_name="regression_metrics.csv",
        mime="text/csv",
        key="download-reg-csv"
    )

# --- VISUALIZATION SECTION ---
st.markdown("### Accuracy Comparison Plot")
fig, ax = plt.subplots(figsize=(8, 4))
models = list(c_results.keys())
accuracies = [c_results[m]['accuracy'] * 100 for m in models]

ax.bar(models, accuracies, color=['#4F46E5', '#3B82F6', '#10B981'])
ax.set_ylabel('Accuracy (%)')
ax.set_ylim(80, 100)
for i, v in enumerate(accuracies):
    ax.text(i, v + 0.5, f"{v:.2f}%", ha='center', fontweight='bold')
st.pyplot(fig)
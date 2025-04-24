import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import xgboost

def plot_feature_importance():
    """Plot feature importance for tree-based models"""
    # Load models
    rf = joblib.load('models/random_forest.pkl')
    xgb = joblib.load('models/xgboost_model.pkl')
    
    # Random Forest importance
    plt.figure(figsize=(12,6))
    pd.Series(rf.feature_importances_, 
             index=rf.feature_names_in_).nlargest(10).plot(kind='barh')
    plt.title('Random Forest Feature Importance')
    plt.savefig('visualizations/rf_feature_importance.png')
    plt.close()
    
    # XGBoost importance
    plt.figure(figsize=(12,6))
    xgb.plot_importance(xgb)
    plt.title('XGBoost Feature Importance')
    plt.tight_layout()
    plt.savefig('visualizations/xgb_feature_importance.png')
    plt.close()

def plot_shap_values():
    """Generate SHAP explanations"""
    # Load model and data
    model = joblib.load('models/xgboost_model.pkl')
    X, _ = prepare_xgboost_data()
    X_train, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # SHAP summary plot
    explainer = shap.Explainer(model)
    shap_values = explainer(X_train)
    
    plt.figure()
    shap.summary_plot(shap_values, X_train, plot_type="bar")
    plt.savefig('visualizations/shap_summary.png')
    plt.close()

def plot_model_comparison():
    """Compare model performance visually"""
    metrics = pd.read_csv('data/model_metrics.csv')
    
    plt.figure(figsize=(12,6))
    sns.barplot(data=metrics, x='Model', y='RMSE')
    plt.title('Model Comparison by RMSE')
    plt.ylabel('Root Mean Squared Error ($)')
    plt.savefig('visualizations/model_comparison.png')
    plt.close()

if __name__ == "__main__":
    from xgboost_model import prepare_xgboost_data
    from sklearn.model_selection import train_test_split
    import pandas as pd
    
    plot_feature_importance()
    plot_shap_values()
    plot_model_comparison()
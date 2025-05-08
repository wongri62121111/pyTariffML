import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score
import shap

class ModelEvaluator:
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = model.predict(X_test)
        
    def calculate_metrics(self):
        """Compute comprehensive evaluation metrics"""
        return {
            'MAE': mean_absolute_error(self.y_test, self.y_pred),
            'R2': r2_score(self.y_test, self.y_pred),
            'Error Distribution': self.y_pred - self.y_test
        }
    
    def plot_residuals(self):
        """Visualize prediction errors"""
        plt.figure(figsize=(10, 6))
        sns.residplot(
            x=self.y_pred, 
            y=self.y_test,
            lowess=True,
            line_kws={'color': 'red'}
        )
        plt.title('Residual Analysis')
        plt.savefig('visualizations/residual_analysis.png')
        
    def explain_model(self):
        """SHAP value analysis for model interpretability"""
        explainer = shap.Explainer(self.model.named_steps['regressor'])
        shap_values = explainer.shap_values(self.X_test)
        
        plt.figure()
        shap.summary_plot(shap_values, self.X_test)
        plt.savefig('visualizations/shap_summary.png')
        
    def generate_report(self):
        """Create comprehensive evaluation report"""
        metrics = self.calculate_metrics()
        
        report = f"""
        MODEL EVALUATION REPORT
        -------------------------
        Mean Absolute Error: ${metrics['MAE']:,.2f}
        R-squared: {metrics['R2']:.3f}
        
        Key Insights:
        1. Model tends to under-predict tariffs for countries with {self._identify_error_patterns()}
        2. Most important features: {self._get_top_features()}
        """
        
        with open('reports/model_evaluation.txt', 'w') as f:
            f.write(report)
        
        return report
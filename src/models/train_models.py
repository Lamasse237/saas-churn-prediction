"""
Train and evaluate multiple churn prediction models
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                            roc_auc_score, roc_curve, precision_recall_curve,
                            average_precision_score)
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class ChurnModelTrainer:
    def __init__(self, target_horizon='30d'):
        """
        Initialize model trainer
        target_horizon: '30d', '60d', or '90d'
        """
        self.target_horizon = target_horizon
        self.target_col = f'churned_{target_horizon}'
        self.models = {}
        self.feature_cols = None
        self.results = {}
        
    def prepare_data(self, df):
        """Prepare features and target for modeling"""
        # Drop non-feature columns
        drop_cols = [
            'customer_id', 'signup_date', 'churned_30d', 'churned_60d', 
            'churned_90d', 'churn_risk_score', 'customer_lifetime_value',
            # Keep only encoded versions of categoricals
            'company_size', 'industry', 'plan_type', 'payment_method',
            'contract_length', 'revenue_tier', 'tenure_bucket'
        ]
        
        drop_cols = [col for col in drop_cols if col in df.columns]
        
        # Prepare features
        X = df.drop(columns=drop_cols)
        y = df[self.target_col]
        
        # Store feature columns
        self.feature_cols = X.columns.tolist()
        
        print(f"\nTarget: {self.target_col}")
        print(f"Features: {len(self.feature_cols)}")
        print(f"Churn rate: {y.mean():.2%}")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2):
        """Split data with stratification"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"\nTrain size: {len(X_train)}")
        print(f"Test size: {len(X_test)}")
        print(f"Train churn rate: {y_train.mean():.2%}")
        print(f"Test churn rate: {y_test.mean():.2%}")
        
        return X_train, X_test, y_train, y_test
    
    def handle_imbalance(self, X_train, y_train, method='smote'):
        """Handle class imbalance"""
        if method == 'smote':
            print("\nApplying SMOTE...")
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            
            print(f"Original class distribution: {y_train.value_counts().to_dict()}")
            print(f"Resampled class distribution: {y_resampled.value_counts().to_dict()}")
            
            return X_resampled, y_resampled
        
        return X_train, y_train
    
    def train_logistic_regression(self, X_train, y_train):
        """Train Logistic Regression (interpretable baseline)"""
        print("\n" + "="*50)
        print("Training Logistic Regression...")
        print("="*50)
        
        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )
        
        model.fit(X_train, y_train)
        self.models['logistic_regression'] = model
        
        # Feature importance
        importance_df = pd.DataFrame({
            'feature': self.feature_cols,
            'coefficient': model.coef_[0]
        }).sort_values('coefficient', key=abs, ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(importance_df.head(10))
        
        return model
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest"""
        print("\n" + "="*50)
        print("Training Random Forest...")
        print("="*50)
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=50,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        self.models['random_forest'] = model
        
        # Feature importance
        importance_df = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(importance_df.head(10))
        
        return model
    
    def train_xgboost(self, X_train, y_train):
        """Train XGBoost"""
        print("\n" + "="*50)
        print("Training XGBoost...")
        print("="*50)
        
        # Calculate scale_pos_weight for imbalance
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
        
        model.fit(X_train, y_train)
        self.models['xgboost'] = model
        
        # Feature importance
        importance_df = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(importance_df.head(10))
        
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Comprehensive model evaluation"""
        print(f"\n{'='*50}")
        print(f"Evaluating {model_name}")
        print(f"{'='*50}")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        # ROC AUC
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"\nROC AUC Score: {roc_auc:.4f}")
        
        # Average Precision
        avg_precision = average_precision_score(y_test, y_pred_proba)
        print(f"Average Precision Score: {avg_precision:.4f}")
        
        # Store results
        self.results[model_name] = {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'roc_auc': roc_auc,
            'avg_precision': avg_precision,
            'confusion_matrix': cm
        }
        
        return self.results[model_name]
    
    def compare_models(self):
        """Compare all models"""
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        comparison_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'ROC AUC': [r['roc_auc'] for r in self.results.values()],
            'Avg Precision': [r['avg_precision'] for r in self.results.values()]
        }).sort_values('ROC AUC', ascending=False)
        
        print("\n", comparison_df)
        
        return comparison_df
    
    def plot_roc_curves(self, y_test):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))
        
        for model_name, results in self.results.items():
            fpr, tpr, _ = roc_curve(y_test, results['probabilities'])
            plt.plot(fpr, tpr, label=f"{model_name} (AUC = {results['roc_auc']:.3f})")
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves - {self.target_horizon} Churn Prediction')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'reports/roc_curves_{self.target_horizon}.png', dpi=300)
        plt.close()
        
        print(f"\nROC curves saved to reports/roc_curves_{self.target_horizon}.png")
    
    def save_models(self):
        """Save all trained models"""
        for model_name, model in self.models.items():
            filename = f'models/{model_name}_{self.target_horizon}.pkl'
            joblib.dump(model, filename)
            print(f"Saved {model_name} to {filename}")
    
    def train_all_models(self, df, use_smote=True):
        """Complete training pipeline"""
        # Prepare data
        X, y = self.prepare_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # Handle imbalance
        if use_smote:
            X_train_balanced, y_train_balanced = self.handle_imbalance(X_train, y_train)
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
        
        # Train models
        self.train_logistic_regression(X_train_balanced, y_train_balanced)
        self.train_random_forest(X_train_balanced, y_train_balanced)
        self.train_xgboost(X_train_balanced, y_train_balanced)
        
        # Evaluate all models
        for model_name, model in self.models.items():
            self.evaluate_model(model, X_test, y_test, model_name)
        
        # Compare models
        comparison = self.compare_models()
        
        # Plot ROC curves
        self.plot_roc_curves(y_test)
        
        # Save models
        self.save_models()
        
        return comparison

def main():
    """Train models for all time horizons"""
    # Load processed data
    print("Loading processed data...")
    df = pd.read_csv('data/processed/features_engineered.csv')
    
    results_all = {}
    
    # Train for each time horizon
    for horizon in ['30d', '60d', '90d']:
        print("\n" + "="*70)
        print(f"TRAINING MODELS FOR {horizon} CHURN PREDICTION")
        print("="*70)
        
        trainer = ChurnModelTrainer(target_horizon=horizon)
        comparison = trainer.train_all_models(df, use_smote=True)
        results_all[horizon] = comparison
    
    # Summary across all horizons
    print("\n" + "="*70)
    print("SUMMARY: BEST MODELS BY HORIZON")
    print("="*70)
    
    for horizon, comparison in results_all.items():
        best_model = comparison.iloc[0]
        print(f"\n{horizon}: {best_model['Model']} (ROC AUC: {best_model['ROC AUC']:.4f})")

if __name__ == '__main__':
    # Create reports directory if it doesn't exist
    import os
    os.makedirs('reports', exist_ok=True)
    
    main()
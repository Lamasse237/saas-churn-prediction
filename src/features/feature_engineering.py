"""
Advanced feature engineering for churn prediction
"""
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class ChurnFeatureEngineer:
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        
    def create_engagement_features(self, df):
        """Create engagement-based features"""
        df = df.copy()
        
        # Engagement score (0-100)
        df['engagement_score'] = (
            (df['avg_daily_logins'] / df['avg_daily_logins'].max() * 30) +
            (df['feature_adoption_rate'] * 30) +
            (df['active_users_pct'] * 20) +
            (df['avg_session_duration'] / df['avg_session_duration'].max() * 20)
        )
        
        # Usage intensity
        df['usage_intensity'] = df['avg_daily_logins'] * df['avg_session_duration']
        
        # Feature utilization
        df['feature_utilization'] = df['features_used'] * df['feature_adoption_rate']
        
        # Seat utilization
        df['seat_utilization'] = df['active_users_pct']
        
        # Engagement trend (velocity)
        df['engagement_velocity'] = (df['usage_trend_30d'] + df['usage_trend_60d']) / 2
        
        # Critical engagement flags
        df['low_engagement'] = ((df['avg_daily_logins'] < 1) | 
                                 (df['feature_adoption_rate'] < 0.3)).astype(int)
        
        df['declining_engagement'] = (df['usage_trend_30d'] < -0.15).astype(int)
        
        return df
    
    def create_support_features(self, df):
        """Create support-related features"""
        df = df.copy()
        
        # Support intensity
        df['support_intensity'] = (
            df['support_tickets_30d'] + 
            df['critical_issues_30d'] * 3
        )
        
        # Support health score (inverse - lower is better)
        df['support_health'] = 100 - (
            (df['support_tickets_30d'] / df['support_tickets_30d'].max() * 30) +
            (df['critical_issues_30d'] / (df['critical_issues_30d'].max() + 1) * 40) +
            ((5 - df['satisfaction_score']) / 5 * 30)
        )
        
        # High-risk support flags
        df['high_support_load'] = (df['support_tickets_30d'] > 3).astype(int)
        df['has_critical_issues'] = (df['critical_issues_30d'] > 0).astype(int)
        df['low_satisfaction'] = (df['satisfaction_score'] < 3.5).astype(int)
        
        # Resolution efficiency
        df['slow_resolution'] = (df['avg_resolution_time'] > 48).astype(int)
        
        return df
    
    def create_billing_features(self, df):
        """Create billing and revenue features"""
        df = df.copy()
        
        # Revenue tier
        df['revenue_tier'] = pd.cut(
            df['monthly_revenue'],
            bins=[0, 100, 300, 1000, np.inf],
            labels=['low', 'medium', 'high', 'enterprise']
        )
        
        # Payment health
        df['payment_health'] = 100 - (df['payment_failures_90d'] * 25)
        df['payment_health'] = df['payment_health'].clip(0, 100)
        
        # Contract risk
        df['short_contract'] = (df['contract_length'] == 'monthly').astype(int)
        df['renewal_approaching'] = (df['days_to_renewal'] < 30).astype(int)
        
        # Revenue at risk (if they churn)
        # Assume 12-month LTV for simplicity
        df['revenue_at_risk'] = df['monthly_revenue'] * 12
        
        # Discount dependency
        df['heavy_discount'] = (df['discount_pct'] > 0.2).astype(int)
        
        return df
    
    def create_customer_profile_features(self, df):
        """Create customer profile features"""
        df = df.copy()
        
        # Tenure features
        reference_date = pd.to_datetime('2024-12-31')
        df['tenure_days'] = (reference_date - pd.to_datetime(df['signup_date'])).dt.days
        df['tenure_months'] = df['tenure_days'] / 30
        
        # Tenure buckets
        df['tenure_bucket'] = pd.cut(
            df['tenure_months'],
            bins=[0, 3, 6, 12, 24, np.inf],
            labels=['0-3mo', '3-6mo', '6-12mo', '12-24mo', '24mo+']
        )
        
        # New customer flag
        df['is_new_customer'] = (df['tenure_months'] < 6).astype(int)
        
        # Company size score
        size_scores = {'small': 1, 'medium': 2, 'large': 3}
        df['company_size_score'] = df['company_size'].map(size_scores)
        
        # Plan value
        plan_scores = {'Basic': 1, 'Professional': 2, 'Enterprise': 3}
        df['plan_value'] = df['plan_type'].map(plan_scores)
        
        return df
    
    def create_interaction_features(self, df):
        """Create interaction features between different domains"""
        df = df.copy()
        
        # Value × Engagement
        df['value_engagement'] = df['monthly_revenue'] * df['engagement_score']
        
        # Support × Tenure (new customers with high support is bad)
        df['support_tenure_risk'] = df['support_intensity'] / (df['tenure_months'] + 1)
        
        # Engagement × Seats (low engagement despite many seats)
        df['seat_engagement_gap'] = df['num_seats'] * (1 - df['active_users_pct'])
        
        # Revenue × Payment issues
        df['payment_revenue_risk'] = df['monthly_revenue'] * df['payment_failures_90d']
        
        return df
    
    def create_risk_flags(self, df):
        """Create composite risk flags"""
        df = df.copy()
        
        # High-risk combination flags
        df['critical_risk'] = (
            (df['declining_engagement'] == 1) & 
            (df['has_critical_issues'] == 1) &
            (df['payment_failures_90d'] > 0)
        ).astype(int)
        
        df['engagement_risk'] = (
            (df['low_engagement'] == 1) &
            (df['usage_trend_30d'] < 0)
        ).astype(int)
        
        df['support_risk'] = (
            (df['high_support_load'] == 1) &
            (df['low_satisfaction'] == 1)
        ).astype(int)
        
        df['payment_risk'] = (
            (df['payment_failures_90d'] > 0) &
            (df['short_contract'] == 1)
        ).astype(int)
        
        # Total risk flags
        df['total_risk_flags'] = (
            df['critical_risk'] + df['engagement_risk'] + 
            df['support_risk'] + df['payment_risk']
        )
        
        return df
    
    def encode_categorical(self, df, fit=True):
        """Encode categorical variables"""
        df = df.copy()
        
        categorical_cols = ['company_size', 'industry', 'plan_type', 
                           'payment_method', 'contract_length', 
                           'revenue_tier', 'tenure_bucket']
        
        for col in categorical_cols:
            if col in df.columns:
                if fit:
                    self.encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.encoders[col].fit_transform(df[col])
                else:
                    df[f'{col}_encoded'] = self.encoders[col].transform(df[col])
        
        return df
    
    def scale_features(self, df, fit=True):
        """Scale numerical features"""
        df = df.copy()
        
        # Features to scale
        scale_cols = [
            'avg_daily_logins', 'features_used', 'avg_session_duration',
            'num_seats', 'monthly_revenue', 'engagement_score',
            'support_intensity', 'usage_intensity', 'tenure_months'
        ]
        
        scale_cols = [col for col in scale_cols if col in df.columns]
        
        if fit:
            self.scalers['standard'] = StandardScaler()
            df[scale_cols] = self.scalers['standard'].fit_transform(df[scale_cols])
        else:
            df[scale_cols] = self.scalers['standard'].transform(df[scale_cols])
        
        return df
    
    def engineer_features(self, df, fit=True):
        """Apply all feature engineering steps"""
        print("Creating engagement features...")
        df = self.create_engagement_features(df)
        
        print("Creating support features...")
        df = self.create_support_features(df)
        
        print("Creating billing features...")
        df = self.create_billing_features(df)
        
        print("Creating customer profile features...")
        df = self.create_customer_profile_features(df)
        
        print("Creating interaction features...")
        df = self.create_interaction_features(df)
        
        print("Creating risk flags...")
        df = self.create_risk_flags(df)
        
        print("Encoding categorical variables...")
        df = self.encode_categorical(df, fit=fit)
        
        print("Scaling numerical features...")
        df = self.scale_features(df, fit=fit)
        
        return df

def main():
    """Load data, engineer features, and save"""
    # Load raw data
    print("Loading raw data...")
    df = pd.read_csv('data/raw/saas_customers.csv')
    
    # Initialize feature engineer
    engineer = ChurnFeatureEngineer()
    
    # Engineer features
    df_processed = engineer.engineer_features(df, fit=True)
    
    # Save processed data
    output_path = 'data/processed/features_engineered.csv'
    df_processed.to_csv(output_path, index=False)
    print(f"\nProcessed data saved to {output_path}")
    
    print(f"\nOriginal features: {len(df.columns)}")
    print(f"Engineered features: {len(df_processed.columns)}")
    print(f"New features created: {len(df_processed.columns) - len(df.columns)}")
    
    print("\nNew feature samples:")
    new_cols = [col for col in df_processed.columns if col not in df.columns]
    print(df_processed[new_cols[:10]].head())

if __name__ == '__main__':
    main()
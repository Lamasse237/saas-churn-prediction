"""
Generate realistic SaaS customer data with churn patterns
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

class SaaSDataGenerator:
    def __init__(self, n_customers=10000):
        self.n_customers = n_customers
        self.start_date = datetime(2022, 1, 1)
        self.end_date = datetime(2024, 12, 31)
        
    def generate_customer_base(self):
        """Generate base customer information"""
        data = {
            'customer_id': [f'CUST_{i:06d}' for i in range(self.n_customers)],
            'signup_date': [self._random_date() for _ in range(self.n_customers)],
            'company_size': np.random.choice(
                ['small', 'medium', 'large'], 
                self.n_customers, 
                p=[0.5, 0.35, 0.15]
            ),
            'industry': np.random.choice(
                ['Tech', 'Finance', 'Healthcare', 'Retail', 'Education'],
                self.n_customers
            ),
            'plan_type': np.random.choice(
                ['Basic', 'Professional', 'Enterprise'],
                self.n_customers,
                p=[0.4, 0.4, 0.2]
            ),
        }
        
        df = pd.DataFrame(data)
        
        # Add pricing based on plan
        plan_prices = {'Basic': 49, 'Professional': 199, 'Enterprise': 999}
        df['monthly_revenue'] = df['plan_type'].map(plan_prices)
        
        # Add variation to enterprise pricing
        mask = df['plan_type'] == 'Enterprise'
        df.loc[mask, 'monthly_revenue'] *= np.random.uniform(1, 3, mask.sum())
        
        return df
    
    def _random_date(self):
        """Generate random signup date"""
        delta = self.end_date - self.start_date
        random_days = random.randint(0, delta.days)
        return self.start_date + timedelta(days=random_days)
    
    def generate_usage_features(self, df):
        """Generate usage behavior features"""
        tenure_days = (self.end_date - df['signup_date']).dt.days
        
        # Base engagement levels
        df['avg_daily_logins'] = np.random.lognormal(1, 0.8, len(df))
        df['features_used'] = np.random.poisson(5, len(df))
        df['avg_session_duration'] = np.random.gamma(2, 15, len(df))  # minutes
        
        # Feature adoption (0-10 features)
        df['feature_adoption_rate'] = np.random.beta(2, 3, len(df))
        
        # User seats
        company_size_seats = {'small': (1, 5), 'medium': (5, 50), 'large': (50, 500)}
        df['num_seats'] = df['company_size'].apply(
            lambda x: random.randint(*company_size_seats[x])
        )
        
        # Active users (percentage of seats)
        df['active_users_pct'] = np.random.beta(3, 2, len(df))
        
        # Declining engagement pattern (key churn signal)
        df['usage_trend_30d'] = np.random.normal(0, 0.3, len(df))  # % change
        df['usage_trend_60d'] = np.random.normal(0, 0.35, len(df))
        
        return df
    
    def generate_support_features(self, df):
        """Generate customer support interaction features"""
        df['support_tickets_30d'] = np.random.poisson(1.5, len(df))
        df['avg_resolution_time'] = np.random.gamma(2, 24, len(df))  # hours
        df['satisfaction_score'] = np.random.beta(4, 1.5, len(df)) * 5  # 0-5 scale
        
        # Critical issues
        df['critical_issues_30d'] = np.random.poisson(0.3, len(df))
        
        return df
    
    def generate_billing_features(self, df):
        """Generate billing and payment features"""
        df['payment_method'] = np.random.choice(
            ['credit_card', 'invoice', 'paypal'],
            len(df),
            p=[0.6, 0.3, 0.1]
        )
        
        df['payment_failures_90d'] = np.random.poisson(0.5, len(df))
        df['contract_length'] = np.random.choice(
            ['monthly', 'annual', 'bi-annual'],
            len(df),
            p=[0.5, 0.4, 0.1]
        )
        
        # Days until renewal
        df['days_to_renewal'] = np.random.randint(0, 365, len(df))
        
        # Discount
        df['has_discount'] = np.random.choice([0, 1], len(df), p=[0.7, 0.3])
        df['discount_pct'] = df['has_discount'] * np.random.uniform(0.1, 0.3, len(df))
        
        return df
    
    def generate_churn_labels(self, df):
        """
        Generate churn labels based on realistic patterns
        Churn probability increases with:
        - Declining usage
        - Support issues
        - Payment problems
        - Lower engagement
        """
        churn_score = 0
        
        # Usage factors
        churn_score += (df['usage_trend_30d'] < -0.2).astype(int) * 0.3
        churn_score += (df['avg_daily_logins'] < 1).astype(int) * 0.2
        churn_score += (df['feature_adoption_rate'] < 0.3).astype(int) * 0.15
        
        # Support factors
        churn_score += (df['support_tickets_30d'] > 3).astype(int) * 0.2
        churn_score += (df['satisfaction_score'] < 3).astype(int) * 0.25
        churn_score += (df['critical_issues_30d'] > 0).astype(int) * 0.3
        
        # Billing factors
        churn_score += (df['payment_failures_90d'] > 0).astype(int) * 0.4
        churn_score += (df['contract_length'] == 'monthly').astype(int) * 0.1
        
        # Engagement factors
        churn_score += (df['active_users_pct'] < 0.3).astype(int) * 0.2
        
        # Convert to probability
        churn_prob = 1 / (1 + np.exp(-2 * (churn_score - 0.5)))
        
        # Add some randomness
        churn_prob = np.clip(churn_prob + np.random.normal(0, 0.1, len(df)), 0, 1)
        
        # Generate actual churn (30-day)
        df['churned_30d'] = (np.random.random(len(df)) < churn_prob).astype(int)
        
        # 60-day churn (higher probability)
        df['churned_60d'] = (np.random.random(len(df)) < churn_prob * 1.3).astype(int)
        
        # 90-day churn (even higher)
        df['churned_90d'] = (np.random.random(len(df)) < churn_prob * 1.5).astype(int)
        
        # Ensure consistency (if churned in 30d, also churned in 60d and 90d)
        df.loc[df['churned_30d'] == 1, 'churned_60d'] = 1
        df.loc[df['churned_60d'] == 1, 'churned_90d'] = 1
        
        df['churn_risk_score'] = churn_prob
        
        return df
    
    def generate_dataset(self):
        """Generate complete dataset"""
        print("Generating customer base...")
        df = self.generate_customer_base()
        
        print("Generating usage features...")
        df = self.generate_usage_features(df)
        
        print("Generating support features...")
        df = self.generate_support_features(df)
        
        print("Generating billing features...")
        df = self.generate_billing_features(df)
        
        print("Generating churn labels...")
        df = self.generate_churn_labels(df)
        
        # Calculate customer lifetime value
        tenure_months = ((self.end_date - df['signup_date']).dt.days / 30).astype(int)
        df['customer_lifetime_value'] = df['monthly_revenue'] * tenure_months
        
        print(f"\nDataset generated: {len(df)} customers")
        print(f"Churn rate (30d): {df['churned_30d'].mean():.2%}")
        print(f"Churn rate (60d): {df['churned_60d'].mean():.2%}")
        print(f"Churn rate (90d): {df['churned_90d'].mean():.2%}")
        
        return df

def main():
    """Generate and save dataset"""
    generator = SaaSDataGenerator(n_customers=10000)
    df = generator.generate_dataset()
    
    # Save to CSV
    output_path = 'data/raw/saas_customers.csv'
    df.to_csv(output_path, index=False)
    print(f"\nDataset saved to {output_path}")
    
    # Print sample
    print("\nSample data:")
    print(df.head())
    
    print("\nDataset info:")
    print(df.info())
    
    print("\nChurn statistics:")
    print(df[['churned_30d', 'churned_60d', 'churned_90d']].describe())

if __name__ == '__main__':
    main()
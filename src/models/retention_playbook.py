"""
Customer segmentation and retention intervention playbook
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

class RetentionPlaybook:
    def __init__(self):
        self.segments = None
        self.interventions = self._define_interventions()
        
    def _define_interventions(self):
        """Define intervention strategies with expected outcomes"""
        return {
            'executive_outreach': {
                'description': 'Executive/CSM personal outreach call',
                'cost': 200,
                'success_rate': 0.65,
                'timeline': '24-48 hours',
                'applicability': ['high_value_at_risk', 'enterprise_churn_risk']
            },
            'discount_offer': {
                'description': '20% discount for 3 months',
                'cost': 0,  # Discount, not cost
                'success_rate': 0.45,
                'timeline': 'Immediate',
                'applicability': ['price_sensitive', 'medium_value_at_risk']
            },
            'onboarding_refresh': {
                'description': 'Dedicated onboarding specialist session',
                'cost': 150,
                'success_rate': 0.55,
                'timeline': '3-5 days',
                'applicability': ['low_engagement', 'new_customer_struggle']
            },
            'feature_training': {
                'description': 'Customized feature training webinar',
                'cost': 100,
                'success_rate': 0.50,
                'timeline': '1 week',
                'applicability': ['low_adoption', 'underutilizers']
            },
            'technical_review': {
                'description': 'Technical health check and optimization',
                'cost': 250,
                'success_rate': 0.60,
                'timeline': '1-2 weeks',
                'applicability': ['technical_issues', 'integration_problems']
            },
            'success_plan': {
                'description': 'Quarterly business review and success planning',
                'cost': 300,
                'success_rate': 0.70,
                'timeline': '2 weeks',
                'applicability': ['strategic_accounts', 'high_value_at_risk']
            },
            'community_engagement': {
                'description': 'Invite to user community and events',
                'cost': 50,
                'success_rate': 0.35,
                'timeline': 'Ongoing',
                'applicability': ['isolated_users', 'low_engagement']
            },
            'product_roadmap_preview': {
                'description': 'Early access to new features',
                'cost': 0,
                'success_rate': 0.40,
                'timeline': 'Immediate',
                'applicability': ['feature_requesters', 'power_users']
            },
            'account_upgrade': {
                'description': 'Upgrade path discussion with added value',
                'cost': 100,
                'success_rate': 0.55,
                'timeline': '1 week',
                'applicability': ['expansion_ready', 'hitting_limits']
            },
            'automated_nurture': {
                'description': 'Automated email campaign with best practices',
                'cost': 10,
                'success_rate': 0.25,
                'timeline': 'Immediate',
                'applicability': ['low_risk', 'passive_users']
            }
        }
    
    def segment_customers(self, df, n_segments=6):
        """
        Segment customers based on churn risk and value
        """
        # Features for segmentation
        segment_features = [
            'monthly_revenue',
            'churn_risk_score',
            'engagement_score',
            'support_health',
            'tenure_months',
            'usage_trend_30d'
        ]
        
        # Ensure all features exist
        segment_features = [f for f in segment_features if f in df.columns]
        
        # Prepare data
        X_segment = df[segment_features].fillna(0)
        
        # Normalize for clustering
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_segment)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_segments, random_state=42, n_init=10)
        df['segment'] = kmeans.fit_predict(X_scaled)
        
        # Analyze segments
        segment_profiles = df.groupby('segment').agg({
            'monthly_revenue': 'mean',
            'churn_risk_score': 'mean',
            'engagement_score': 'mean',
            'support_health': 'mean',
            'tenure_months': 'mean',
            'customer_id': 'count'
        }).round(2)
        
        segment_profiles.columns = [
            'avg_revenue', 'avg_churn_risk', 'avg_engagement',
            'avg_support_health', 'avg_tenure', 'customer_count'
        ]
        
        # Name segments based on characteristics
        segment_names = self._name_segments(segment_profiles)
        df['segment_name'] = df['segment'].map(segment_names)
        
        self.segments = segment_profiles
        self.segment_names = segment_names
        
        return df, segment_profiles, segment_names
    
    def _name_segments(self, profiles):
        """Assign meaningful names to segments"""
        names = {}
        
        for idx, row in profiles.iterrows():
            if row['avg_revenue'] > 500 and row['avg_churn_risk'] > 0.6:
                names[idx] = 'high_value_at_risk'
            elif row['avg_revenue'] > 500 and row['avg_churn_risk'] < 0.4:
                names[idx] = 'champions'
            elif row['avg_engagement'] < 40 and row['avg_tenure'] < 6:
                names[idx] = 'new_customer_struggle'
            elif row['avg_engagement'] < 30:
                names[idx] = 'low_engagement'
            elif row['avg_support_health'] < 60:
                names[idx] = 'technical_issues'
            elif row['avg_churn_risk'] > 0.5:
                names[idx] = 'medium_value_at_risk'
            else:
                names[idx] = 'stable_users'
        
        return names
    
    def recommend_interventions(self, df):
        """
        Recommend specific interventions for each customer
        """
        df = df.copy()
        interventions_list = []
        
        for idx, row in df.iterrows():
            customer_interventions = []
            
            # High-value at-risk customers
            if row['monthly_revenue'] > 500 and row.get('churn_risk_score', 0) > 0.6:
                customer_interventions.extend([
                    'executive_outreach',
                    'success_plan',
                    'technical_review'
                ])
            
            # Low engagement
            if row.get('engagement_score', 50) < 40:
                customer_interventions.extend([
                    'onboarding_refresh',
                    'feature_training',
                    'community_engagement'
                ])
            
            # Support issues
            if row.get('support_intensity', 0) > 3:
                customer_interventions.append('technical_review')
            
            # Payment issues
            if row.get('payment_failures_90d', 0) > 0:
                customer_interventions.append('executive_outreach')
            
            # Declining usage
            if row.get('usage_trend_30d', 0) < -0.2:
                customer_interventions.extend([
                    'feature_training',
                    'success_plan'
                ])
            
            # New customers struggling
            if row.get('tenure_months', 12) < 3 and row.get('engagement_score', 50) < 50:
                customer_interventions.append('onboarding_refresh')
            
            # Remove duplicates and prioritize
            customer_interventions = list(dict.fromkeys(customer_interventions))
            
            # If no specific interventions, default based on risk
            if not customer_interventions:
                if row.get('churn_risk_score', 0) > 0.5:
                    customer_interventions = ['discount_offer']
                else:
                    customer_interventions = ['automated_nurture']
            
            interventions_list.append(customer_interventions[:3])  # Top 3
        
        df['recommended_interventions'] = interventions_list
        df['primary_intervention'] = df['recommended_interventions'].apply(
            lambda x: x[0] if x else 'none'
        )
        
        return df
    
    def calculate_intervention_roi(self, df):
        """
        Calculate expected ROI for each intervention
        """
        df = df.copy()
        
        # Calculate revenue at risk (12-month LTV)
        df['revenue_at_risk'] = df['monthly_revenue'] * 12
        
        roi_data = []
        
        for intervention, details in self.interventions.items():
            # Filter applicable customers
            mask = df['primary_intervention'] == intervention
            
            if mask.sum() == 0:
                continue
            
            customers = df[mask]
            
            # Calculate expected outcomes
            total_customers = len(customers)
            total_revenue_at_risk = customers['revenue_at_risk'].sum()
            intervention_cost = details['cost'] * total_customers
            
            # Expected revenue saved
            expected_saves = total_customers * details['success_rate']
            expected_revenue_saved = total_revenue_at_risk * details['success_rate']
            
            # ROI
            roi = (expected_revenue_saved - intervention_cost) / intervention_cost if intervention_cost > 0 else np.inf
            
            roi_data.append({
                'intervention': intervention,
                'customers': total_customers,
                'total_cost': intervention_cost,
                'revenue_at_risk': total_revenue_at_risk,
                'expected_saves': expected_saves,
                'expected_revenue_saved': expected_revenue_saved,
                'roi': roi,
                'success_rate': details['success_rate']
            })
        
        roi_df = pd.DataFrame(roi_data).sort_values('expected_revenue_saved', ascending=False)
        
        return roi_df
    
    def generate_playbook_report(self, df, output_path='reports/retention_playbook.txt'):
        """Generate comprehensive retention playbook"""
        
        report = []
        report.append("="*80)
        report.append("CUSTOMER RETENTION PLAYBOOK")
        report.append("="*80)
        report.append("")
        
        # Segment overview
        report.append("CUSTOMER SEGMENTS")
        report.append("-"*80)
        
        if self.segments is not None:
            for idx, name in self.segment_names.items():
                segment_data = self.segments.loc[idx]
                report.append(f"\nSegment: {name.upper()}")
                report.append(f"  Customers: {int(segment_data['customer_count'])}")
                report.append(f"  Avg Revenue: ${segment_data['avg_revenue']:.2f}/month")
                report.append(f"  Avg Churn Risk: {segment_data['avg_churn_risk']:.1%}")
                report.append(f"  Avg Engagement: {segment_data['avg_engagement']:.1f}/100")
        
        report.append("")
        report.append("="*80)
        report.append("INTERVENTION STRATEGIES")
        report.append("="*80)
        
        # ROI analysis
        roi_df = self.calculate_intervention_roi(df)
        
        report.append("\nINTERVENTION ROI ANALYSIS")
        report.append("-"*80)
        
        for _, row in roi_df.iterrows():
            intervention_details = self.interventions[row['intervention']]
            
            report.append(f"\n{row['intervention'].upper().replace('_', ' ')}")
            report.append(f"  Description: {intervention_details['description']}")
            report.append(f"  Target Customers: {int(row['customers'])}")
            report.append(f"  Total Cost: ${row['total_cost']:,.2f}")
            report.append(f"  Revenue at Risk: ${row['revenue_at_risk']:,.2f}")
            report.append(f"  Expected Revenue Saved: ${row['expected_revenue_saved']:,.2f}")
            report.append(f"  Success Rate: {row['success_rate']:.1%}")
            if row['roi'] != np.inf:
                report.append(f"  ROI: {row['roi']:.1f}x")
            else:
                report.append(f"  ROI: Infinite (no cost)")
            report.append(f"  Timeline: {intervention_details['timeline']}")
        
        report.append("")
        report.append("="*80)
        report.append("PRIORITY ACTION ITEMS")
        report.append("="*80)
        
        # High-priority customers
        high_risk = df[df.get('churn_risk_score', 0) > 0.7].sort_values(
            'monthly_revenue', ascending=False
        )
        
        report.append(f"\nIMMEDIATE ACTION REQUIRED: {len(high_risk)} high-risk customers")
        report.append("-"*80)
        
        for idx, (_, customer) in enumerate(high_risk.head(10).iterrows(), 1):
            report.append(f"\n{idx}. Customer {customer['customer_id']}")
            report.append(f"   Revenue: ${customer['monthly_revenue']:.2f}/month")
            report.append(f"   Churn Risk: {customer.get('churn_risk_score', 0):.1%}")
            report.append(f"   Recommended: {customer.get('primary_intervention', 'N/A')}")
            if 'segment_name' in customer:
                report.append(f"   Segment: {customer['segment_name']}")
        
        # Save report
        with open(output_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"\nRetention playbook saved to {output_path}")
        print('\n'.join(report))
        
        return '\n'.join(report)

def main():
    """Generate retention playbook"""
    # Load data with predictions
    df = pd.read_csv('data/processed/features_engineered.csv')
    
    # Load best model (assume XGBoost 30d)
    model = joblib.load('models/xgboost_30d.pkl')
    
    # Get predictions
    feature_cols = [col for col in df.columns if col not in [
        'customer_id', 'signup_date', 'churned_30d', 'churned_60d',
        'churned_90d', 'churn_risk_score', 'customer_lifetime_value',
        'company_size', 'industry', 'plan_type', 'payment_method',
        'contract_length', 'revenue_tier', 'tenure_bucket'
    ]]
    
    X = df[feature_cols]
    df['churn_risk_score'] = model.predict_proba(X)[:, 1]
    
    # Initialize playbook
    playbook = RetentionPlaybook()
    
    # Segment customers
    print("Segmenting customers...")
    df, segments, segment_names = playbook.segment_customers(df)
    
    # Recommend interventions
    print("\nRecommending interventions...")
    df = playbook.recommend_interventions(df)
    
    # Generate report
    print("\nGenerating playbook report...")
    playbook.generate_playbook_report(df)
    
    # Save enhanced data
    df.to_csv('data/processed/customers_with_recommendations.csv', index=False)
    print("\nEnhanced data saved to data/processed/customers_with_recommendations.csv")

if __name__ == '__main__':
    main()
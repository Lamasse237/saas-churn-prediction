"""
Survival analysis for time-to-churn prediction
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import seaborn as sns

class ChurnSurvivalAnalysis:
    def __init__(self):
        self.kmf = KaplanMeierFitter()
        self.cph = CoxPHFitter()
        
    def prepare_survival_data(self, df):
        """
        Prepare data for survival analysis
        Duration: days until churn or censoring
        Event: whether customer churned (1) or is still active (0)
        """
        df = df.copy()
        
        # Calculate tenure
        reference_date = pd.to_datetime('2024-12-31')
        df['signup_date'] = pd.to_datetime(df['signup_date'])
        df['duration_days'] = (reference_date - df['signup_date']).dt.days
        
        # Event indicator (using 30d churn as event)
        df['event'] = df['churned_30d']
        
        # For churned customers, assume they churned midway through observation
        # For active customers, they are censored at the observation end
        df.loc[df['event'] == 1, 'duration_days'] = df.loc[df['event'] == 1, 'duration_days'] * 0.7
        
        return df
    
    def fit_kaplan_meier(self, df, duration_col='duration_days', event_col='event'):
        """Fit Kaplan-Meier survival curve"""
        self.kmf.fit(
            durations=df[duration_col],
            event_observed=df[event_col],
            label='All Customers'
        )
        
        return self.kmf
    
    def plot_survival_curve(self, df, save_path='reports/survival_curve.png'):
        """Plot overall survival curve"""
        self.fit_kaplan_meier(df)
        
        plt.figure(figsize=(12, 6))
        self.kmf.plot_survival_function()
        plt.title('Customer Survival Curve (Kaplan-Meier)', fontsize=14, fontweight='bold')
        plt.xlabel('Days Since Signup', fontsize=12)
        plt.ylabel('Survival Probability (% Still Active)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add median survival time
        median_survival = self.kmf.median_survival_time_
        plt.axvline(median_survival, color='red', linestyle='--', alpha=0.7,
                   label=f'Median Survival: {median_survival:.0f} days')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Survival curve saved to {save_path}")
        print(f"Median survival time: {median_survival:.0f} days")
        
        # Survival at key timepoints
        print("\nSurvival Probabilities:")
        for days in [30, 90, 180, 365]:
            prob = self.kmf.survival_function_at_times(days).values[0]
            print(f"  {days} days: {prob:.1%}")
    
    def compare_groups(self, df, group_col, save_path='reports/survival_by_group.png'):
        """Compare survival curves across groups"""
        plt.figure(figsize=(14, 7))
        
        groups = df[group_col].unique()
        results = []
        
        for group in groups:
            mask = df[group_col] == group
            group_data = df[mask]
            
            kmf = KaplanMeierFitter()
            kmf.fit(
                durations=group_data['duration_days'],
                event_observed=group_data['event'],
                label=str(group)
            )
            kmf.plot_survival_function()
            
            results.append({
                'group': group,
                'median_survival': kmf.median_survival_time_,
                'customers': len(group_data)
            })
        
        plt.title(f'Survival Curves by {group_col}', fontsize=14, fontweight='bold')
        plt.xlabel('Days Since Signup', fontsize=12)
        plt.ylabel('Survival Probability', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nSurvival curves by {group_col} saved to {save_path}")
        
        # Print comparison
        results_df = pd.DataFrame(results).sort_values('median_survival', ascending=False)
        print(f"\nMedian Survival Time by {group_col}:")
        print(results_df)
        
        return results_df
    
    def cox_proportional_hazards(self, df):
        """
        Fit Cox Proportional Hazards model
        Identifies which factors increase/decrease churn risk
        """
        # Select features for Cox model
        cox_features = [
            'monthly_revenue',
            'engagement_score',
            'support_intensity',
            'payment_failures_90d',
            'usage_trend_30d',
            'feature_adoption_rate',
            'active_users_pct',
            'company_size_score',
            'plan_value'
        ]
        
        # Ensure features exist
        cox_features = [f for f in cox_features if f in df.columns]
        
        # Prepare data
        cox_data = df[['duration_days', 'event'] + cox_features].copy()
        cox_data = cox_data.dropna()
        
        # Fit model
        self.cph.fit(cox_data, duration_col='duration_days', event_col='event')
        
        # Print summary
        print("\n" + "="*80)
        print("COX PROPORTIONAL HAZARDS MODEL")
        print("="*80)
        print("\nHazard Ratios (>1 increases churn risk, <1 decreases):")
        print(self.cph.summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']])
        
        # Save summary
        self.cph.summary.to_csv('reports/cox_model_summary.csv')
        print("\nCox model summary saved to reports/cox_model_summary.csv")
        
        # Plot hazard ratios
        plt.figure(figsize=(10, 8))
        self.cph.plot()
        plt.title('Hazard Ratios - Impact on Churn Risk', fontsize=14, fontweight='bold')
        plt.xlabel('Hazard Ratio (log scale)', fontsize=12)
        plt.tight_layout()
        plt.savefig('reports/hazard_ratios.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Hazard ratio plot saved to reports/hazard_ratios.png")
        
        return self.cph
    
    def predict_survival_probability(self, df, customer_id, time_points=[30, 60, 90, 180, 365]):
        """Predict survival probability for specific customer"""
        customer = df[df['customer_id'] == customer_id].iloc[0]
        
        cox_features = [
            'monthly_revenue',
            'engagement_score',
            'support_intensity',
            'payment_failures_90d',
            'usage_trend_30d',
            'feature_adoption_rate',
            'active_users_pct',
            'company_size_score',
            'plan_value'
        ]
        
        cox_features = [f for f in cox_features if f in df.columns]
        customer_features = customer[cox_features].to_frame().T
        
        # Predict survival function
        survival_func = self.cph.predict_survival_function(customer_features)
        
        print(f"\nSurvival Probabilities for Customer {customer_id}:")
        for days in time_points:
            if days in survival_func.index:
                prob = survival_func.loc[days].values[0]
                print(f"  {days} days: {prob:.1%} chance of retention")
        
        return survival_func
    
    def generate_survival_report(self, df, output_path='reports/survival_analysis_report.txt'):
        """Generate comprehensive survival analysis report"""
        report = []
        
        report.append("="*80)
        report.append("SURVIVAL ANALYSIS REPORT")
        report.append("="*80)
        report.append("")
        
        # Overall survival
        self.fit_kaplan_meier(df)
        median = self.kmf.median_survival_time_
        
        report.append("OVERALL CUSTOMER SURVIVAL")
        report.append("-"*80)
        report.append(f"Median Survival Time: {median:.0f} days ({median/30:.1f} months)")
        report.append("")
        report.append("Survival at Key Milestones:")
        
        for days in [30, 60, 90, 180, 365, 730]:
            prob = self.kmf.survival_function_at_times(days).values[0]
            report.append(f"  {days:4d} days ({days/30:5.1f} months): {prob:6.1%} still active")
        
        report.append("")
        report.append("="*80)
        report.append("SURVIVAL BY CUSTOMER SEGMENT")
        report.append("="*80)
        
        # By plan type
        if 'plan_type' in df.columns:
            report.append("\nBy Plan Type:")
            results = self.compare_groups(df, 'plan_type', 
                                         save_path='reports/survival_by_plan.png')
            for _, row in results.iterrows():
                report.append(f"  {row['group']:15s}: {row['median_survival']:6.0f} days "
                            f"({row['median_survival']/30:4.1f} months) - {int(row['customers'])} customers")
        
        # By company size
        if 'company_size' in df.columns:
            report.append("\nBy Company Size:")
            results = self.compare_groups(df, 'company_size',
                                         save_path='reports/survival_by_size.png')
            for _, row in results.iterrows():
                report.append(f"  {row['group']:15s}: {row['median_survival']:6.0f} days "
                            f"({row['median_survival']/30:4.1f} months) - {int(row['customers'])} customers")
        
        # Cox model
        report.append("")
        report.append("="*80)
        report.append("CHURN RISK FACTORS (Cox Proportional Hazards)")
        report.append("="*80)
        report.append("\nTop factors increasing churn risk (Hazard Ratio > 1):")
        
        summary = self.cph.summary.sort_values('exp(coef)', ascending=False)
        for idx, row in summary[summary['exp(coef)'] > 1].head(5).iterrows():
            report.append(f"  {idx:30s}: {row['exp(coef)']:5.2f}x "
                        f"(p={row['p']:.4f})")
        
        report.append("\nTop factors decreasing churn risk (Hazard Ratio < 1):")
        for idx, row in summary[summary['exp(coef)'] < 1].head(5).iterrows():
            report.append(f"  {idx:30s}: {row['exp(coef)']:5.2f}x "
                        f"(p={row['p']:.4f})")
        
        # Save report
        with open(output_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"\n\nSurvival analysis report saved to {output_path}")
        print('\n'.join(report))

def main():
    """Run complete survival analysis"""
    # Load data
    df = pd.read_csv('data/processed/features_engineered.csv')
    
    # Initialize analyzer
    analyzer = ChurnSurvivalAnalysis()
    
    # Prepare data
    print("Preparing survival data...")
    df = analyzer.prepare_survival_data(df)
    
    # Overall survival curve
    print("\nFitting Kaplan-Meier survival curve...")
    analyzer.plot_survival_curve(df)
    
    # Compare by segments
    print("\nComparing survival across segments...")
    if 'plan_type' in df.columns:
        analyzer.compare_groups(df, 'plan_type', 'reports/survival_by_plan.png')
    
    if 'company_size' in df.columns:
        analyzer.compare_groups(df, 'company_size', 'reports/survival_by_size.png')
    
    # Cox proportional hazards
    print("\nFitting Cox Proportional Hazards model...")
    analyzer.cox_proportional_hazards(df)
    
    # Generate report
    print("\nGenerating survival analysis report...")
    analyzer.generate_survival_report(df)

if __name__ == '__main__':
    main()
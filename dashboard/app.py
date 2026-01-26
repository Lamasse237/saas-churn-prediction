"""
Interactive Churn Prediction Dashboard
Run with: streamlit run dashboard/app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page config
st.set_page_config(
    page_title="SaaS Churn Prediction System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .risk-high {
        color: #d62728;
        font-weight: bold;
    }
    .risk-medium {
        color: #ff7f0e;
        font-weight: bold;
    }
    .risk-low {
        color: #2ca02c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load processed data with predictions"""
    df = pd.read_csv('data/processed/customers_with_recommendations.csv')
    return df

@st.cache_resource
def load_models():
    """Load trained models"""
    models = {
        '30d': joblib.load('models/xgboost_30d.pkl'),
        '60d': joblib.load('models/xgboost_60d.pkl'),
        '90d': joblib.load('models/xgboost_90d.pkl')
    }
    return models

def risk_category(score):
    """Categorize churn risk"""
    if score >= 0.7:
        return "High"
    elif score >= 0.4:
        return "Medium"
    else:
        return "Low"

def main():
    # Header
    st.markdown('<div class="main-header">SaaS Churn Prediction System</div>', 
                unsafe_allow_html=True)
    st.markdown("**Early Warning System with Actionable Interventions**")
    
    # Load data
    try:
        df = load_data()
    except FileNotFoundError:
        st.error("Data not found. Please run the data generation and model training scripts first.")
        return
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Time horizon selection
    horizon = st.sidebar.selectbox(
        "Prediction Horizon",
        ['30 days', '60 days', '90 days'],
        index=0
    )
    horizon_key = horizon.split()[0] + 'd'
    
    # Risk level filter
    risk_filter = st.sidebar.multiselect(
        "Risk Level",
        ['High', 'Medium', 'Low'],
        default=['High', 'Medium', 'Low']
    )
    
    # Plan type filter
    if 'plan_type' in df.columns:
        plan_filter = st.sidebar.multiselect(
            "Plan Type",
            df['plan_type'].unique().tolist(),
            default=df['plan_type'].unique().tolist()
        )
    else:
        plan_filter = None
    
    # Revenue filter
    min_revenue, max_revenue = st.sidebar.slider(
        "Monthly Revenue Range ($)",
        float(df['monthly_revenue'].min()),
        float(df['monthly_revenue'].max()),
        (float(df['monthly_revenue'].min()), float(df['monthly_revenue'].max()))
    )
    
    # Apply filters
    df['risk_category'] = df['churn_risk_score'].apply(risk_category)
    
    filtered_df = df[
        (df['risk_category'].isin(risk_filter)) &
        (df['monthly_revenue'] >= min_revenue) &
        (df['monthly_revenue'] <= max_revenue)
    ]
    
    if plan_filter:
        filtered_df = filtered_df[filtered_df['plan_type'].isin(plan_filter)]
    
    # Main metrics
    st.header("Overview Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Customers",
            f"{len(filtered_df):,}",
            f"{len(filtered_df)/len(df)*100:.1f}% of total"
        )
    
    with col2:
        high_risk = (filtered_df['churn_risk_score'] >= 0.7).sum()
        st.metric(
            "High Risk Customers",
            f"{high_risk:,}",
            f"{high_risk/len(filtered_df)*100:.1f}%"
        )
    
    with col3:
        revenue_at_risk = filtered_df[filtered_df['churn_risk_score'] >= 0.5]['monthly_revenue'].sum() * 12
        st.metric(
            "Revenue at Risk (Annual)",
            f"${revenue_at_risk:,.0f}",
            delta_color="inverse"
        )
    
    with col4:
        avg_risk = filtered_df['churn_risk_score'].mean()
        st.metric(
            "Average Churn Risk",
            f"{avg_risk:.1%}",
            delta_color="inverse"
        )
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Risk Dashboard", 
        "👥 Customer Segments", 
        "💡 Interventions", 
        "📈 Analysis",
        "🔍 Customer Lookup"
    ])
    
    with tab1:
        st.header("Risk Dashboard")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk distribution
            risk_dist = filtered_df['risk_category'].value_counts()
            fig = px.pie(
                values=risk_dist.values,
                names=risk_dist.index,
                title="Customer Risk Distribution",
                color=risk_dist.index,
                color_discrete_map={'High': '#d62728', 'Medium': '#ff7f0e', 'Low': '#2ca02c'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Revenue by risk
            revenue_by_risk = filtered_df.groupby('risk_category')['monthly_revenue'].sum().sort_values()
            fig = px.bar(
                x=revenue_by_risk.values,
                y=revenue_by_risk.index,
                orientation='h',
                title="Monthly Revenue by Risk Category",
                labels={'x': 'Monthly Revenue ($)', 'y': 'Risk Category'},
                color=revenue_by_risk.index,
                color_discrete_map={'High': '#d62728', 'Medium': '#ff7f0e', 'Low': '#2ca02c'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Risk score distribution
        fig = px.histogram(
            filtered_df,
            x='churn_risk_score',
            nbins=50,
            title="Churn Risk Score Distribution",
            labels={'churn_risk_score': 'Churn Risk Score'},
            color_discrete_sequence=['#1f77b4']
        )
        fig.add_vline(x=0.5, line_dash="dash", line_color="orange", 
                     annotation_text="Medium Risk Threshold")
        fig.add_vline(x=0.7, line_dash="dash", line_color="red",
                     annotation_text="High Risk Threshold")
        st.plotly_chart(fig, use_container_width=True)
        
        # Top at-risk customers
        st.subheader("Top 20 At-Risk Customers")
        top_risk = filtered_df.nlargest(20, 'churn_risk_score')[
            ['customer_id', 'monthly_revenue', 'churn_risk_score', 
             'engagement_score', 'primary_intervention', 'segment_name']
        ].copy()
        
        top_risk['churn_risk_score'] = top_risk['churn_risk_score'].apply(lambda x: f"{x:.1%}")
        top_risk['monthly_revenue'] = top_risk['monthly_revenue'].apply(lambda x: f"${x:,.2f}")
        top_risk['engagement_score'] = top_risk['engagement_score'].apply(lambda x: f"{x:.1f}")
        
        st.dataframe(top_risk, use_container_width=True)
    
    with tab2:
        st.header("Customer Segments")
        
        if 'segment_name' in filtered_df.columns:
            # Segment overview
            segment_stats = filtered_df.groupby('segment_name').agg({
                'customer_id': 'count',
                'monthly_revenue': 'mean',
                'churn_risk_score': 'mean',
                'engagement_score': 'mean'
            }).round(2)
            
            segment_stats.columns = ['Customers', 'Avg Revenue', 'Avg Churn Risk', 'Avg Engagement']
            segment_stats = segment_stats.sort_values('Avg Churn Risk', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Segment sizes
                fig = px.bar(
                    segment_stats,
                    y=segment_stats.index,
                    x='Customers',
                    orientation='h',
                    title="Customers by Segment",
                    labels={'x': 'Number of Customers', 'y': 'Segment'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Risk by segment
                fig = px.bar(
                    segment_stats,
                    y=segment_stats.index,
                    x='Avg Churn Risk',
                    orientation='h',
                    title="Average Churn Risk by Segment",
                    labels={'x': 'Average Churn Risk', 'y': 'Segment'},
                    color='Avg Churn Risk',
                    color_continuous_scale='RdYlGn_r'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Segment details
            st.subheader("Segment Details")
            st.dataframe(segment_stats, use_container_width=True)
            
            # Scatter: Revenue vs Risk by Segment
            # Create positive size values (scaled engagement scores can be negative)
            plot_df = filtered_df.copy()
            plot_df['size_metric'] = abs(plot_df['engagement_score']) + 10
            
            fig = px.scatter(
                plot_df,
                x='monthly_revenue',
                y='churn_risk_score',
                color='segment_name',
                size='size_metric',
                hover_data=['customer_id', 'engagement_score'],
                title="Revenue vs Churn Risk by Segment",
                labels={
                    'monthly_revenue': 'Monthly Revenue ($)',
                    'churn_risk_score': 'Churn Risk Score',
                    'segment_name': 'Segment',
                    'size_metric': 'Engagement'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Intervention Recommendations")
        
        if 'primary_intervention' in filtered_df.columns:
            # Intervention distribution
            intervention_counts = filtered_df['primary_intervention'].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    x=intervention_counts.index,
                    y=intervention_counts.values,
                    title="Recommended Interventions",
                    labels={'x': 'Intervention', 'y': 'Number of Customers'},
                    color=intervention_counts.values,
                    color_continuous_scale='Blues'
                )
                fig.update_xaxes(tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Revenue at risk by intervention
                intervention_revenue = filtered_df.groupby('primary_intervention')['monthly_revenue'].sum() * 12
                fig = px.bar(
                    x=intervention_revenue.index,
                    y=intervention_revenue.values,
                    title="Annual Revenue at Risk by Intervention",
                    labels={'x': 'Intervention', 'y': 'Revenue ($)'},
                    color=intervention_revenue.values,
                    color_continuous_scale='Reds'
                )
                fig.update_xaxes(tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            
            # Intervention playbook
            st.subheader("📘 Intervention Playbook")
            
            interventions = {
                'executive_outreach': {
                    'name': 'Executive Outreach',
                    'description': 'Personal call from executive or CSM',
                    'cost': '$200',
                    'success_rate': '65%',
                    'timeline': '24-48 hours'
                },
                'discount_offer': {
                    'name': 'Discount Offer',
                    'description': '20% discount for 3 months',
                    'cost': 'Revenue reduction',
                    'success_rate': '45%',
                    'timeline': 'Immediate'
                },
                'onboarding_refresh': {
                    'name': 'Onboarding Refresh',
                    'description': 'Dedicated onboarding specialist session',
                    'cost': '$150',
                    'success_rate': '55%',
                    'timeline': '3-5 days'
                },
                'feature_training': {
                    'name': 'Feature Training',
                    'description': 'Customized feature training webinar',
                    'cost': '$100',
                    'success_rate': '50%',
                    'timeline': '1 week'
                },
                'technical_review': {
                    'name': 'Technical Review',
                    'description': 'Technical health check and optimization',
                    'cost': '$250',
                    'success_rate': '60%',
                    'timeline': '1-2 weeks'
                }
            }
            
            for key, intervention in interventions.items():
                with st.expander(f"**{intervention['name']}** - {intervention['description']}"):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Cost", intervention['cost'])
                    col2.metric("Success Rate", intervention['success_rate'])
                    col3.metric("Timeline", intervention['timeline'])
                    
                    # Customers needing this intervention
                    applicable = (filtered_df['primary_intervention'] == key).sum()
                    st.write(f"**{applicable}** customers recommended for this intervention")
    
    with tab4:
        st.header("Advanced Analysis")
        
        # Feature importance (if available)
        st.subheader("Key Churn Indicators")
        
        # Correlation with churn risk
        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
        correlations = filtered_df[numeric_cols].corr()['churn_risk_score'].sort_values(ascending=False)[1:11]
        
        fig = px.bar(
            x=correlations.values,
            y=correlations.index,
            orientation='h',
            title="Top Features Correlated with Churn Risk",
            labels={'x': 'Correlation', 'y': 'Feature'},
            color=correlations.values,
            color_continuous_scale='RdYlGn_r'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Engagement vs Revenue
        st.subheader("Engagement vs Revenue Analysis")
        
        # Create positive size values for tenure
        plot_df = filtered_df.copy()
        plot_df['tenure_size'] = plot_df['tenure_months'].clip(lower=1)
        
        fig = px.scatter(
            plot_df,
            x='engagement_score',
            y='monthly_revenue',
            color='churn_risk_score',
            size='tenure_size',
            hover_data=['customer_id', 'segment_name', 'tenure_months'],
            title="Customer Value & Engagement Matrix",
            labels={
                'engagement_score': 'Engagement Score',
                'monthly_revenue': 'Monthly Revenue ($)',
                'churn_risk_score': 'Churn Risk',
                'tenure_size': 'Tenure (months)'
            },
            color_continuous_scale='RdYlGn_r'
        )
        
        # Add quadrant lines
        median_engagement = filtered_df['engagement_score'].median()
        median_revenue = filtered_df['monthly_revenue'].median()
        
        fig.add_hline(y=median_revenue, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=median_engagement, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Quadrant labels
        fig.add_annotation(x=median_engagement*1.5, y=median_revenue*1.5,
                          text="Stars", showarrow=False, font=dict(size=12, color="green"))
        fig.add_annotation(x=median_engagement*0.5, y=median_revenue*1.5,
                          text="High Value Risk", showarrow=False, font=dict(size=12, color="red"))
        fig.add_annotation(x=median_engagement*1.5, y=median_revenue*0.5,
                          text="Engaged & Growing", showarrow=False, font=dict(size=12, color="blue"))
        fig.add_annotation(x=median_engagement*0.5, y=median_revenue*0.5,
                          text="At Risk", showarrow=False, font=dict(size=12, color="orange"))
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.header("Customer Lookup")
        
        # Search
        customer_id = st.selectbox(
            "Select Customer ID",
            filtered_df['customer_id'].tolist()
        )
        
        if customer_id:
            customer = filtered_df[filtered_df['customer_id'] == customer_id].iloc[0]
            
            # Customer overview
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Monthly Revenue", f"${customer['monthly_revenue']:,.2f}")
            with col2:
                risk_score = customer['churn_risk_score']
                st.metric("Churn Risk", f"{risk_score:.1%}", 
                         delta=f"{risk_category(risk_score)} Risk")
            with col3:
                st.metric("Engagement Score", f"{customer['engagement_score']:.1f}/100")
            with col4:
                st.metric("Tenure", f"{customer['tenure_months']:.1f} months")
            
            # Details
            st.subheader("Customer Details")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Account Information**")
                st.write(f"- Plan: {customer.get('plan_type', 'N/A')}")
                st.write(f"- Company Size: {customer.get('company_size', 'N/A')}")
                st.write(f"- Industry: {customer.get('industry', 'N/A')}")
                st.write(f"- Segment: {customer.get('segment_name', 'N/A')}")
            
            with col2:
                st.write("**Engagement Metrics**")
                st.write(f"- Daily Logins: {customer.get('avg_daily_logins', 0):.1f}")
                st.write(f"- Feature Adoption: {customer.get('feature_adoption_rate', 0):.1%}")
                st.write(f"- Active Users: {customer.get('active_users_pct', 0):.1%}")
                st.write(f"- Usage Trend: {customer.get('usage_trend_30d', 0):.1%}")
            
            # Recommended actions
            st.subheader("Recommended Actions")
            st.info(f"**Primary Intervention:** {customer.get('primary_intervention', 'N/A')}")
            
            # Risk factors
            st.subheader("Risk Factors")
            risk_factors = []
            
            if customer.get('engagement_score', 50) < 40:
                risk_factors.append("Low engagement score")
            if customer.get('usage_trend_30d', 0) < -0.15:
                risk_factors.append("Declining usage trend")
            if customer.get('support_intensity', 0) > 3:
                risk_factors.append("High support ticket volume")
            if customer.get('payment_failures_90d', 0) > 0:
                risk_factors.append("Recent payment failures")
            if customer.get('satisfaction_score', 5) < 3.5:
                risk_factors.append("Low satisfaction score")
            
            if risk_factors:
                for factor in risk_factors:
                    st.write(factor)
            else:
                st.success("✅ No major risk factors identified")

if __name__ == '__main__':
    main()
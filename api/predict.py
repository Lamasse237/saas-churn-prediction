"""
FastAPI endpoint for real-time churn predictions
Run with: uvicorn api.predict:app --reload
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import os
from typing import Optional, Dict, List

app = FastAPI(
    title="SaaS Churn Prediction API",
    description="Predict customer churn with actionable recommendations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models at startup
models = {}
try:
    models['30d'] = joblib.load('models/xgboost_30d.pkl')
    models['60d'] = joblib.load('models/xgboost_60d.pkl')
    models['90d'] = joblib.load('models/xgboost_90d.pkl')
    print("✅ All models loaded successfully!")
except Exception as e:
    print(f"⚠️  Warning: Could not load models - {str(e)}")

class CustomerData(BaseModel):
    """Customer features for prediction"""
    # Basic info
    monthly_revenue: float = Field(..., description="Monthly revenue from customer", gt=0)
    
    # Usage metrics
    avg_daily_logins: float = Field(..., description="Average daily logins", ge=0)
    features_used: int = Field(..., description="Number of features used", ge=0)
    avg_session_duration: float = Field(..., description="Average session duration in minutes", ge=0)
    feature_adoption_rate: float = Field(..., description="Feature adoption rate (0-1)", ge=0, le=1)
    
    # Engagement
    num_seats: int = Field(..., description="Number of seats/licenses", gt=0)
    active_users_pct: float = Field(..., description="Percentage of active users (0-1)", ge=0, le=1)
    usage_trend_30d: float = Field(..., description="30-day usage trend (-1 to 1)")
    usage_trend_60d: float = Field(..., description="60-day usage trend (-1 to 1)")
    
    # Support
    support_tickets_30d: int = Field(..., description="Support tickets in last 30 days", ge=0)
    avg_resolution_time: float = Field(..., description="Average resolution time in hours", ge=0)
    satisfaction_score: float = Field(..., description="Customer satisfaction score (0-5)", ge=0, le=5)
    critical_issues_30d: int = Field(..., description="Critical issues in last 30 days", ge=0)
    
    # Billing
    payment_failures_90d: int = Field(..., description="Payment failures in last 90 days", ge=0)
    days_to_renewal: int = Field(..., description="Days until contract renewal", ge=0)
    has_discount: int = Field(..., description="Has discount (0 or 1)", ge=0, le=1)
    discount_pct: float = Field(..., description="Discount percentage (0-1)", ge=0, le=1)
    
    # Account
    tenure_months: float = Field(..., description="Months as customer", ge=0)
    company_size_score: int = Field(..., description="Company size (1=small, 2=medium, 3=large)", ge=1, le=3)
    plan_value: int = Field(..., description="Plan value (1=basic, 2=pro, 3=enterprise)", ge=1, le=3)
    
    class Config:
        schema_extra = {
            "example": {
                "monthly_revenue": 199.0,
                "avg_daily_logins": 2.5,
                "features_used": 5,
                "avg_session_duration": 25.0,
                "feature_adoption_rate": 0.6,
                "num_seats": 10,
                "active_users_pct": 0.7,
                "usage_trend_30d": -0.15,
                "usage_trend_60d": -0.1,
                "support_tickets_30d": 2,
                "avg_resolution_time": 24.0,
                "satisfaction_score": 4.0,
                "critical_issues_30d": 0,
                "payment_failures_90d": 0,
                "days_to_renewal": 45,
                "has_discount": 0,
                "discount_pct": 0.0,
                "tenure_months": 8.0,
                "company_size_score": 2,
                "plan_value": 2
            }
        }

class PredictionResponse(BaseModel):
    """Churn prediction response"""
    customer_data: Dict
    prediction: Dict
    financial_impact: Dict
    recommendation: Dict
    risk_factors: List[str]

@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "name": "SaaS Churn Prediction API",
        "version": "1.0.0",
        "status": "active",
        "models_loaded": list(models.keys()),
        "endpoints": {
            "/predict": "POST - Get churn prediction for a customer",
            "/batch-predict": "POST - Get predictions for multiple customers",
            "/health": "GET - Health check",
            "/docs": "GET - Interactive API documentation"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": list(models.keys()),
        "models_available": len(models) > 0
    }

def get_intervention_recommendation(churn_prob: float, data: CustomerData) -> Dict:
    """Determine intervention based on risk and customer profile"""
    interventions = {
        'executive_outreach': {
            'name': 'Executive Outreach',
            'description': 'Personal call from executive or CSM',
            'cost': 200,
            'success_rate': 0.65,
            'timeline': '24-48 hours'
        },
        'discount_offer': {
            'name': 'Discount Offer',
            'description': '20% discount for 3 months',
            'cost': data.monthly_revenue * 0.2 * 3,
            'success_rate': 0.45,
            'timeline': 'Immediate'
        },
        'onboarding_refresh': {
            'name': 'Onboarding Refresh',
            'description': 'Dedicated onboarding specialist session',
            'cost': 150,
            'success_rate': 0.55,
            'timeline': '3-5 days'
        },
        'feature_training': {
            'name': 'Feature Training',
            'description': 'Customized feature training webinar',
            'cost': 100,
            'success_rate': 0.50,
            'timeline': '1 week'
        },
        'technical_review': {
            'name': 'Technical Review',
            'description': 'Technical health check and optimization',
            'cost': 250,
            'success_rate': 0.60,
            'timeline': '1-2 weeks'
        }
    }
    
    # Determine primary intervention
    if churn_prob >= 0.7:
        if data.monthly_revenue > 500:
            intervention_key = 'executive_outreach'
        elif data.support_tickets_30d > 3:
            intervention_key = 'technical_review'
        else:
            intervention_key = 'discount_offer'
    elif churn_prob >= 0.4:
        if data.feature_adoption_rate < 0.4:
            intervention_key = 'feature_training'
        else:
            intervention_key = 'onboarding_refresh'
    else:
        intervention_key = 'feature_training'
    
    intervention = interventions[intervention_key]
    
    return {
        'primary_intervention': intervention_key,
        'intervention_name': intervention['name'],
        'description': intervention['description'],
        'estimated_cost': intervention['cost'],
        'expected_success_rate': intervention['success_rate'],
        'timeline': intervention['timeline'],
        'expected_revenue_saved': data.monthly_revenue * 12 * intervention['success_rate'],
        'roi': ((data.monthly_revenue * 12 * intervention['success_rate']) - intervention['cost']) / intervention['cost'] if intervention['cost'] > 0 else float('inf')
    }

def identify_risk_factors(data: CustomerData) -> List[str]:
    """Identify specific risk factors for the customer"""
    factors = []
    
    if data.usage_trend_30d < -0.15:
        factors.append("📉 Significant decline in usage over last 30 days")
    
    if data.avg_daily_logins < 1:
        factors.append("⚠️ Low engagement - less than 1 login per day")
    
    if data.feature_adoption_rate < 0.3:
        factors.append("🔧 Low feature adoption - using less than 30% of features")
    
    if data.active_users_pct < 0.4:
        factors.append("👥 Low seat utilization - less than 40% of seats active")
    
    if data.support_tickets_30d > 3:
        factors.append("🎫 High support ticket volume")
    
    if data.critical_issues_30d > 0:
        factors.append("🚨 Recent critical issues reported")
    
    if data.satisfaction_score < 3.5:
        factors.append("😞 Low satisfaction score")
    
    if data.payment_failures_90d > 0:
        factors.append("💳 Recent payment failures")
    
    if data.days_to_renewal < 30:
        factors.append("📅 Contract renewal approaching soon")
    
    if data.tenure_months < 3:
        factors.append("🆕 New customer - higher churn risk")
    
    if not factors:
        factors.append("✅ No major risk factors identified")
    
    return factors

@app.post("/predict", response_model=PredictionResponse)
def predict_churn(data: CustomerData, horizon: str = "30d"):
    """
    Predict churn probability for a customer
    
    Args:
        data: Customer features
        horizon: Prediction horizon ('30d', '60d', or '90d')
    
    Returns:
        Churn prediction with recommendations and financial impact
    """
    if horizon not in models:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid horizon '{horizon}'. Choose from: {list(models.keys())}"
        )
    
    if not models:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please ensure model files exist in 'models/' directory."
        )
    
    # Convert to DataFrame
    df = pd.DataFrame([data.dict()])
    
    # Get prediction
    model = models[horizon]
    churn_prob = float(model.predict_proba(df)[0, 1])
    prediction = int(model.predict(df)[0])
    
    # Risk category
    if churn_prob >= 0.7:
        risk = "High"
        priority = "Critical"
    elif churn_prob >= 0.4:
        risk = "Medium"
        priority = "Important"
    else:
        risk = "Low"
        priority = "Monitor"
    
    # Calculate financial impact
    monthly_revenue = data.monthly_revenue
    annual_revenue = monthly_revenue * 12
    revenue_at_risk = annual_revenue * churn_prob
    
    # Get intervention recommendation
    intervention = get_intervention_recommendation(churn_prob, data)
    
    # Identify risk factors
    risk_factors = identify_risk_factors(data)
    
    return {
        "customer_data": data.dict(),
        "prediction": {
            "will_churn": bool(prediction),
            "churn_probability": round(churn_prob, 3),
            "risk_level": risk,
            "priority": priority,
            "horizon": horizon,
            "confidence": "high" if abs(churn_prob - 0.5) > 0.3 else "medium"
        },
        "financial_impact": {
            "monthly_revenue": monthly_revenue,
            "annual_revenue": annual_revenue,
            "revenue_at_risk": round(revenue_at_risk, 2),
            "lifetime_value_estimate": round(annual_revenue * data.tenure_months / 12, 2)
        },
        "recommendation": intervention,
        "risk_factors": risk_factors
    }

@app.post("/batch-predict")
def batch_predict(customers: List[CustomerData], horizon: str = "30d"):
    """
    Predict churn for multiple customers at once
    
    Args:
        customers: List of customer data
        horizon: Prediction horizon
    
    Returns:
        List of predictions
    """
    results = []
    for customer in customers:
        try:
            result = predict_churn(customer, horizon)
            results.append(result)
        except Exception as e:
            results.append({"error": str(e), "customer": customer.dict()})
    
    return {
        "total_customers": len(customers),
        "successful_predictions": len([r for r in results if "error" not in r]),
        "predictions": results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
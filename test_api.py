"""
Test script for the Churn Prediction API
"""
import requests
import json

# API endpoint
BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("\n" + "="*60)
    print("Testing Health Endpoint")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_predict_high_risk():
    """Test prediction for high-risk customer"""
    print("\n" + "="*60)
    print("Testing High-Risk Customer")
    print("="*60)
    
    customer_data = {
        "monthly_revenue": 999.0,
        "avg_daily_logins": 0.5,
        "features_used": 2,
        "avg_session_duration": 10.0,
        "feature_adoption_rate": 0.2,
        "num_seats": 50,
        "active_users_pct": 0.2,
        "usage_trend_30d": -0.4,
        "usage_trend_60d": -0.5,
        "support_tickets_30d": 5,
        "avg_resolution_time": 72.0,
        "satisfaction_score": 2.0,
        "critical_issues_30d": 2,
        "payment_failures_90d": 1,
        "days_to_renewal": 15,
        "has_discount": 0,
        "discount_pct": 0.0,
        "tenure_months": 6.0,
        "company_size_score": 3,
        "plan_value": 3
    }
    
    response = requests.post(
        f"{BASE_URL}/predict?horizon=30d",
        json=customer_data
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nPrediction Results:")
        print(f"  Churn Probability: {result['prediction']['churn_probability']:.1%}")
        print(f"  Risk Level: {result['prediction']['risk_level']}")
        print(f"  Revenue at Risk: ${result['financial_impact']['revenue_at_risk']:,.2f}")
        print(f"\nRecommended Intervention:")
        print(f"  {result['recommendation']['intervention_name']}")
        print(f"  Cost: ${result['recommendation']['estimated_cost']:,.2f}")
        print(f"  Expected Success: {result['recommendation']['expected_success_rate']:.1%}")
        print(f"  ROI: {result['recommendation']['roi']:.2f}x")
        print(f"\nRisk Factors:")
        for factor in result['risk_factors']:
            print(f"  - {factor}")
    else:
        print(f"Error: {response.text}")

def test_predict_low_risk():
    """Test prediction for low-risk customer"""
    print("\n" + "="*60)
    print("Testing Low-Risk Customer")
    print("="*60)
    
    customer_data = {
        "monthly_revenue": 199.0,
        "avg_daily_logins": 5.0,
        "features_used": 8,
        "avg_session_duration": 45.0,
        "feature_adoption_rate": 0.8,
        "num_seats": 10,
        "active_users_pct": 0.9,
        "usage_trend_30d": 0.2,
        "usage_trend_60d": 0.15,
        "support_tickets_30d": 1,
        "avg_resolution_time": 12.0,
        "satisfaction_score": 4.5,
        "critical_issues_30d": 0,
        "payment_failures_90d": 0,
        "days_to_renewal": 200,
        "has_discount": 0,
        "discount_pct": 0.0,
        "tenure_months": 24.0,
        "company_size_score": 2,
        "plan_value": 2
    }
    
    response = requests.post(
        f"{BASE_URL}/predict?horizon=30d",
        json=customer_data
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nPrediction Results:")
        print(f"  Churn Probability: {result['prediction']['churn_probability']:.1%}")
        print(f"  Risk Level: {result['prediction']['risk_level']}")
        print(f"  Revenue at Risk: ${result['financial_impact']['revenue_at_risk']:,.2f}")
        print(f"\nRecommended Intervention:")
        print(f"  {result['recommendation']['intervention_name']}")
        print(f"\nRisk Factors:")
        for factor in result['risk_factors']:
            print(f"  - {factor}")
    else:
        print(f"Error: {response.text}")

def test_batch_predict():
    """Test batch prediction"""
    print("\n" + "="*60)
    print("Testing Batch Prediction (2 customers)")
    print("="*60)
    
    customers = [
        {
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
        },
        {
            "monthly_revenue": 999.0,
            "avg_daily_logins": 0.8,
            "features_used": 3,
            "avg_session_duration": 15.0,
            "feature_adoption_rate": 0.25,
            "num_seats": 50,
            "active_users_pct": 0.3,
            "usage_trend_30d": -0.3,
            "usage_trend_60d": -0.35,
            "support_tickets_30d": 4,
            "avg_resolution_time": 48.0,
            "satisfaction_score": 3.0,
            "critical_issues_30d": 1,
            "payment_failures_90d": 0,
            "days_to_renewal": 20,
            "has_discount": 1,
            "discount_pct": 0.15,
            "tenure_months": 12.0,
            "company_size_score": 3,
            "plan_value": 3
        }
    ]
    
    response = requests.post(
        f"{BASE_URL}/batch-predict?horizon=30d",
        json=customers
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nTotal Customers: {result['total_customers']}")
        print(f"Successful Predictions: {result['successful_predictions']}")
        
        for i, pred in enumerate(result['predictions'], 1):
            print(f"\nCustomer {i}:")
            print(f"  Risk: {pred['prediction']['risk_level']}")
            print(f"  Churn Probability: {pred['prediction']['churn_probability']:.1%}")
            print(f"  Revenue at Risk: ${pred['financial_impact']['revenue_at_risk']:,.2f}")
    else:
        print(f"Error: {response.text}")

def main():
    """Run all tests"""
    print("\n🚀 Starting API Tests...")
    
    try:
        test_health()
        test_predict_high_risk()
        test_predict_low_risk()
        test_batch_predict()
        
        print("\n" + "="*60)
        print("✅ All Tests Completed!")
        print("="*60)
        print("\n📖 Visit http://localhost:8000/docs for interactive API documentation")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API")
        print("Make sure the API is running with: uvicorn api.predict:app --reload")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Test NIR Inference API với dữ liệu thực
"""
import requests
import json
import sys

# API endpoint
API_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("=" * 60)
    print("1. HEALTH CHECK")
    print("=" * 60)
    response = requests.get(f"{API_URL}/health")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    print()

def test_api_info():
    """Test API info endpoint"""
    print("=" * 60)
    print("2. API INFORMATION")
    print("=" * 60)
    response = requests.get(f"{API_URL}/")
    data = response.json()
    print(f"Title: {data['title']}")
    print(f"Version: {data['version']}")
    print("\nModels:")
    for model in data['models']:
        status = "✅" if model['status'] == 'loaded' else "❌"
        print(f"  {status} {model['key']:15s} - {model['name']:20s} ({model['unit']})")
    print()

def test_predict_all(csv_file="test_nir_data.csv"):
    """Test predict all models"""
    print("=" * 60)
    print("3. PREDICT ALL MODELS")
    print("=" * 60)
    
    try:
        with open(csv_file, 'rb') as f:
            files = {'file': (csv_file, f, 'text/csv')}
            response = requests.post(f"{API_URL}/predict/all", files=files)
        
        if response.status_code == 200:
            data = response.json()
            print(f"📊 Input: {data['n_samples']} samples x {data['n_wavelengths']} wavelengths")
            print("\n🔬 Predictions:\n")
            
            # In kết quả từng model
            for model_key, result in data['results'].items():
                print(f"  📌 {result['name']} ({result['unit']})")
                print(f"     Range: {result['valid_range']['low']} - {result['valid_range']['high']}")
                print(f"     Features: {result['n_features_used']}")
                print(f"     Predictions:")
                for i, pred in enumerate(result['predictions'], 1):
                    print(f"       Sample {i}: {pred:.4f} {result['unit']}")
                print()
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
    
    except FileNotFoundError:
        print(f"❌ File not found: {csv_file}")
        print("   Tạo file test bằng lệnh:")
        print(f"   head -4 mean_by_ma_mau.csv | cut -d',' -f2-229 > {csv_file}")
        sys.exit(1)

def test_predict_single(model_key="do_am", csv_file="test_nir_data.csv"):
    """Test predict single model"""
    print("=" * 60)
    print(f"4. PREDICT SINGLE MODEL: {model_key}")
    print("=" * 60)
    
    try:
        with open(csv_file, 'rb') as f:
            files = {'file': (csv_file, f, 'text/csv')}
            response = requests.post(f"{API_URL}/predict/{model_key}", files=files)
        
        if response.status_code == 200:
            data = response.json()
            result = data['result']
            print(f"📊 Input: {data['n_samples']} samples x {data['n_wavelengths']} wavelengths")
            print(f"\n🔬 Model: {result['name']} ({result['unit']})")
            print(f"   Range: {result['valid_range']['low']} - {result['valid_range']['high']}")
            print(f"   Features: {result['n_features_used']}")
            print(f"   Predictions:")
            for i, pred in enumerate(result['predictions'], 1):
                print(f"     Sample {i}: {pred:.4f} {result['unit']}")
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
    
    except FileNotFoundError:
        print(f"❌ File not found: {csv_file}")
        sys.exit(1)

def main():
    print("\n" + "🚀 " * 20)
    print("NIR INFERENCE API - TEST SUITE")
    print("🚀 " * 20 + "\n")
    
    try:
        # Test health
        test_health()
        
        # Test API info
        test_api_info()
        
        # Test predict all
        test_predict_all()
        
        # Test predict single
        test_predict_single("do_am")
        
        print("=" * 60)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure it's running:")
        print("   docker-compose up -d")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

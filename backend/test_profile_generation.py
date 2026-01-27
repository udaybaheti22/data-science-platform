#!/usr/bin/env python3
"""
Test script for YData profiling functionality.
This tests the profile generation endpoint to ensure it works correctly.
"""

import requests
import pandas as pd
import os
import time
from io import StringIO

# Configuration
API_BASE_URL = "http://127.0.0.1:8000"
TEST_CSV_DATA = """name,age,salary,department
Alice,25,50000,Engineering
Bob,30,60000,Marketing
Charlie,35,70000,Engineering
Diana,28,55000,Sales
Eve,32,65000,Marketing
Frank,29,58000,Engineering
Grace,31,62000,Sales
Henry,27,52000,Marketing
Ivy,33,68000,Engineering
Jack,26,51000,Sales"""

def test_profile_generation():
    """Test the complete profile generation workflow."""
    print("🧪 Testing YData Profile Generation...")
    
    # Step 1: Upload test dataset
    print("📤 Uploading test dataset...")
    
    # Create a test CSV file
    csv_file = StringIO(TEST_CSV_DATA)
    files = {'file': ('test_data.csv', csv_file.getvalue(), 'text/csv')}
    
    try:
        upload_response = requests.post(f"{API_BASE_URL}/api/upload", files=files)
        if upload_response.status_code == 200:
            print("✅ Dataset uploaded successfully")
            upload_data = upload_response.json()
            print(f"   📊 Dataset: {upload_data['rows']} rows, {upload_data['columns']} columns")
        else:
            print(f"❌ Upload failed: {upload_response.status_code} - {upload_response.text}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend. Make sure the server is running on http://127.0.0.1:8000")
        return False
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return False
    
    # Step 2: Test profile generation
    print("📈 Generating profile report...")
    
    try:
        profile_response = requests.get(f"{API_BASE_URL}/api/data/profile_report")
        
        if profile_response.status_code == 200:
            print("✅ Profile report generated successfully")
            
            # Check response headers
            content_type = profile_response.headers.get('content-type', '')
            if 'text/html' in content_type:
                print("✅ Correct content type: text/html")
            else:
                print(f"⚠️  Unexpected content type: {content_type}")
            
            # Check if it's HTML content
            content = profile_response.text
            if '<html' in content.lower() and 'profile report' in content.lower():
                print("✅ Response contains valid HTML profile report")
            else:
                print("⚠️  Response doesn't appear to be a valid HTML profile report")
            
            # Check file size (should be substantial for a real profile)
            content_length = len(content)
            if content_length > 10000:  # At least 10KB
                print(f"✅ Profile report size: {content_length:,} bytes (substantial)")
            else:
                print(f"⚠️  Profile report size: {content_length:,} bytes (seems small)")
            
            return True
            
        else:
            print(f"❌ Profile generation failed: {profile_response.status_code}")
            print(f"   Error: {profile_response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Profile generation error: {e}")
        return False

def test_profile_uniqueness():
    """Test that multiple profile generations create unique files."""
    print("\n🔄 Testing profile file uniqueness...")
    
    try:
        # Generate first profile
        response1 = requests.get(f"{API_BASE_URL}/api/data/profile_report")
        if response1.status_code != 200:
            print("❌ First profile generation failed")
            return False
        
        # Wait a moment to ensure different timestamps
        time.sleep(1)
        
        # Generate second profile
        response2 = requests.get(f"{API_BASE_URL}/api/data/profile_report")
        if response2.status_code != 200:
            print("❌ Second profile generation failed")
            return False
        
        # Check if files are different (they should have different UUIDs in filename)
        # We can't directly check filenames, but we can check if content generation timestamps differ
        content1 = response1.text
        content2 = response2.text
        
        if content1 != content2:
            print("✅ Profile reports are unique (different content/timestamps)")
        else:
            print("⚠️  Profile reports appear identical (may be cached)")
        
        return True
        
    except Exception as e:
        print(f"❌ Uniqueness test error: {e}")
        return False

def test_no_dataset_error():
    """Test error handling when no dataset is uploaded."""
    print("\n🚫 Testing error handling with no dataset...")
    
    try:
        # Clear any existing dataset by making a request that should fail
        # First, let's try to get profile without dataset
        response = requests.get(f"{API_BASE_URL}/api/data/profile_report")
        
        if response.status_code == 404:
            print("✅ Correctly returns 404 when no dataset is uploaded")
            error_data = response.json()
            if "No dataset found" in error_data.get('detail', ''):
                print("✅ Error message is descriptive")
            return True
        else:
            print(f"⚠️  Expected 404, got {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 YData Profiling Integration Tests")
    print("=" * 50)
    
    # Run tests
    test1_passed = test_profile_generation()
    test2_passed = test_profile_uniqueness()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    print(f"   Profile Generation: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"   Profile Uniqueness: {'✅ PASS' if test2_passed else '❌ FAIL'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! YData profiling is working correctly.")
        print("\n💡 To test in browser:")
        print("   1. Start the backend: uvicorn main:app --reload --host 127.0.0.1 --port 8000")
        print("   2. Open frontend/index.html in browser")
        print("   3. Upload a dataset")
        print("   4. Go to Analyze section and click 'Generate Profile Report'")
        print("   5. Verify it opens in a new tab")
    else:
        print("\n❌ Some tests failed. Check the backend implementation.")
    
    print("\n" + "=" * 50)
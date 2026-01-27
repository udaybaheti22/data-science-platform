#!/usr/bin/env python3
"""
Test script for data type management functionality.
This tests the column type conversion endpoints and validation.
"""

import requests
import json
from io import StringIO

# Configuration
API_BASE_URL = "http://127.0.0.1:8000"
TEST_CSV_DATA = """name,age,salary,is_manager,join_date
Alice,25,50000.5,true,2023-01-15
Bob,30,60000.0,false,2022-06-20
Charlie,35,70000.25,true,2021-03-10
Diana,28,55000.75,false,2023-08-05"""

def upload_test_dataset():
    """Upload a test dataset for type conversion testing."""
    print("📤 Uploading test dataset...")
    
    csv_file = StringIO(TEST_CSV_DATA)
    files = {'file': ('test_data.csv', csv_file.getvalue(), 'text/csv')}
    
    try:
        response = requests.post(f"{API_BASE_URL}/api/upload", files=files)
        if response.status_code == 200:
            print("✅ Dataset uploaded successfully")
            return True
        else:
            print(f"❌ Upload failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return False

def test_get_column_types():
    """Test getting column type information."""
    print("\n📋 Testing column types retrieval...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/api/data/column_types")
        
        if response.status_code == 200:
            columns = response.json()
            print(f"✅ Retrieved {len(columns)} columns")
            
            # Check expected columns
            expected_columns = ['name', 'age', 'salary', 'is_manager', 'join_date']
            actual_columns = [col['name'] for col in columns]
            
            if set(expected_columns) == set(actual_columns):
                print("✅ All expected columns present")
            else:
                print(f"⚠️  Column mismatch. Expected: {expected_columns}, Got: {actual_columns}")
            
            # Print column info
            for col in columns:
                print(f"   📊 {col['name']}: {col['current_type']} ({col['non_null_count']} non-null)")
            
            return columns
        else:
            print(f"❌ Failed to get column types: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Error getting column types: {e}")
        return None

def test_valid_conversion():
    """Test a valid data type conversion."""
    print("\n✅ Testing valid conversion (age: int64 -> float64)...")
    
    try:
        payload = {
            "column_name": "age",
            "new_type": "float64"
        }
        
        response = requests.post(
            f"{API_BASE_URL}/api/data/change_type",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Conversion successful")
            print(f"   📊 {result['message']}")
            return True
        else:
            print(f"❌ Conversion failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Conversion error: {e}")
        return False

def test_invalid_conversion():
    """Test an invalid data type conversion."""
    print("\n❌ Testing invalid conversion (name: object -> int64)...")
    
    try:
        payload = {
            "column_name": "name",
            "new_type": "int64"
        }
        
        response = requests.post(
            f"{API_BASE_URL}/api/data/change_type",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )
        
        if response.status_code == 400:
            error = response.json()
            print("✅ Correctly rejected invalid conversion")
            print(f"   📝 Error message: {error['detail']}")
            return True
        else:
            print(f"⚠️  Expected 400 error, got {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing invalid conversion: {e}")
        return False

def test_nonexistent_column():
    """Test conversion on a non-existent column."""
    print("\n🚫 Testing non-existent column conversion...")
    
    try:
        payload = {
            "column_name": "nonexistent_column",
            "new_type": "float64"
        }
        
        response = requests.post(
            f"{API_BASE_URL}/api/data/change_type",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )
        
        if response.status_code == 404:
            error = response.json()
            print("✅ Correctly returned 404 for non-existent column")
            print(f"   📝 Error message: {error['detail']}")
            return True
        else:
            print(f"⚠️  Expected 404 error, got {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing non-existent column: {e}")
        return False

def test_same_type_conversion():
    """Test converting to the same type."""
    print("\n🔄 Testing same type conversion (salary: float64 -> float64)...")
    
    try:
        payload = {
            "column_name": "salary",
            "new_type": "float64"
        }
        
        response = requests.post(
            f"{API_BASE_URL}/api/data/change_type",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )
        
        if response.status_code == 400:
            error = response.json()
            print("✅ Correctly rejected same-type conversion")
            print(f"   📝 Error message: {error['detail']}")
            return True
        else:
            print(f"⚠️  Expected 400 error, got {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing same type conversion: {e}")
        return False

def test_boolean_conversion():
    """Test boolean conversion."""
    print("\n🔘 Testing boolean conversion (is_manager: object -> bool)...")
    
    try:
        payload = {
            "column_name": "is_manager",
            "new_type": "bool"
        }
        
        response = requests.post(
            f"{API_BASE_URL}/api/data/change_type",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Boolean conversion successful")
            print(f"   📊 {result['message']}")
            return True
        else:
            print(f"❌ Boolean conversion failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Boolean conversion error: {e}")
        return False

def verify_conversion_persistence():
    """Verify that conversions persist by checking column types again."""
    print("\n🔍 Verifying conversion persistence...")
    
    columns = test_get_column_types()
    if not columns:
        return False
    
    # Check if our conversions persisted
    age_col = next((col for col in columns if col['name'] == 'age'), None)
    is_manager_col = next((col for col in columns if col['name'] == 'is_manager'), None)
    
    success = True
    
    if age_col and age_col['current_type'] == 'float64':
        print("✅ Age column conversion persisted (int64 -> float64)")
    else:
        print(f"❌ Age column conversion not persisted. Current type: {age_col['current_type'] if age_col else 'Not found'}")
        success = False
    
    if is_manager_col and is_manager_col['current_type'] == 'bool':
        print("✅ Is_manager column conversion persisted (object -> bool)")
    else:
        print(f"❌ Is_manager column conversion not persisted. Current type: {is_manager_col['current_type'] if is_manager_col else 'Not found'}")
        success = False
    
    return success

if __name__ == "__main__":
    print("🧪 Data Type Management Integration Tests")
    print("=" * 60)
    
    # Check backend connection
    try:
        response = requests.get(f"{API_BASE_URL}/api/data/column_types")
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend. Make sure the server is running on http://127.0.0.1:8000")
        exit(1)
    
    # Run tests
    tests = [
        ("Dataset Upload", upload_test_dataset),
        ("Column Types Retrieval", test_get_column_types),
        ("Valid Conversion", test_valid_conversion),
        ("Invalid Conversion", test_invalid_conversion),
        ("Non-existent Column", test_nonexistent_column),
        ("Same Type Conversion", test_same_type_conversion),
        ("Boolean Conversion", test_boolean_conversion),
        ("Conversion Persistence", verify_conversion_persistence),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            if test_name == "Column Types Retrieval":
                # This test returns data, not just boolean
                result = test_func()
                results.append((test_name, result is not None))
            else:
                result = test_func()
                results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Test Summary:")
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\n📊 Results: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Data type management is working correctly.")
        print("\n💡 To test in browser:")
        print("   1. Start the backend: uvicorn main:app --reload --host 127.0.0.1 --port 8000")
        print("   2. Open frontend/index.html in browser")
        print("   3. Upload a dataset")
        print("   4. Go to Data section")
        print("   5. Try changing column types using the dropdowns and Save buttons")
        print("   6. Verify inline error messages for invalid conversions")
    else:
        print(f"\n❌ {len(results) - passed} tests failed. Check the implementation.")
    
    print("\n" + "=" * 60)
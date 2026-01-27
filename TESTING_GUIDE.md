# Testing Guide for UI Functionality Refinement

## Quick Start

### 1. Start the Backend
```bash
cd backend
python -m venv venv
venv\Scripts\activate  # On Windows
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### 2. Test Individual Components

#### Backend API Tests
```bash
cd backend
python test_profile_generation.py
python test_data_types.py
```

#### Frontend Component Tests
Open these files in your browser (with backend running):

1. **Complete Workflow Test**: `frontend/complete-test.html`
   - Tests upload → data types → profile generation
   - Most comprehensive test

2. **Data Type Management**: `frontend/data-test.html`
   - Tests column type conversion
   - Tests inline error handling

3. **Profile Generation**: `frontend/profile-test.html`
   - Tests YData profiling
   - Tests new tab opening

4. **Main Application**: `frontend/index.html`
   - The actual application
   - Should work with all implemented features

## What Should Work Now

### ✅ Fixed Issues

1. **Generate Profile Button**: 
   - Now works and opens reports in new browser tabs
   - Uses `explorative=True` for comprehensive reports
   - Generates unique filenames

2. **Data Type Management UI**:
   - Complete table with Column Name, Current Type, Target Type, Action
   - Per-row Save buttons that enable/disable based on changes
   - Inline error messages for failed conversions
   - Immediate UI updates on successful conversions

3. **Section Loading**:
   - Upload now preloads data for Clean and Data sections
   - Sections are ready when you navigate to them
   - No more empty sections after upload

### 🔧 Implementation Details

#### Backend Enhancements
- Enhanced profile endpoint with unique filenames and `explorative=True`
- Improved data type conversion with pre-validation
- Better error messages for invalid conversions
- Atomic operations to prevent partial dataset corruption

#### Frontend Enhancements
- Added debugging logs to track event wiring
- Enhanced data type table with inline error containers
- Improved upload handler to preload section data
- Per-row error handling without global alerts

## Testing Workflow

### 1. Test Backend APIs
```bash
# In backend directory
python test_profile_generation.py
python test_data_types.py
```
Both should show "All tests passed!"

### 2. Test Frontend Components
1. Open `frontend/complete-test.html` in browser
2. Click "1. Upload Dataset" - should succeed
3. Click "2. Test Data Types" - should convert age column
4. Click "3. Generate Profile" - should open new tab with report

### 3. Test Main Application
1. Open `frontend/index.html` in browser
2. Upload a CSV file in Preview section
3. Navigate to Data section - should show column type table
4. Try changing a column type - should work with inline feedback
5. Navigate to Analyze section
6. Click "Generate Profile Report" - should open in new tab
7. Navigate to Clean section - should show missing values

## Troubleshooting

### Backend Issues
- **Connection refused**: Make sure backend is running on port 8000
- **Import errors**: Check if all packages are installed (`pip install -r requirements.txt`)
- **Profile generation fails**: Check if `reports/` directory exists (created automatically)

### Frontend Issues
- **Buttons not working**: Check browser console for JavaScript errors
- **Modules not loading**: Make sure you're serving files via HTTP (not file://)
- **CORS errors**: Backend should handle CORS, but check console

### Browser Console Debugging
The frontend now includes extensive logging:
- `🚀 DOM Content Loaded` - App initialization
- `🔌 Wiring events` - Event handler setup
- `🖱️ Sidebar clicked` - Navigation events
- `📊 Loading Data section` - Section loading
- `📈 Profile generation` - Profile button clicks

## Expected Behavior

### Data Type Management
1. **Table Structure**: 4 columns (Name, Current Type, Target Type, Action)
2. **Save Button**: Disabled by default, blue when enabled
3. **Dropdown Changes**: Enable save button when different from current type
4. **Successful Conversion**: Updates current type immediately, disables save button
5. **Failed Conversion**: Shows red error message inline, keeps button enabled
6. **Error Isolation**: Other rows remain functional during errors

### Profile Generation
1. **Button Click**: Should show "Opening profile report in new tab..."
2. **New Tab**: Should open with comprehensive HTML report
3. **Report Content**: Should be substantial (500KB+) with interactive charts
4. **No Downloads**: Report opens in browser, user saves manually if desired

### Upload Workflow
1. **File Upload**: Shows success message with row/column count
2. **Section Preloading**: Automatically loads data for Clean and Data sections
3. **Navigation Ready**: All sections should work immediately after upload

## Files Created/Modified

### New Test Files
- `backend/test_profile_generation.py` - Backend API tests
- `backend/test_data_types.py` - Data type conversion tests
- `frontend/complete-test.html` - Complete workflow test
- `frontend/data-test.html` - Data type UI test
- `frontend/profile-test.html` - Profile generation test

### Modified Files
- `backend/main.py` - Enhanced profile and data type endpoints
- `frontend/static/js/events.js` - Added debugging and preloading
- `frontend/static/js/render.js` - Enhanced data type table rendering
- `README.md` - Updated documentation

All changes maintain the existing file structure and preserve existing functionality while adding the requested features.
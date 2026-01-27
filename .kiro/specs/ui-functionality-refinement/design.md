# Design Document

## Overview

This design implements three focused functionality changes to the existing FastAPI + Vanilla JS data science platform: removing redundant automated profiling, consolidating YData profiling in the Analyze section with Jupyter-like behavior, and implementing granular per-row data type management. The design maintains the existing UI structure while improving user experience through elimination of confusion and better error handling.

## Architecture

The solution maintains the current three-tier architecture:
- **Frontend**: Vanilla JavaScript with modular file structure (api.js, events.js, render.js, ui.js)
- **Backend**: FastAPI with in-memory data storage using pandas DataFrames
- **Data Layer**: Pandas DataFrames stored in global data_store dictionary

### Key Architectural Decisions

1. **No structural changes**: Preserve existing file organization and module boundaries
2. **Incremental modifications**: Update existing endpoints rather than creating new architecture
3. **Error isolation**: Implement row-level error handling to prevent cascading failures
4. **Browser-native behavior**: Use standard browser mechanisms for report viewing and downloading

## Components and Interfaces

### Backend Components

#### Modified Endpoints

**Profile Report Endpoint** (`/api/data/profile_report`)
- **Input**: GET request (no parameters)
- **Processing**: 
  - Generate ProfileReport(df, explorative=True)
  - Save to unique HTML file path
  - Return FileResponse with media_type="text/html"
- **Output**: HTML file served directly to browser

**Data Type Change Endpoint** (`/api/data/change_type`)
- **Input**: JSON with column_name and new_type
- **Processing**:
  - Validate conversion feasibility
  - Attempt single column conversion
  - Update dataframe or return specific error
- **Output**: Success confirmation or detailed error message

#### Removed Components
- Any automated profile endpoints separate from `/api/data/profile_report`
- Batch conversion logic (if any exists)

### Frontend Components

#### Modified Event Handlers (events.js)

**Profile Generation Handler**
```javascript
// Replace existing profile generation logic
function wireAnalyzeSection() {
  const btn = document.getElementById("generate-analyze-report");
  btn.addEventListener("click", () => {
    window.open(PROFILE_REPORT_URL, '_blank');
  });
}
```

**Data Type Management Handler**
```javascript
// Enhanced per-row save functionality
container.addEventListener("click", async (e) => {
  if (e.target.dataset.action === "save-type") {
    // Row-specific error handling
    // Individual API calls per column
    // Immediate UI updates on success
  }
});
```

#### Modified Rendering Logic (render.js)

**Data Type Table Renderer**
- Enhanced with per-row save buttons
- Dynamic enable/disable based on dropdown changes
- Inline error message containers for each row

#### Removed Components
- All automated profile navigation items
- All automated profile render functions
- All automated profile event handlers

## Data Models

### Existing Data Models (Unchanged)
- `data_store` dictionary structure remains identical
- Pandas DataFrame storage mechanism unchanged
- Column metadata structure preserved

### API Request/Response Models

**Data Type Change Request**
```json
{
  "column_name": "string",
  "new_type": "string"  // One of: int64, float64, object, bool, datetime64[ns]
}
```

**Data Type Change Response (Success)**
```json
{
  "message": "Successfully changed type of 'column_name' to 'new_type'",
  "rows": 1000,
  "columns": 15,
  "history_length": 3
}
```

**Data Type Change Response (Error)**
```json
{
  "detail": "Invalid conversion: cannot convert string 'abc' to float64"
}
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property Analysis

Before defining properties, I need to analyze which acceptance criteria are testable using the prework tool:

### Correctness Properties

Based on the prework analysis and property reflection, the following properties ensure system correctness:

**Property 1: Automated Profile Elimination**
*For any* navigation state in the application, the system should not display any automated profile functionality outside the Analyze section
**Validates: Requirements 1.2**

**Property 2: Profile File Generation Uniqueness**
*For any* profile generation request, the system should create an HTML file with a unique identifier that doesn't conflict with existing files
**Validates: Requirements 2.2**

**Property 3: Data Type Table Structure**
*For any* dataset loaded in the Data section, the table should display all required columns (Column Name, Current Data Type, Target Data Type, Action) with correct initial states
**Validates: Requirements 3.2, 3.3, 3.4, 3.5**

**Property 4: Save Button State Management**
*For any* dropdown change in the data type table, the Save button should be enabled if and only if the target type differs from the current type
**Validates: Requirements 3.6, 3.7**

**Property 5: Single Column API Calls**
*For any* save button click, exactly one API request should be made for that specific column conversion
**Validates: Requirements 3.8**

**Property 6: UI Update Consistency**
*For any* successful data type conversion, the Current Data Type text should be updated immediately to reflect the new type
**Validates: Requirements 3.9**

**Property 7: Error Isolation**
*For any* failed data type conversion, the error should be displayed inline for that specific row only, without affecting other rows or global UI state
**Validates: Requirements 3.10, 3.11, 3.12**

**Property 8: Backend Validation Integrity**
*For any* invalid conversion request, the backend should reject it with a clear error message and leave the dataset completely unchanged
**Validates: Requirements 3.13, 3.14, 3.17**

## Error Handling

### Row-Level Error Isolation
- Each data type conversion operates independently
- Conversion failures are contained to the specific row
- Error messages appear inline next to the failed conversion
- Other rows remain fully functional during error states

### Backend Validation Strategy
- Pre-validate conversion feasibility before attempting changes
- Use pandas dtype conversion with proper exception handling
- Maintain dataset integrity through atomic operations
- Return descriptive error messages for user guidance

### Frontend Error Display
- Inline error messages within table rows
- No global error alerts for conversion failures
- Clear visual distinction between error and success states
- Automatic error clearing on successful retry

## Testing Strategy

### Dual Testing Approach
The implementation will use both unit tests and property-based tests to ensure comprehensive coverage:

**Unit Tests** focus on:
- Specific examples of successful conversions
- Edge cases like empty datasets or invalid column names
- Integration points between frontend and backend
- Error conditions and boundary cases

**Property-Based Tests** focus on:
- Universal properties that hold across all inputs
- Comprehensive input coverage through randomization
- Validation of correctness properties across many scenarios

### Property-Based Testing Configuration
- Use **pytest** with **hypothesis** library for Python backend testing
- Use **Jest** with **fast-check** library for JavaScript frontend testing
- Configure minimum 100 iterations per property test
- Tag each test with format: **Feature: ui-functionality-refinement, Property {number}: {property_text}**

### Testing Framework Requirements
- Backend: pytest + hypothesis for property-based testing
- Frontend: Jest + fast-check for JavaScript property testing
- Each correctness property must be implemented by a single property-based test
- Unit tests complement property tests for specific examples and edge cases

### Test Coverage Areas
1. **Automated Profile Removal**: Verify no automated profile elements exist in any navigation state
2. **Profile Generation**: Test file creation, uniqueness, and browser integration
3. **Data Type Management**: Test UI rendering, state management, and API integration
4. **Error Handling**: Test error isolation, message display, and recovery scenarios
5. **API Validation**: Test backend validation logic and data integrity preservation
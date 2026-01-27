# Implementation Plan: UI Functionality Refinement

## Overview

This implementation plan converts the approved design into discrete coding tasks for refining the FastAPI + Vanilla JS data science platform. The tasks focus on removing redundant automated profiling, consolidating YData profiling with Jupyter-like behavior, and implementing granular per-row data type management with proper error handling.

## Tasks

- [x] 1. Remove Automated Profile Section
  - Remove any automated profile navigation items from frontend/index.html
  - Remove automated profile event handlers from frontend/static/js/events.js
  - Remove automated profile render functions from frontend/static/js/render.js
  - Remove any automated profile API endpoints from backend/main.py (if they exist separately from /api/data/profile_report)
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [ ] 2. Enhance YData Profiling in Analyze Section
  - [x] 2.1 Update backend profile report endpoint
    - Modify /api/data/profile_report to use ProfileReport(df, explorative=True)
    - Ensure report is saved with unique filename (e.g., profile_<uuid>.html)
    - Return FileResponse with media_type="text/html"
    - _Requirements: 2.1, 2.2, 2.3_

  - [x] 2.2 Update frontend profile generation
    - Modify wireAnalyzeSection() in events.js to use window.open(PROFILE_REPORT_URL, '_blank')
    - Remove any iframe embedding logic
    - Remove any automatic download forcing
    - Ensure no PDF conversion or JSON response handling
    - _Requirements: 2.4, 2.7, 2.8, 2.9, 2.10_

  - [ ] 2.3 Write property test for profile file generation
    - **Property 2: Profile File Generation Uniqueness**
    - **Validates: Requirements 2.2**

  - [ ] 2.4 Write unit tests for profile generation behavior
    - Test ProfileReport is called with explorative=True
    - Test FileResponse returns correct media type
    - Test window.open is called with correct parameters
    - _Requirements: 2.1, 2.3, 2.4_

- [ ] 3. Implement Per-Row Data Type Management
  - [x] 3.1 Enhance data type table rendering
    - Update renderDatatypeTable() in render.js to include proper table structure
    - Add Column Name, Current Data Type, Target Data Type, and Action columns
    - Implement per-row Save buttons that are disabled by default
    - Add dropdown change handlers to enable/disable Save buttons
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7_

  - [x] 3.2 Implement per-row save functionality
    - Update wireDataSection() in events.js for per-row save handling
    - Implement single API call per save button click
    - Add immediate UI updates on successful conversion
    - Add inline error message display for failed conversions
    - Ensure error isolation (no global alerts)
    - _Requirements: 3.8, 3.9, 3.10, 3.11, 3.12_

  - [x] 3.3 Enhance backend data type conversion
    - Update /api/data/change_type endpoint for better validation
    - Add pre-conversion validation to prevent invalid conversions
    - Implement atomic operations to prevent partial dataset mutation
    - Return clear error messages for invalid conversions
    - Ensure no silent failures
    - _Requirements: 3.13, 3.14, 3.15, 3.16, 3.17_

- [ ]* 3.4 Write property tests for data type management
  - **Property 3: Data Type Table Structure**
  - **Property 4: Save Button State Management**
  - **Property 5: Single Column API Calls**
  - **Property 6: UI Update Consistency**
  - **Property 7: Error Isolation**
  - **Property 8: Backend Validation Integrity**
  - **Validates: Requirements 3.2-3.17**

- [ ]* 3.5 Write unit tests for data type conversion
  - Test successful conversion scenarios
  - Test invalid conversion error handling
  - Test UI state management
  - Test error message display
  - _Requirements: 3.8, 3.9, 3.10, 3.13_

- [ ] 4. Integration and Testing
  - [x] 4.1 Verify all automated profile functionality is removed
    - Test navigation through all sections
    - Verify no automated profile elements exist
    - Test that only one profiling feature remains (in Analyze section)
    - _Requirements: 1.1, 1.2, 1.4_

  - [x] 4.2 Test YData profiling integration
    - Test profile generation opens in new browser tab
    - Verify HTML report is generated correctly
    - Test that no automatic downloads occur
    - _Requirements: 2.1, 2.2, 2.4, 2.10_

  - [x] 4.3 Test data type management end-to-end
    - Test per-row conversion functionality
    - Verify error isolation works correctly
    - Test UI updates after successful conversions
    - Test inline error messages for failures
    - _Requirements: 3.6, 3.8, 3.9, 3.10, 3.11_

- [ ]* 4.4 Write integration tests
  - Test complete user workflows
  - Test error recovery scenarios
  - Test cross-component interactions
  - _Requirements: All requirements_

- [x] 5. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- YData profiling tests (2.3, 2.4) are required as requested by user
- Each task references specific requirements for traceability
- Focus on maintaining existing file structure and module boundaries
- Preserve all existing functionality not related to the three changes
- User will handle most testing themselves except for YData profiling functionality
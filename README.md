# Data Science Platform

A comprehensive data science platform that allows users to upload, clean, analyze, and export datasets through an intuitive web interface.

## Features

- **Dataset Preview**: View uploaded datasets in a clean table format
- **Data Cleaning**: Remove columns, handle missing values, remove duplicates, encode categorical data
- **Categorical Encoding**: One-hot and label encoding for categorical columns
- **Column Type Management**: Change data types of columns interactively
- **Data Analysis**: View descriptive statistics and column information
- **Model Building**: Prepare for machine learning model development
- **Export**: Download processed datasets as CSV files
- **Project Management**: Save progress and work on multiple projects

## Project Structure

```
ml-project/
├── backend/
│   ├── main.py              # FastAPI backend server
│   ├── requirements.txt     # Python dependencies
│   └── venv/               # Virtual environment
├── frontend/
│   └── index.html          # React-based frontend
└── README.md               # This file
```

## Setup Instructions

### Backend Setup

1. **Navigate to the backend directory:**
   ```bash
   cd backend
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Start the backend server:**
   ```bash
   python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
   ```

The backend will be available at `http://127.0.0.1:8000`

### Frontend Setup

1. **Open the frontend file:**
   - Navigate to the `frontend` directory
   - Open `index.html` in your web browser
   - Or use a local server like Live Server in VS Code

2. **Access the application:**
   - Open `http://127.0.0.1:5500/frontend/index.html` (if using Live Server)
   - Or simply open the `index.html` file directly in your browser

## How to Use

### 1. Getting Started

1. **Open the application** in your web browser
2. **Click "Let's Begin"** to start
3. **Create a new project** or select an existing one

### 2. Uploading Data

1. **Navigate to a project**
2. **Upload a CSV file** using the file upload interface
3. **View the dataset preview** in the main area

### 3. Data Cleaning & Encoding

1. **Click "Clean"** in the sidebar
2. **Drop Columns**: Select columns to remove (optional)
3. **Handle Missing Values**: Choose a method for each column (drop, fill with mean/median/mode)
4. **Remove Duplicates**: Use the "Duplicate Data" card to see duplicate count, remove all or a specific number
5. **Categorical Encoding**: Use the "Categorical Encoding" card to select categorical columns and apply one-hot or label encoding
   - **One-Hot Encoding**: Creates binary columns for each unique value
   - **Label Encoding**: Converts categories to numeric codes (warning: best for ordinal data)
6. **Undo**: Use the Undo button to revert the last cleaning/encoding action
7. **Click "Save Progress"** to save your changes

### 4. Data Type Management

1. **Click "Data"** in the sidebar
2. **View and change column types** using the dropdown menus in the table
3. **Handle errors**: If a conversion fails, an error message will be shown
4. **Click "Save Progress"** to save your changes

### 5. Data Analysis

1. **Click "Analyze"** in the sidebar
2. **View column information** and data types
3. **Review descriptive statistics** for numerical columns
4. **Click "Save Progress"** to save your work

### 6. Model Building

1. **Click "Build Model"** in the sidebar
2. **Prepare for machine learning** (coming soon)
3. **Click "Save Progress"** to save your work

### 7. Export Data

1. **Click "Export"** in the sidebar
2. **Click "Download CSV"** to download the processed dataset
3. **The file will be saved** to your downloads folder

## API Endpoints

The backend provides the following API endpoints:

- `POST /api/upload` - Upload CSV dataset
- `GET /api/data/preview` - Get dataset preview
- `POST /api/data/clean` - Clean dataset, remove duplicates, encode categorical columns
- `GET /api/data/profile` - Get data profile and statistics
- `POST /api/data/change_type` - Change column data type
- `GET /api/data/duplicates_summary` - Get duplicate row count
- `GET /api/data/export` - Export dataset as CSV
- `POST /api/project/save` - Save project state
- `GET /api/project/{project_id}` - Get project state

### Example: Categorical Encoding API Usage

**One-Hot Encoding**
```json
{
  "operations": [
    { "type": "one_hot_encode", "columns": ["City", "Name"] }
  ]
}
```

**Label Encoding**
```json
{
  "operations": [
    { "type": "label_encode", "columns": ["City"] }
  ]
}
```

## Features in Detail

### Dataset Preview
- Displays uploaded data in a clean, scrollable table
- Shows row and column counts
- Responsive design for different screen sizes

### Data Cleaning & Encoding
- **Column Removal**: Select and remove unwanted columns
- **Missing Value Handling**: Multiple strategies for handling missing data
- **Duplicate Removal**: Remove all or a specific number of duplicate rows
- **Categorical Encoding**: One-hot and label encoding for categorical columns
- **Real-time Processing**: Apply changes immediately
- **Undo**: Revert the last cleaning/encoding action

### Data Type Management
- **Change Types**: Change column types interactively (object, int64, float64, datetime64[ns])
- **Error Handling**: Robust error messages for invalid conversions

### Data Analysis
- **Column Information**: Data types and non-null counts
- **Descriptive Statistics**: Mean, median, standard deviation, etc.
- **Value Counts**: For categorical variables

### Project Management
- **Save Progress**: Save current state at any stage
- **Multiple Projects**: Work on different datasets simultaneously
- **Persistent Storage**: Progress is saved locally

### Export Functionality
- **CSV Download**: Export processed datasets
- **Automatic Naming**: Files are named appropriately
- **Browser Integration**: Uses native download functionality

## Technical Details

### Backend (FastAPI)
- **FastAPI**: Modern Python web framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **scikit-learn**: Label encoding for categorical columns
- **CORS**: Cross-origin resource sharing enabled

### Frontend (React)
- **React 18**: Modern JavaScript framework
- **Tailwind CSS**: Utility-first CSS framework
- **Local Storage**: Client-side data persistence
- **Responsive Design**: Works on desktop and mobile

## Troubleshooting

### Backend Issues
- **Port already in use**: Change the port in the uvicorn command
- **Missing dependencies**: Ensure all requirements are installed
- **CORS errors**: Check that the frontend origin is allowed

### Frontend Issues
- **API connection errors**: Ensure the backend is running
- **File upload issues**: Check file format (CSV only)
- **Display issues**: Try refreshing the page

### Data Issues
- **Large files**: Consider reducing file size for better performance
- **Encoding issues**: Ensure CSV files use UTF-8 encoding
- **Missing data**: Check that the CSV format is correct

## Future Enhancements

- **Machine Learning Models**: Implement actual ML model building
- **Data Visualization**: Add charts and graphs
- **Advanced Cleaning**: More sophisticated data cleaning options
- **User Authentication**: Multi-user support
- **Database Storage**: Persistent project storage
- **Real-time Collaboration**: Multiple users working on the same project
- **More Encoding Methods**: Target/frequency encoding, encoding visualization, and recommendations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License. 

# MediPredictor Data

This directory contains data for the MediPredictor application.

## Structure

- `medical_data.ts`: TypeScript file with structured disease and symptoms data for the frontend
- `eda/`: Python scripts for Exploratory Data Analysis on medical data
  - `medical_data_analysis.py`: Main Python module for data analysis
  - `requirements.txt`: Python dependencies
  - `example_data.csv`: Sample CSV data for testing
  - `README.md`: Documentation for the Python EDA module

## Data Flow

1. Raw CSV data is processed through the Python EDA module
2. The module cleans, analyzes, and visualizes the data
3. Processed data is exported to a JSON format
4. The frontend imports this processed data for disease prediction

## Running the EDA

To perform exploratory data analysis on your CSV data:

1. Navigate to the `eda` directory
2. Install requirements: `pip install -r requirements.txt`
3. Run the analysis: `python medical_data_analysis.py`

This will generate visualizations and a processed JSON file in an `analysis_output` directory.

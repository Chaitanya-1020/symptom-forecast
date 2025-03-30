
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
3. Run the analysis: `python medical_data_analysis.py path/to/your/data.csv`

This will generate visualizations and processed JSON files in an `analysis_output` directory.

## Generated Visualizations

The EDA process generates several visualizations:

- **Disease Prevalence**: Bar chart showing the prevalence of each disease
- **Symptom Frequency**: Bar chart showing the most common symptoms
- **Symptom Correlation**: Heatmap showing correlations between symptoms
- **Disease-Symptom Network**: Scatter plot showing relationship between symptom count and disease prevalence
- **Symptom PCA**: Principal Component Analysis visualization for disease clustering based on symptoms

## Using Analysis Results in the Frontend

After running the EDA, you can use the generated files in your frontend by:

1. Moving the `medical_data.json` file to the appropriate location
2. Importing the data in your React components
3. Using the data for symptom selection, disease prediction, and visualization

## Example Analysis Output

```json
{
  "statistics": {
    "disease_count": 12,
    "unique_symptoms": 45,
    "top_symptoms": ["Cough", "Fever", "Fatigue", "Headache", "Nausea"],
    "mean_prevalence": 47.5,
    "min_prevalence": 25,
    "max_prevalence": 78
  },
  "symptom_frequencies": {
    "Cough": 4,
    "Fever": 3,
    "Headache": 3,
    "Fatigue": 3,
    "Nausea": 2
  }
}
```

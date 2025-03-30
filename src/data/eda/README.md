
# Medical Data EDA (Exploratory Data Analysis)

This Python module performs exploratory data analysis on medical symptom-disease data from CSV files.

## Features

- Data cleaning (handling missing values, duplicates, formatting)
- Statistical analysis of disease-symptom relationships
- Visualization generation:
  - Disease prevalence chart
  - Symptom frequency analysis
  - Symptom correlation heatmap
- Export to JSON for frontend consumption

## Requirements

Install dependencies using:

```
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from medical_data_analysis import MedicalDataAnalyzer

# Initialize with CSV file
analyzer = MedicalDataAnalyzer(csv_path='path/to/your/medical_data.csv')

# Or use the default test data
analyzer = MedicalDataAnalyzer()

# Run full analysis pipeline
results = analyzer.run_full_analysis(output_dir="analysis_output")
```

### Individual Analysis Steps

```python
analyzer = MedicalDataAnalyzer(csv_path='path/to/your/medical_data.csv')

# Clean the data
analyzer.clean_data()

# View basic statistics
stats = analyzer.explore_data()

# Generate specific visualizations
analyzer.disease_prevalence_chart()
analyzer.symptom_frequency_chart()
analyzer.symptom_correlation_heatmap()

# Export processed data for frontend
analyzer.export_to_json("processed_data.json")
```

## Data Format

The expected CSV format should have at minimum these columns:
- `disease`: The name of the disease/condition
- `symptoms`: Comma-separated list of symptoms
- `prevalence`: A numeric value (0-100) indicating disease prevalence

## Integration with Frontend

The JSON output from this analysis can be directly consumed by the MediPredictor React frontend. To integrate:

1. Run the full analysis
2. Copy the generated JSON file to the frontend data directory
3. Update the frontend import path to use the new data source

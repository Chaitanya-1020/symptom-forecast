
# Medical Data EDA (Exploratory Data Analysis)

This Python module performs exploratory data analysis on medical symptom-disease data from CSV files.

## Features

- Data cleaning (handling missing values, duplicates, formatting)
- Statistical analysis of disease-symptom relationships
- Advanced visualization generation:
  - Disease prevalence chart
  - Symptom frequency analysis
  - Symptom correlation heatmap
  - Disease-symptom network visualization
  - PCA-based disease clustering visualization
- Export to JSON for frontend consumption
- Comprehensive EDA metrics and insights

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

### Command Line Usage

```bash
python medical_data_analysis.py path/to/your/data.csv output_directory
```

If no arguments are provided, it will look for example_data.csv in the same directory or use default data.

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
analyzer.disease_symptom_network()
analyzer.symptom_pca_visualization()

# Export processed data for frontend
analyzer.export_to_json("processed_data.json")

# Export EDA results for frontend
analyzer.export_eda_results("eda_results.json")
```

## Data Format

The expected CSV format should have at minimum these columns:
- `disease`: The name of the disease/condition
- `symptoms`: Comma-separated list of symptoms
- `prevalence`: A numeric value (0-100) indicating disease prevalence

Example:
```csv
disease,symptoms,prevalence
Common Cold,"Cough,Runny Nose,Sore Throat,Sneezing",78
Influenza,"High Fever,Cough,Fatigue,Body Aches",65
```

## Output Files

The analysis generates several output files:
- **disease_prevalence.png**: Bar chart of disease prevalence
- **symptom_frequency.png**: Bar chart of symptom frequencies
- **symptom_correlation.png**: Heatmap of symptom correlations
- **disease_symptom_network.png**: Scatter plot of symptom count vs. prevalence
- **symptom_pca.png**: PCA visualization of diseases based on symptoms
- **statistics.json**: Basic statistical metrics
- **medical_data.json**: Processed data for frontend consumption
- **eda_results.json**: Complete EDA metrics and visualization metadata

## Integration with Frontend

The JSON output from this analysis can be directly consumed by the MediPredictor React frontend. To integrate:

1. Run the full analysis
2. Copy the generated JSON file to the frontend data directory
3. Update the frontend import path to use the new data source

## Analyzing Your Own Data

To analyze your own medical data:
1. Format your CSV file according to the expected format
2. Run the analysis script with your CSV as input
3. Review the generated visualizations and metrics
4. Use the exported JSON files in your application

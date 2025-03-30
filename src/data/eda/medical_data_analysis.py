
# Medical Data EDA (Exploratory Data Analysis)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

class MedicalDataAnalyzer:
    """
    Performs Exploratory Data Analysis on medical symptom-disease data
    """
    def __init__(self, csv_path=None, df=None):
        """Initialize with either a CSV path or an existing DataFrame"""
        if df is not None:
            self.df = df
        elif csv_path:
            self.df = self.load_data(csv_path)
        else:
            # Use default data if nothing provided
            # This would be replaced by loading actual CSV in production
            self.create_default_dataframe()

    def load_data(self, csv_path):
        """Load data from CSV file"""
        try:
            return pd.read_csv(csv_path)
        except Exception as e:
            print(f"Error loading CSV: {e}")
            self.create_default_dataframe()
            return self.df

    def create_default_dataframe(self):
        """Create a default DataFrame structure matching our expected medical data"""
        # This represents the format we expect from the CSV
        diseases = [
            "Common Cold", "Influenza", "COVID-19", "Allergic Rhinitis", 
            "Sinusitis", "Bronchitis", "Asthma", "Pneumonia", 
            "Gastroenteritis", "Migraine", "Urinary Tract Infection", "Food Poisoning"
        ]
        
        # Create a multi-symptom format (comma-separated in CSV)
        symptoms_list = [
            "Cough,Runny Nose,Sore Throat,Sneezing,Headache,Fatigue",
            "High Fever,Cough,Fatigue,Body Aches,Headache,Chills",
            "Fever,Dry Cough,Fatigue,Loss of Taste/Smell,Shortness of Breath,Body Aches",
            "Sneezing,Runny Nose,Itchy Eyes,Nasal Congestion,Watery Eyes",
            "Facial Pain,Nasal Congestion,Headache,Thick Nasal Discharge,Reduced Sense of Smell",
            "Persistent Cough,Mucus Production,Fatigue,Shortness of Breath,Chest Discomfort",
            "Wheezing,Shortness of Breath,Chest Tightness,Coughing,Difficulty Breathing",
            "High Fever,Cough with Phlegm,Shortness of Breath,Chest Pain,Fatigue,Confusion",
            "Nausea,Vomiting,Diarrhea,Stomach Cramps,Fever,Headache",
            "Severe Headache,Nausea,Light Sensitivity,Sound Sensitivity,Visual Disturbances",
            "Frequent Urination,Burning Sensation,Cloudy Urine,Strong Odor,Pelvic Pain",
            "Nausea,Vomiting,Diarrhea,Abdominal Pain,Fever,Dehydration"
        ]
        
        prevalence = [78, 65, 60, 45, 40, 35, 32, 25, 55, 42, 38, 50]
        
        # Create DataFrame
        self.df = pd.DataFrame({
            'disease': diseases,
            'symptoms': symptoms_list,
            'prevalence': prevalence
        })
    
    def clean_data(self):
        """Clean the data - handling missing values, duplicates, etc."""
        # Handle missing values
        self.df = self.df.dropna(subset=['disease', 'symptoms'])
        
        # Remove duplicates
        self.df = self.df.drop_duplicates()
        
        # Ensure prevalence is numeric
        self.df['prevalence'] = pd.to_numeric(self.df['prevalence'], errors='coerce')
        self.df['prevalence'] = self.df['prevalence'].fillna(0)
        
        # Normalize prevalence to 0-100 if needed
        if self.df['prevalence'].max() > 100:
            self.df['prevalence'] = 100 * self.df['prevalence'] / self.df['prevalence'].max()
        
        return self.df
    
    def explore_data(self):
        """Print basic statistics and information about the data"""
        print("Data Overview:")
        print(f"Number of diseases: {len(self.df)}")
        
        # Extract all symptoms
        all_symptoms = []
        for symptom_list in self.df['symptoms']:
            if isinstance(symptom_list, str):
                symptoms = symptom_list.split(',')
                all_symptoms.extend(symptoms)
        
        unique_symptoms = set(all_symptoms)
        print(f"Number of unique symptoms: {len(unique_symptoms)}")
        print(f"Top 10 symptoms: {list(unique_symptoms)[:10]}")
        
        # Prevalence statistics
        print("\nPrevalence Statistics:")
        print(f"Mean prevalence: {self.df['prevalence'].mean():.2f}")
        print(f"Min prevalence: {self.df['prevalence'].min()}")
        print(f"Max prevalence: {self.df['prevalence'].max()}")
        
        return {
            "disease_count": len(self.df),
            "unique_symptoms": len(unique_symptoms),
            "top_symptoms": list(unique_symptoms)[:10],
            "mean_prevalence": float(self.df['prevalence'].mean()),
            "min_prevalence": float(self.df['prevalence'].min()),
            "max_prevalence": float(self.df['prevalence'].max()),
        }
    
    def disease_prevalence_chart(self, output_path="disease_prevalence.png"):
        """Generate a bar chart of disease prevalence"""
        plt.figure(figsize=(12, 8))
        chart = sns.barplot(x='disease', y='prevalence', data=self.df)
        chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
        plt.title('Disease Prevalence')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        return output_path
    
    def symptom_frequency_chart(self, output_path="symptom_frequency.png"):
        """Generate a bar chart of symptom frequency across all diseases"""
        symptom_counts = {}
        
        # Count occurrences of each symptom
        for symptom_list in self.df['symptoms']:
            if isinstance(symptom_list, str):
                symptoms = symptom_list.split(',')
                for symptom in symptoms:
                    symptom = symptom.strip()
                    if symptom in symptom_counts:
                        symptom_counts[symptom] += 1
                    else:
                        symptom_counts[symptom] = 1
        
        # Create dataframe for plotting
        symptom_df = pd.DataFrame({
            'symptom': list(symptom_counts.keys()),
            'frequency': list(symptom_counts.values())
        }).sort_values('frequency', ascending=False).head(15)  # Top 15 symptoms
        
        plt.figure(figsize=(12, 8))
        chart = sns.barplot(x='symptom', y='frequency', data=symptom_df)
        chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
        plt.title('Top 15 Symptom Frequencies Across All Diseases')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        return output_path
    
    def symptom_correlation_heatmap(self, output_path="symptom_correlation.png"):
        """Generate a heatmap showing correlation between symptoms"""
        # Create a one-hot encoding of symptoms
        all_symptoms = set()
        for symptom_list in self.df['symptoms']:
            if isinstance(symptom_list, str):
                symptoms = [s.strip() for s in symptom_list.split(',')]
                all_symptoms.update(symptoms)
        
        # Create a dataframe with one-hot encoded symptoms
        one_hot_df = pd.DataFrame(0, index=range(len(self.df)), columns=list(all_symptoms))
        
        for i, symptom_list in enumerate(self.df['symptoms']):
            if isinstance(symptom_list, str):
                symptoms = [s.strip() for s in symptom_list.split(',')]
                for symptom in symptoms:
                    one_hot_df.loc[i, symptom] = 1
        
        # Calculate correlation matrix (limited to avoid too large visualization)
        top_symptoms = one_hot_df.sum().sort_values(ascending=False).head(10).index
        correlation_matrix = one_hot_df[top_symptoms].corr()
        
        # Plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Between Top 10 Symptoms')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        return output_path
    
    def export_to_json(self, output_path="medical_data.json"):
        """Export cleaned data to JSON format for frontend consumption"""
        data = []
        
        for _, row in self.df.iterrows():
            symptoms = []
            if isinstance(row['symptoms'], str):
                symptoms = [s.strip() for s in row['symptoms'].split(',')]
            
            disease_data = {
                "disease": row['disease'],
                "symptoms": symptoms,
                "prevalence": float(row['prevalence'])
            }
            data.append(disease_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        return output_path
    
    def run_full_analysis(self, output_dir="analysis_output"):
        """Run the complete EDA process and save results"""
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(exist_ok=True)
        
        # Clean the data
        self.clean_data()
        
        # Explore data and save statistics
        stats = self.explore_data()
        with open(f"{output_dir}/statistics.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Generate charts
        self.disease_prevalence_chart(f"{output_dir}/disease_prevalence.png")
        self.symptom_frequency_chart(f"{output_dir}/symptom_frequency.png")
        self.symptom_correlation_heatmap(f"{output_dir}/symptom_correlation.png")
        
        # Export processed data for frontend
        self.export_to_json(f"{output_dir}/medical_data.json")
        
        return {
            "statistics": f"{output_dir}/statistics.json",
            "disease_prevalence": f"{output_dir}/disease_prevalence.png",
            "symptom_frequency": f"{output_dir}/symptom_frequency.png",
            "symptom_correlation": f"{output_dir}/symptom_correlation.png",
            "processed_data": f"{output_dir}/medical_data.json"
        }


if __name__ == "__main__":
    # Example usage
    analyzer = MedicalDataAnalyzer()
    results = analyzer.run_full_analysis()
    print("Analysis complete. Files generated:")
    for key, path in results.items():
        print(f"- {key}: {path}")

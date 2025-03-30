
# Medical Data EDA (Exploratory Data Analysis)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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
            
        # Convert symptoms to a standardized format and remove extra whitespace
        if 'symptoms' in self.df.columns:
            self.df['symptoms'] = self.df['symptoms'].apply(
                lambda x: ','.join([s.strip() for s in str(x).split(',')]) if isinstance(x, str) else ''
            )
        
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
        
        # Calculate symptom frequencies
        symptom_counts = {}
        for symptom_list in self.df['symptoms']:
            if isinstance(symptom_list, str):
                symptoms = symptom_list.split(',')
                for symptom in symptoms:
                    symptom = symptom.strip()
                    if symptom in symptom_counts:
                        symptom_counts[symptom] += 1
                    else:
                        symptom_counts[symptom] = 1
        
        print("\nTop 5 most common symptoms:")
        top_symptoms = sorted(symptom_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        for symptom, count in top_symptoms:
            print(f"- {symptom}: {count} occurrences")
        
        return {
            "disease_count": len(self.df),
            "unique_symptoms": len(unique_symptoms),
            "top_symptoms": list(unique_symptoms)[:10],
            "mean_prevalence": float(self.df['prevalence'].mean()),
            "min_prevalence": float(self.df['prevalence'].min()),
            "max_prevalence": float(self.df['prevalence'].max()),
            "symptom_frequency": {symptom: count for symptom, count in top_symptoms}
        }
    
    def disease_prevalence_chart(self, output_path="disease_prevalence.png"):
        """Generate a bar chart of disease prevalence"""
        plt.figure(figsize=(12, 8))
        
        # Sort by prevalence for better visualization
        sorted_df = self.df.sort_values('prevalence', ascending=False)
        
        chart = sns.barplot(x='disease', y='prevalence', data=sorted_df)
        chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
        plt.title('Disease Prevalence')
        plt.xlabel('Disease')
        plt.ylabel('Prevalence (%)')
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
        plt.xlabel('Symptom')
        plt.ylabel('Frequency')
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
    
    def disease_symptom_network(self, output_path="disease_symptom_network.png"):
        """Generate a scatter plot showing relationship between symptom count and prevalence"""
        # Count number of symptoms per disease
        self.df['symptom_count'] = self.df['symptoms'].apply(
            lambda x: len(str(x).split(',')) if isinstance(x, str) else 0
        )
        
        plt.figure(figsize=(10, 6))
        scatter = sns.scatterplot(
            x='symptom_count', 
            y='prevalence', 
            data=self.df,
            hue='disease',
            s=100,
            alpha=0.7
        )
        
        plt.title('Relationship Between Number of Symptoms and Disease Prevalence')
        plt.xlabel('Number of Symptoms')
        plt.ylabel('Prevalence (%)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        return output_path
    
    def symptom_pca_visualization(self, output_path="symptom_pca.png"):
        """Perform PCA analysis on symptoms to visualize disease clustering"""
        # Create a one-hot encoding of symptoms
        all_symptoms = set()
        for symptom_list in self.df['symptoms']:
            if isinstance(symptom_list, str):
                symptoms = [s.strip() for s in symptom_list.split(',')]
                all_symptoms.update(symptoms)
        
        # Create a dataframe with one-hot encoded symptoms
        one_hot_df = pd.DataFrame(0, index=self.df['disease'], columns=list(all_symptoms))
        
        for i, row in self.df.iterrows():
            if isinstance(row['symptoms'], str):
                symptoms = [s.strip() for s in row['symptoms'].split(',')]
                for symptom in symptoms:
                    one_hot_df.loc[row['disease'], symptom] = 1
        
        # Apply PCA if we have enough samples
        if len(one_hot_df) >= 3:
            # Standardize the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(one_hot_df)
            
            # Apply PCA
            n_components = min(2, len(one_hot_df) - 1)  # At most 2 components
            pca = PCA(n_components=n_components)
            pca_result = pca.fit_transform(scaled_data)
            
            # Create DataFrame for plotting
            pca_df = pd.DataFrame(
                data=pca_result,
                columns=[f'PC{i+1}' for i in range(n_components)],
                index=one_hot_df.index
            )
            
            # Add prevalence as a column
            pca_df['prevalence'] = self.df.set_index('disease')['prevalence']
            
            # Create the plot
            plt.figure(figsize=(10, 8))
            scatter = sns.scatterplot(
                x='PC1',
                y='PC2' if n_components > 1 else 'prevalence',
                data=pca_df,
                hue=pca_df.index,
                s=100,
                alpha=0.7
            )
            
            # Add labels
            for i, txt in enumerate(pca_df.index):
                plt.annotate(
                    txt, 
                    (pca_df.iloc[i, 0], pca_df.iloc[i, 1] if n_components > 1 else pca_df.iloc[i, 2]),
                    fontsize=9,
                    alpha=0.8
                )
            
            plt.title('PCA of Diseases Based on Symptoms')
            plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            if n_components > 1:
                plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            else:
                plt.ylabel('Prevalence')
                
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            return output_path
        
        return None
    
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
    
    def export_eda_results(self, output_path="eda_results.json"):
        """Export EDA metrics and visualizations metadata for frontend consumption"""
        # Ensure the data is cleaned
        self.clean_data()
        
        # Get basic statistics
        stats = self.explore_data()
        
        # Count symptoms by disease
        disease_symptom_counts = {}
        for _, row in self.df.iterrows():
            if isinstance(row['symptoms'], str):
                symptom_count = len(row['symptoms'].split(','))
                disease_symptom_counts[row['disease']] = symptom_count
        
        # Get symptom frequency
        symptom_counts = {}
        for symptom_list in self.df['symptoms']:
            if isinstance(symptom_list, str):
                symptoms = symptom_list.split(',')
                for symptom in symptoms:
                    symptom = symptom.strip()
                    if symptom in symptom_counts:
                        symptom_counts[symptom] += 1
                    else:
                        symptom_counts[symptom] = 1
        
        # Format data for export
        eda_results = {
            "statistics": stats,
            "disease_symptom_counts": disease_symptom_counts,
            "symptom_frequencies": {k: v for k, v in sorted(symptom_counts.items(), key=lambda item: item[1], reverse=True)},
            "visualization_paths": {
                "disease_prevalence": "disease_prevalence.png",
                "symptom_frequency": "symptom_frequency.png",
                "symptom_correlation": "symptom_correlation.png",
                "disease_symptom_network": "disease_symptom_network.png",
                "symptom_pca": "symptom_pca.png"
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(eda_results, f, indent=2)
        
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
        self.disease_symptom_network(f"{output_dir}/disease_symptom_network.png")
        
        # Try PCA visualization if we have enough data
        pca_path = self.symptom_pca_visualization(f"{output_dir}/symptom_pca.png")
        
        # Export processed data for frontend
        self.export_to_json(f"{output_dir}/medical_data.json")
        
        # Export EDA results
        self.export_eda_results(f"{output_dir}/eda_results.json")
        
        return {
            "statistics": f"{output_dir}/statistics.json",
            "disease_prevalence": f"{output_dir}/disease_prevalence.png",
            "symptom_frequency": f"{output_dir}/symptom_frequency.png",
            "symptom_correlation": f"{output_dir}/symptom_correlation.png",
            "disease_symptom_network": f"{output_dir}/disease_symptom_network.png",
            "symptom_pca": pca_path if pca_path else "Not available - insufficient data",
            "processed_data": f"{output_dir}/medical_data.json",
            "eda_results": f"{output_dir}/eda_results.json"
        }


def run_analysis_on_file(csv_path, output_dir="analysis_output"):
    """
    Helper function to run analysis on a specific CSV file
    """
    print(f"Running analysis on file: {csv_path}")
    analyzer = MedicalDataAnalyzer(csv_path=csv_path)
    results = analyzer.run_full_analysis(output_dir=output_dir)
    print("Analysis complete. Files generated:")
    for key, path in results.items():
        print(f"- {key}: {path}")
    return results


if __name__ == "__main__":
    # Example usage with command line args
    import sys
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "analysis_output"
        run_analysis_on_file(csv_path, output_dir)
    else:
        # Default operation - use example_data.csv in the same directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        example_csv = os.path.join(script_dir, "example_data.csv")
        
        if os.path.exists(example_csv):
            run_analysis_on_file(example_csv)
        else:
            # Use default data
            analyzer = MedicalDataAnalyzer()
            results = analyzer.run_full_analysis()
            print("Analysis complete using default data. Files generated:")
            for key, path in results.items():
                print(f"- {key}: {path}")

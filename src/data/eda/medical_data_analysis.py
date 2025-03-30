
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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import skew, kurtosis
import joblib

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
            self.create_default_dataframe()
        
        # Initialize model
        self.model = None

    def load_data(self, csv_path):
        """Load data from CSV file"""
        try:
            df = pd.read_csv(csv_path)
            print(f"Successfully loaded data from {csv_path}")
            print(f"Shape of the dataset: {df.shape}")
            return df
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
        print("Cleaning data...")
        
        # Check for missing values
        print(f"Missing values: \n{self.df.isnull().sum()}")
        
        # Handle missing values
        self.df = self.df.dropna(subset=['disease', 'symptoms'])
        
        # Check for duplicates
        duplicate_count = self.df.duplicated().sum()
        print(f"Duplicate rows: {duplicate_count}")
        
        # Remove duplicates
        self.df = self.df.drop_duplicates()
        
        # Standardize column names
        self.df.columns = self.df.columns.str.strip().str.lower().str.replace(" ", "_")
        
        # Standardize disease names
        if 'disease' in self.df.columns:
            self.df['disease'] = self.df['disease'].astype(str).str.lower().str.title()
            self.df['disease'] = self.df['disease'].str.replace(r'[^a-zA-Z0-9 ]', '', regex=True)
        
        # Ensure prevalence is numeric
        if 'prevalence' in self.df.columns:
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
        
        # If age column exists, create age bins
        if 'age' in self.df.columns:
            age_bins = [0, 25, 45, 65, 100]
            age_labels = ["Young", "Middle-aged", "Senior", "Elderly"]
            self.df["age_binned"] = pd.cut(self.df["age"], bins=age_bins, labels=age_labels)
            print(self.df[["age", "age_binned"]].head(10))
        
        # Group rare diseases
        if 'disease' in self.df.columns:
            disease_counts = self.df["disease"].value_counts()
            common_diseases = disease_counts[disease_counts > 5].index
            self.df["disease_grouped"] = self.df["disease"].apply(lambda x: x if x in common_diseases else "Others")
        
        print("Data cleaning completed.")
        return self.df
    
    def analyze_distribution(self):
        """Analyze the distribution of numerical columns"""
        print("Analyzing distributions...")
        
        # Get numerical columns
        numeric_cols = self.df.select_dtypes(include=['number'])
        
        if len(numeric_cols.columns) > 0:
            # Calculate skewness and kurtosis
            skewness_values = numeric_cols.apply(lambda x: skew(x, nan_policy='omit'))
            kurtosis_values = numeric_cols.apply(lambda x: kurtosis(x, nan_policy='omit'))
            
            print("Skewness of numerical columns:\n", skewness_values)
            print("\nKurtosis of numerical columns:\n", kurtosis_values)
            
            # Check for outliers in age if available
            if 'age' in self.df.columns:
                Q1 = self.df["age"].quantile(0.25)
                Q3 = self.df["age"].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = self.df[(self.df["age"] < lower_bound) | (self.df["age"] > upper_bound)]
                print(f"Number of outliers in age: {len(outliers)}")
        
        print("Distribution analysis completed.")
        return {
            "numeric_columns": list(numeric_cols.columns),
            "skewness": skewness_values.to_dict() if len(numeric_cols.columns) > 0 else {},
            "kurtosis": kurtosis_values.to_dict() if len(numeric_cols.columns) > 0 else {}
        }
    
    def explore_data(self):
        """Print basic statistics and information about the data"""
        print("Data Overview:")
        print(f"Number of records: {len(self.df)}")
        
        if 'disease' in self.df.columns:
            print(f"Number of diseases: {self.df['disease'].nunique()}")
            
            # Disease counts
            disease_counts = self.df['disease'].value_counts()
            print(f"Top 5 most common diseases:")
            for disease, count in disease_counts.head(5).items():
                print(f"- {disease}: {count} occurrences")
        
        # Extract all symptoms if available
        all_symptoms = []
        if 'symptoms' in self.df.columns:
            for symptom_list in self.df['symptoms']:
                if isinstance(symptom_list, str):
                    symptoms = symptom_list.split(',')
                    all_symptoms.extend(symptoms)
            
            unique_symptoms = set(all_symptoms)
            print(f"Number of unique symptoms: {len(unique_symptoms)}")
            print(f"Top 10 symptoms: {list(unique_symptoms)[:10]}")
        
        # Prevalence statistics if available
        if 'prevalence' in self.df.columns:
            print("\nPrevalence Statistics:")
            print(f"Mean prevalence: {self.df['prevalence'].mean():.2f}")
            print(f"Min prevalence: {self.df['prevalence'].min()}")
            print(f"Max prevalence: {self.df['prevalence'].max()}")
        
        # Age statistics if available
        if 'age' in self.df.columns:
            print("\nAge Statistics:")
            print(f"Mean age: {self.df['age'].mean():.2f}")
            print(f"Min age: {self.df['age'].min()}")
            print(f"Max age: {self.df['age'].max()}")
            
            if 'age_binned' in self.df.columns:
                age_distribution = self.df['age_binned'].value_counts(normalize=True) * 100
                print("\nAge group distribution (%):")
                for age_group, percentage in age_distribution.items():
                    print(f"- {age_group}: {percentage:.1f}%")
        
        # Calculate symptom frequencies
        symptom_counts = {}
        if 'symptoms' in self.df.columns:
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
        
        # Gender distribution if available
        if 'gender' in self.df.columns:
            gender_distribution = self.df['gender'].value_counts(normalize=True) * 100
            print("\nGender distribution (%):")
            for gender, percentage in gender_distribution.items():
                print(f"- {gender}: {percentage:.1f}%")
            
            if 'disease' in self.df.columns:
                # Gender-disease relationship
                gender_disease = pd.crosstab(self.df["disease_grouped"] if "disease_grouped" in self.df.columns else self.df["disease"], 
                                            self.df["gender"])
                print("\nDisease by Gender:")
                print(gender_disease)
        
        # Prepare statistics for return
        statistics = {
            "record_count": len(self.df),
            "column_names": list(self.df.columns)
        }
        
        if 'disease' in self.df.columns:
            statistics["disease_count"] = self.df['disease'].nunique()
            statistics["top_diseases"] = disease_counts.head(10).to_dict()
        
        if 'symptoms' in self.df.columns and all_symptoms:
            statistics["unique_symptoms"] = len(set(all_symptoms))
            statistics["top_symptoms"] = dict(sorted(symptom_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        
        if 'prevalence' in self.df.columns:
            statistics["prevalence"] = {
                "mean": float(self.df['prevalence'].mean()),
                "min": float(self.df['prevalence'].min()),
                "max": float(self.df['prevalence'].max())
            }
        
        if 'age' in self.df.columns:
            statistics["age"] = {
                "mean": float(self.df['age'].mean()),
                "min": float(self.df['age'].min()),
                "max": float(self.df['age'].max())
            }
            
            if 'age_binned' in self.df.columns:
                statistics["age_distribution"] = self.df['age_binned'].value_counts(normalize=True).to_dict()
        
        if 'gender' in self.df.columns:
            statistics["gender_distribution"] = self.df['gender'].value_counts(normalize=True).to_dict()
        
        return statistics
    
    def disease_prevalence_chart(self, output_path="disease_prevalence.png"):
        """Generate a bar chart of disease prevalence or counts"""
        if 'disease' not in self.df.columns:
            print("Disease column not found. Cannot create prevalence chart.")
            return None
        
        plt.figure(figsize=(14, 8))
        
        if 'prevalence' in self.df.columns:
            # Use existing prevalence column
            disease_data = self.df.groupby('disease')['prevalence'].mean().sort_values(ascending=False).head(15)
            title = 'Disease Prevalence'
            y_label = 'Prevalence (%)'
        else:
            # Use disease counts
            disease_data = self.df['disease'].value_counts().head(15)
            title = 'Disease Frequency'
            y_label = 'Count'
        
        # Create the chart
        ax = disease_data.plot(kind='bar', color='#4285F4')
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Disease', fontsize=14)
        ax.set_ylabel(y_label, fontsize=14)
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for i, v in enumerate(disease_data):
            ax.text(i, v * 1.01, str(round(v, 1)), ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        return output_path
    
    def symptom_frequency_chart(self, output_path="symptom_frequency.png"):
        """Generate a bar chart of symptom frequency across all diseases"""
        if 'symptoms' not in self.df.columns:
            print("Symptoms column not found. Cannot create symptom frequency chart.")
            return None
        
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
        
        plt.figure(figsize=(14, 8))
        ax = sns.barplot(x='symptom', y='frequency', data=symptom_df, palette='viridis')
        ax.set_title('Top 15 Symptom Frequencies Across All Diseases', fontsize=16)
        ax.set_xlabel('Symptom', fontsize=14)
        ax.set_ylabel('Frequency', fontsize=14)
        ax.tick_params(axis='x', rotation=45, labelsize=12)
        
        # Add value labels
        for i, v in enumerate(symptom_df['frequency']):
            ax.text(i, v + 0.1, str(v), ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        return output_path
    
    def symptom_correlation_heatmap(self, output_path="symptom_correlation.png"):
        """Generate a heatmap showing correlation between symptoms"""
        if 'symptoms' not in self.df.columns:
            print("Symptoms column not found. Cannot create correlation heatmap.")
            return None
        
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
        top_symptoms = one_hot_df.sum().sort_values(ascending=False).head(15).index
        correlation_matrix = one_hot_df[top_symptoms].corr()
        
        # Plot heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   vmin=-1, vmax=1, square=True, linewidths=.5)
        plt.title('Correlation Between Top 15 Symptoms', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        return output_path
    
    def disease_symptom_network(self, output_path="disease_symptom_network.png"):
        """Generate a scatter plot showing relationship between symptom count and prevalence"""
        if 'symptoms' not in self.df.columns:
            print("Symptoms column not found. Cannot create network visualization.")
            return None
        
        # Count number of symptoms per disease
        self.df['symptom_count'] = self.df['symptoms'].apply(
            lambda x: len(str(x).split(',')) if isinstance(x, str) else 0
        )
        
        plt.figure(figsize=(12, 8))
        
        if 'prevalence' in self.df.columns:
            # If prevalence is available, use it for y-axis
            y_col = 'prevalence'
            y_label = 'Prevalence (%)'
        elif 'age' in self.df.columns:
            # If age is available, use average age for y-axis
            self.df['age_mean'] = self.df.groupby('disease')['age'].transform('mean')
            y_col = 'age_mean'
            y_label = 'Average Patient Age'
        else:
            # Otherwise, use symptom count for both axes with slight jitter
            self.df['jittered'] = self.df['symptom_count'] + np.random.normal(0, 0.1, len(self.df))
            y_col = 'jittered'
            y_label = 'Symptom Count (jittered)'
        
        scatter = sns.scatterplot(
            x='symptom_count', 
            y=y_col, 
            data=self.df,
            hue='disease',
            s=100,
            alpha=0.7
        )
        
        plt.title('Relationship Between Number of Symptoms and ' + y_label, fontsize=16)
        plt.xlabel('Number of Symptoms', fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        
        # Limit legend items to avoid overcrowding
        handles, labels = scatter.get_legend_handles_labels()
        if len(labels) > 15:  # If too many diseases
            plt.legend(handles[:15], labels[:15], bbox_to_anchor=(1.05, 1), loc='upper left', title="Disease")
            plt.figtext(0.9, 0.5, f"+ {len(labels)-15} more diseases...", wrap=True, fontsize=10)
        else:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Disease")
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        return output_path
    
    def age_distribution_chart(self, output_path="age_distribution.png"):
        """Generate a visualization of age distribution by disease"""
        if 'age' not in self.df.columns or 'disease' not in self.df.columns:
            print("Age or disease column not found. Cannot create age distribution chart.")
            return None
        
        plt.figure(figsize=(14, 8))
        
        # Get top 10 diseases by frequency
        top_diseases = self.df['disease'].value_counts().head(10).index
        filtered_df = self.df[self.df['disease'].isin(top_diseases)]
        
        # Create a box plot
        ax = sns.boxplot(x='disease', y='age', data=filtered_df, palette='Set3')
        
        ax.set_title('Age Distribution by Disease Type', fontsize=16)
        ax.set_xlabel('Disease', fontsize=14)
        ax.set_ylabel('Age', fontsize=14)
        ax.tick_params(axis='x', rotation=45, labelsize=12)
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        return output_path
    
    def gender_disease_chart(self, output_path="gender_disease.png"):
        """Generate a visualization of disease distribution by gender"""
        if 'gender' not in self.df.columns or 'disease' not in self.df.columns:
            print("Gender or disease column not found. Cannot create gender-disease chart.")
            return None
        
        plt.figure(figsize=(14, 8))
        
        # Use disease_grouped if available, otherwise use disease
        disease_col = 'disease_grouped' if 'disease_grouped' in self.df.columns else 'disease'
        
        # Get cross-tabulation data
        gender_disease_data = pd.crosstab(self.df[disease_col], self.df['gender'], normalize='index')
        
        # Plot stacked bar chart
        gender_disease_data.plot(kind='bar', stacked=True, colormap='viridis')
        
        plt.title('Disease Distribution by Gender', fontsize=16)
        plt.xlabel('Disease', fontsize=14)
        plt.ylabel('Proportion', fontsize=14)
        plt.xticks(rotation=45)
        plt.legend(title='Gender')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        return output_path
    
    def symptom_pca_visualization(self, output_path="symptom_pca.png"):
        """Perform PCA analysis on symptoms to visualize disease clustering"""
        if 'symptoms' not in self.df.columns or 'disease' not in self.df.columns:
            print("Symptoms or disease column not found. Cannot create PCA visualization.")
            return None
        
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
        
        # Remove duplicates (same disease with different rows)
        one_hot_df = one_hot_df.groupby(level=0).max()
        
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
            
            # Add prevalence as a column if available
            if 'prevalence' in self.df.columns:
                # Calculate average prevalence per disease
                prev_by_disease = self.df.groupby('disease')['prevalence'].mean()
                pca_df['prevalence'] = pca_df.index.map(prev_by_disease)
            
            # Create the plot
            plt.figure(figsize=(12, 10))
            
            # Use color-coding based on prevalence or just by disease
            if 'prevalence' in pca_df.columns:
                scatter = plt.scatter(
                    pca_df['PC1'],
                    pca_df['PC2'] if n_components > 1 else pca_df['prevalence'],
                    c=pca_df['prevalence'],
                    cmap='viridis',
                    s=100,
                    alpha=0.7
                )
                plt.colorbar(scatter, label='Prevalence')
            else:
                scatter = plt.scatter(
                    pca_df['PC1'],
                    pca_df['PC2'] if n_components > 1 else np.zeros(len(pca_df)),
                    s=100,
                    alpha=0.7
                )
            
            # Add labels
            for i, txt in enumerate(pca_df.index):
                plt.annotate(
                    txt, 
                    (pca_df.iloc[i, 0], pca_df.iloc[i, 1] if n_components > 1 else (pca_df['prevalence'].iloc[i] if 'prevalence' in pca_df.columns else 0)),
                    fontsize=9,
                    alpha=0.8
                )
            
            plt.title('PCA of Diseases Based on Symptoms', fontsize=16)
            plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=14)
            
            if n_components > 1:
                plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=14)
            else:
                plt.ylabel('Prevalence' if 'prevalence' in pca_df.columns else '', fontsize=14)
            
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            return output_path
        
        return None
    
    def train_prediction_model(self, target_column=None):
        """Train a Random Forest model for disease prediction"""
        if not target_column:
            # Try to identify a suitable target column
            if 'disease' in self.df.columns:
                target_column = 'disease'
            elif 'outcome' in self.df.columns:
                target_column = 'outcome'
            else:
                # Look for columns that might be outcome variables
                potential_targets = [col for col in self.df.columns if 'outcome' in col.lower() or 'disease' in col.lower()]
                if potential_targets:
                    target_column = potential_targets[0]
                else:
                    print("No suitable target column found for model training.")
                    return None
        
        if target_column not in self.df.columns:
            print(f"Target column '{target_column}' not found in the dataset.")
            return None
        
        print(f"Training prediction model using '{target_column}' as the target variable...")
        
        # Convert target to categorical if needed
        self.df[target_column] = self.df[target_column].astype('category')
        target_categories = self.df[target_column].cat.categories.tolist()
        
        # Get symptoms columns if available
        if 'symptoms' in self.df.columns:
            # One-hot encode symptoms
            all_symptoms = set()
            for symptom_list in self.df['symptoms']:
                if isinstance(symptom_list, str):
                    symptoms = [s.strip() for s in symptom_list.split(',')]
                    all_symptoms.update(symptoms)
            
            # Create a dataframe with one-hot encoded symptoms
            X_symptoms = pd.DataFrame(0, index=range(len(self.df)), columns=list(all_symptoms))
            
            for i, symptom_list in enumerate(self.df['symptoms']):
                if isinstance(symptom_list, str):
                    symptoms = [s.strip() for s in symptom_list.split(',')]
                    for symptom in symptoms:
                        X_symptoms.loc[i, symptom] = 1
            
            # Use X_symptoms as features
            X = X_symptoms
        else:
            # Use all columns except the target as features
            X = self.df.drop(columns=[target_column])
            
            # Drop any non-numeric columns
            for col in X.columns:
                if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                    X = X.drop(columns=[col])
        
        y = self.df[target_column].cat.codes  # Use integer codes for target
        
        # Check if we have enough data
        if len(X) < 10:
            print("Not enough data for model training.")
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.2f}")
        
        # Print classification report
        print("Classification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=[str(cat) for cat in target_categories]))
        
        # Store model and mappings
        self.model = model
        self.feature_names = X.columns.tolist()
        self.target_categories = target_categories
        
        # Save model
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, "models/medical_diagnosis_model.pkl")
        
        # Save feature names and target categories
        with open("models/model_metadata.json", "w") as f:
            json.dump({
                "feature_names": self.feature_names,
                "target_categories": self.target_categories
            }, f)
        
        print("Model saved to models/medical_diagnosis_model.pkl")
        
        return {
            "accuracy": accuracy,
            "feature_names": self.feature_names,
            "target_categories": self.target_categories,
            "model_path": "models/medical_diagnosis_model.pkl"
        }
    
    def predict_disease(self, symptoms):
        """Predict disease based on symptoms"""
        if not self.model:
            # Try to load model if available
            try:
                self.model = joblib.load("models/medical_diagnosis_model.pkl")
                with open("models/model_metadata.json", "r") as f:
                    model_data = json.load(f)
                    self.feature_names = model_data["feature_names"]
                    self.target_categories = model_data["target_categories"]
            except FileNotFoundError:
                print("No model available. Please train a model first.")
                return None
        
        # Create feature vector from symptoms
        X_pred = pd.DataFrame(0, index=[0], columns=self.feature_names)
        
        for symptom in symptoms:
            if symptom in self.feature_names:
                X_pred.loc[0, symptom] = 1
        
        # Make prediction
        prediction_code = self.model.predict(X_pred)[0]
        prediction = self.target_categories[prediction_code]
        
        # Get probabilities
        proba = self.model.predict_proba(X_pred)[0]
        proba_dict = {self.target_categories[i]: float(p) for i, p in enumerate(proba)}
        
        return {
            "prediction": prediction,
            "probabilities": proba_dict
        }
    
    def export_to_json(self, output_path="medical_data.json"):
        """Export cleaned data to JSON format for frontend consumption"""
        data = []
        
        for _, row in self.df.iterrows():
            symptoms = []
            if 'symptoms' in self.df.columns and isinstance(row['symptoms'], str):
                symptoms = [s.strip() for s in row['symptoms'].split(',')]
            
            disease_data = {
                "disease": row['disease'] if 'disease' in self.df.columns else "Unknown",
                "symptoms": symptoms
            }
            
            # Add prevalence if available
            if 'prevalence' in self.df.columns:
                disease_data["prevalence"] = float(row['prevalence'])
            
            # Add age if available
            if 'age' in self.df.columns:
                disease_data["age"] = float(row['age'])
            
            # Add gender if available
            if 'gender' in self.df.columns:
                disease_data["gender"] = row['gender']
            
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
        if 'disease' in self.df.columns and 'symptoms' in self.df.columns:
            for _, row in self.df.iterrows():
                if isinstance(row['symptoms'], str):
                    symptom_count = len(row['symptoms'].split(','))
                    disease_symptom_counts[row['disease']] = symptom_count
        
        # Get symptom frequency
        symptom_counts = {}
        if 'symptoms' in self.df.columns:
            for symptom_list in self.df['symptoms']:
                if isinstance(symptom_list, str):
                    symptoms = symptom_list.split(',')
                    for symptom in symptoms:
                        symptom = symptom.strip()
                        if symptom in symptom_counts:
                            symptom_counts[symptom] += 1
                        else:
                            symptom_counts[symptom] = 1
        
        # Get age distribution by disease
        age_by_disease = {}
        if 'age' in self.df.columns and 'disease' in self.df.columns:
            age_by_disease = self.df.groupby('disease')['age'].mean().to_dict()
        
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
        
        if age_by_disease:
            eda_results["age_by_disease"] = age_by_disease
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(eda_results, f, indent=2)
        
        return output_path
    
    def run_full_analysis(self, output_dir="analysis_output"):
        """Run the complete EDA process and save results"""
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(exist_ok=True)
        
        print("Running full analysis...")
        
        # Clean the data
        self.clean_data()
        
        # Explore data and save statistics
        stats = self.explore_data()
        with open(f"{output_dir}/statistics.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Analyze distributions
        distribution_stats = self.analyze_distribution()
        with open(f"{output_dir}/distribution_stats.json", 'w') as f:
            json.dump(distribution_stats, f, indent=2)
        
        # Generate charts
        charts = {}
        
        # Disease prevalence chart
        disease_chart = self.disease_prevalence_chart(f"{output_dir}/disease_prevalence.png")
        if disease_chart:
            charts["disease_prevalence"] = disease_chart
        
        # Symptom frequency chart
        symptom_chart = self.symptom_frequency_chart(f"{output_dir}/symptom_frequency.png")
        if symptom_chart:
            charts["symptom_frequency"] = symptom_chart
        
        # Symptom correlation heatmap
        correlation_chart = self.symptom_correlation_heatmap(f"{output_dir}/symptom_correlation.png")
        if correlation_chart:
            charts["symptom_correlation"] = correlation_chart
        
        # Disease-symptom network
        network_chart = self.disease_symptom_network(f"{output_dir}/disease_symptom_network.png")
        if network_chart:
            charts["disease_symptom_network"] = network_chart
        
        # Age distribution chart
        age_chart = self.age_distribution_chart(f"{output_dir}/age_distribution.png")
        if age_chart:
            charts["age_distribution"] = age_chart
        
        # Gender-disease chart
        gender_chart = self.gender_disease_chart(f"{output_dir}/gender_disease.png")
        if gender_chart:
            charts["gender_disease"] = gender_chart
        
        # PCA visualization
        pca_chart = self.symptom_pca_visualization(f"{output_dir}/symptom_pca.png")
        if pca_chart:
            charts["symptom_pca"] = pca_chart
        
        # Train prediction model if possible
        model_info = self.train_prediction_model()
        if model_info:
            with open(f"{output_dir}/model_info.json", 'w') as f:
                json.dump(model_info, f, indent=2)
        
        # Export processed data for frontend
        data_json = self.export_to_json(f"{output_dir}/medical_data.json")
        
        # Export EDA results
        eda_json = self.export_eda_results(f"{output_dir}/eda_results.json")
        
        print("Analysis complete.")
        
        return {
            "statistics": f"{output_dir}/statistics.json",
            "distribution_stats": f"{output_dir}/distribution_stats.json",
            "charts": charts,
            "model_info": f"{output_dir}/model_info.json" if model_info else "Not available",
            "processed_data": data_json,
            "eda_results": eda_json
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
        if isinstance(path, dict):
            print(f"- {key}:")
            for sub_key, sub_path in path.items():
                print(f"  - {sub_key}: {sub_path}")
        else:
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
                if isinstance(path, dict):
                    print(f"- {key}:")
                    for sub_key, sub_path in path.items():
                        print(f"  - {sub_key}: {sub_path}")
                else:
                    print(f"- {key}: {path}")


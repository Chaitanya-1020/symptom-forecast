
// This represents our "CSV data" in a structured format
export interface DiseaseData {
  disease: string;
  symptoms: string[];
  prevalence: number; // A number between 0-100 representing how common the disease is
}

// This would typically come from a CSV file in a real backend
export const medicalData: DiseaseData[] = [
  {
    disease: "Common Cold",
    symptoms: ["Cough", "Runny Nose", "Sore Throat", "Sneezing", "Headache", "Fatigue"],
    prevalence: 78
  },
  {
    disease: "Influenza",
    symptoms: ["High Fever", "Cough", "Fatigue", "Body Aches", "Headache", "Chills"],
    prevalence: 65
  },
  {
    disease: "COVID-19",
    symptoms: ["Fever", "Dry Cough", "Fatigue", "Loss of Taste/Smell", "Shortness of Breath", "Body Aches"],
    prevalence: 60
  },
  {
    disease: "Allergic Rhinitis",
    symptoms: ["Sneezing", "Runny Nose", "Itchy Eyes", "Nasal Congestion", "Watery Eyes"],
    prevalence: 45
  },
  {
    disease: "Sinusitis",
    symptoms: ["Facial Pain", "Nasal Congestion", "Headache", "Thick Nasal Discharge", "Reduced Sense of Smell"],
    prevalence: 40
  },
  {
    disease: "Bronchitis",
    symptoms: ["Persistent Cough", "Mucus Production", "Fatigue", "Shortness of Breath", "Chest Discomfort"],
    prevalence: 35
  },
  {
    disease: "Asthma",
    symptoms: ["Wheezing", "Shortness of Breath", "Chest Tightness", "Coughing", "Difficulty Breathing"],
    prevalence: 32
  },
  {
    disease: "Pneumonia",
    symptoms: ["High Fever", "Cough with Phlegm", "Shortness of Breath", "Chest Pain", "Fatigue", "Confusion"],
    prevalence: 25
  },
  {
    disease: "Gastroenteritis",
    symptoms: ["Nausea", "Vomiting", "Diarrhea", "Stomach Cramps", "Fever", "Headache"],
    prevalence: 55
  },
  {
    disease: "Migraine",
    symptoms: ["Severe Headache", "Nausea", "Light Sensitivity", "Sound Sensitivity", "Visual Disturbances"],
    prevalence: 42
  },
  {
    disease: "Urinary Tract Infection",
    symptoms: ["Frequent Urination", "Burning Sensation", "Cloudy Urine", "Strong Odor", "Pelvic Pain"],
    prevalence: 38
  },
  {
    disease: "Food Poisoning",
    symptoms: ["Nausea", "Vomiting", "Diarrhea", "Abdominal Pain", "Fever", "Dehydration"],
    prevalence: 50
  }
];

// Extract all unique symptoms from the data
export const getAllSymptoms = (): string[] => {
  const symptomSet = new Set<string>();
  medicalData.forEach(disease => {
    disease.symptoms.forEach(symptom => {
      symptomSet.add(symptom);
    });
  });
  return Array.from(symptomSet).sort();
};

// Calculate disease probabilities based on selected symptoms
export const calculateDiseaseProbabilities = (selectedSymptoms: string[]): { disease: string; probability: number; matchedSymptoms: number; totalSymptoms: number }[] => {
  if (selectedSymptoms.length === 0) return [];

  return medicalData.map(disease => {
    // Count how many symptoms match
    const matchedSymptoms = disease.symptoms.filter(symptom => 
      selectedSymptoms.includes(symptom)
    ).length;

    // Calculate a basic probability based on symptom match ratio and disease prevalence
    // This is a simple algorithm and could be improved with actual medical data
    const matchRatio = matchedSymptoms / disease.symptoms.length;
    const selectedRatio = matchedSymptoms / selectedSymptoms.length;
    
    // Weigh the probability based on both ratios and disease prevalence
    const probability = (matchRatio * 0.6 + selectedRatio * 0.4) * (disease.prevalence / 100);
    
    // Scale to a percentage and round to 2 decimal places
    const scaledProbability = Math.round(probability * 100 * 100) / 100;

    return {
      disease: disease.disease,
      probability: scaledProbability,
      matchedSymptoms,
      totalSymptoms: disease.symptoms.length
    };
  }).sort((a, b) => b.probability - a.probability); // Sort by probability desc
};

// Get the top prediction
export const getPrediction = (selectedSymptoms: string[]): string => {
  if (selectedSymptoms.length === 0) return "No symptoms selected";
  
  const probabilities = calculateDiseaseProbabilities(selectedSymptoms);
  return probabilities.length > 0 ? probabilities[0].disease : "Unable to determine";
};

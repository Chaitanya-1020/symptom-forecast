import type { Disease, Symptom } from '../types';

// Sample medical data (this would be replaced by data from the Python analysis)
const medicalData: Disease[] = [
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
    disease: "Pneumonia",
    symptoms: ["High Fever", "Cough with Phlegm", "Shortness of Breath", "Chest Pain", "Fatigue", "Confusion"],
    prevalence: 25
  },
  {
    disease: "Tuberculosis",
    symptoms: ["Persistent Cough", "Blood in Cough", "Weight Loss", "Night Sweats", "Fatigue", "Chest Pain"],
    prevalence: 20
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
  },
  {
    disease: "Hypertension",
    symptoms: ["Headache", "Shortness of Breath", "Dizziness", "Chest Pain", "Vision Problems"],
    prevalence: 30
  },
  {
    disease: "Diabetes",
    symptoms: ["Frequent Urination", "Excessive Thirst", "Hunger", "Fatigue", "Blurred Vision", "Slow Healing"],
    prevalence: 28
  },
  {
    disease: "Anemia",
    symptoms: ["Fatigue", "Weakness", "Pale Skin", "Shortness of Breath", "Dizziness", "Cold Hands and Feet"],
    prevalence: 22
  },
  {
    disease: "Hyperthyroidism",
    symptoms: ["Weight Loss", "Rapid Heartbeat", "Increased Appetite", "Nervousness", "Sweating", "Fatigue"],
    prevalence: 18
  },
  {
    disease: "Hypothyroidism",
    symptoms: ["Fatigue", "Weight Gain", "Cold Sensitivity", "Dry Skin", "Depression", "Constipation"],
    prevalence: 19
  },
  {
    disease: "GERD",
    symptoms: ["Heartburn", "Regurgitation", "Chest Pain", "Difficulty Swallowing", "Sore Throat"],
    prevalence: 33
  }
];

/**
 * Get a list of all unique symptoms from the medical data
 */
export function getAllSymptoms(): string[] {
  // Extract all symptoms and deduplicate
  const allSymptoms = new Set<string>();
  
  medicalData.forEach(disease => {
    disease.symptoms.forEach(symptom => {
      allSymptoms.add(symptom);
    });
  });
  
  // Convert to array and sort alphabetically
  return Array.from(allSymptoms).sort();
}

/**
 * Calculate disease probabilities based on symptoms
 * @param selectedSymptoms - Array of symptoms selected by user
 * @returns Array of objects with disease, probability, and symptom match info
 */
export function calculateDiseaseProbabilities(selectedSymptoms: string[]): Array<{
  disease: string;
  probability: number;
  matchedSymptoms: number;
  totalSymptoms: number;
}> {
  if (!selectedSymptoms.length) return [];
  
  // Calculate match percentages for each disease
  const results = medicalData.map(disease => {
    // Count how many of the selected symptoms match this disease
    const matchedSymptoms = disease.symptoms.filter(symptom => 
      selectedSymptoms.includes(symptom)
    ).length;
    
    // Calculate two factors:
    // 1. What percentage of this disease's symptoms were matched
    const percentOfDiseaseSymptoms = matchedSymptoms / disease.symptoms.length;
    
    // 2. What percentage of selected symptoms were matched
    const percentOfSelectedSymptoms = matchedSymptoms / selectedSymptoms.length;
    
    // Combined probability calculation
    // We weight these factors and also consider the base prevalence
    const baseProbability = (percentOfDiseaseSymptoms * 0.5) + (percentOfSelectedSymptoms * 0.5);
    
    // Adjust by the disease prevalence (normalized to 0-1)
    const adjustedProbability = baseProbability * (disease.prevalence / 100);
    
    // Scale to percentage (0-100)
    const scaledProbability = Math.round(adjustedProbability * 100);
    
    return {
      disease: disease.disease,
      probability: scaledProbability,
      matchedSymptoms: matchedSymptoms,
      totalSymptoms: disease.symptoms.length
    };
  });
  
  // Sort by probability (highest first) and filter out zero probability
  return results
    .filter(item => item.probability > 0)
    .sort((a, b) => b.probability - a.probability);
}

/**
 * Get detailed information about a specific disease
 * @param diseaseName - Name of the disease to look up
 * @returns Disease object or null if not found
 */
export function getDiseaseDetails(diseaseName: string): Disease | null {
  return medicalData.find(d => d.disease === diseaseName) || null;
}

/**
 * Get symptoms associated with a specific disease
 * @param diseaseName - Name of the disease to look up
 * @returns Array of symptoms or empty array if disease not found
 */
export function getDiseaseSymptoms(diseaseName: string): string[] {
  const disease = getDiseaseDetails(diseaseName);
  return disease ? disease.symptoms : [];
}

/**
 * Find diseases that share specific symptoms
 * @param symptomList - List of symptoms to match
 * @returns Array of disease names that have at least one matching symptom
 */
export function findDiseasesBySymptoms(symptomList: string[]): string[] {
  if (!symptomList.length) return [];
  
  return medicalData
    .filter(disease => 
      disease.symptoms.some(symptom => symptomList.includes(symptom))
    )
    .map(disease => disease.disease);
}

export { Disease, Symptom };

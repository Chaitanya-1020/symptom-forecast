
export interface Symptom {
  name: string;
  description?: string;
  severity?: "mild" | "moderate" | "severe";
}

export interface Disease {
  disease: string;
  symptoms: string[];
  prevalence: number;
  description?: string;
  treatments?: string[];
  age?: number | { min: number; max: number };
  gender?: string;
}

export interface PredictionResult {
  disease: string;
  probability: number;
  matchedSymptoms: number;
  totalSymptoms: number;
}

export interface MedicalStatistics {
  disease_count: number;
  unique_symptoms: number;
  top_symptoms: string[];
  mean_prevalence: number;
  min_prevalence: number;
  max_prevalence: number;
  symptom_frequency: Record<string, number>;
}

export interface AnalysisResult {
  statistics: MedicalStatistics;
  disease_symptom_counts: Record<string, number>;
  symptom_frequencies: Record<string, number>;
  visualization_paths: Record<string, string>;
  age_by_disease?: Record<string, number>;
}

export interface ModelInfo {
  accuracy: number;
  feature_names: string[];
  target_categories: string[];
  model_path: string;
}

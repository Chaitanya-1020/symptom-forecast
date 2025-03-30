
import React, { useState, useEffect } from 'react';
import Header from '@/components/Header';
import SymptomSelector from '@/components/SymptomSelector';
import DiseaseVisualization from '@/components/DiseaseVisualization';
import PredictionResult from '@/components/PredictionResult';
import Disclaimer from '@/components/Disclaimer';
import { getAllSymptoms, calculateDiseaseProbabilities } from '@/data/medical_data';
import { useToast } from "@/components/ui/use-toast";

const Index = () => {
  const [availableSymptoms] = useState<string[]>(getAllSymptoms());
  const [selectedSymptoms, setSelectedSymptoms] = useState<string[]>([]);
  const [predictionData, setPredictionData] = useState<any[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const { toast } = useToast();

  // Update predictions when symptoms change
  useEffect(() => {
    if (selectedSymptoms.length > 0) {
      setIsAnalyzing(true);
      
      // Simulate API call delay
      const timer = setTimeout(() => {
        const results = calculateDiseaseProbabilities(selectedSymptoms);
        setPredictionData(results);
        setIsAnalyzing(false);
      }, 800);
      
      return () => clearTimeout(timer);
    } else {
      setPredictionData([]);
    }
  }, [selectedSymptoms]);

  // Toggle symptom selection
  const handleSymptomToggle = (symptom: string) => {
    setSelectedSymptoms(prev => {
      if (prev.includes(symptom)) {
        return prev.filter(s => s !== symptom);
      } else {
        const newSelection = [...prev, symptom];
        
        // Show toast when first symptom is selected
        if (prev.length === 0) {
          toast({
            title: "Analysis started",
            description: "Select more symptoms for a more accurate prediction.",
          });
        }
        
        return newSelection;
      }
    });
  };

  // Clear all selected symptoms
  const handleClearAll = () => {
    setSelectedSymptoms([]);
    setPredictionData([]);
    toast({
      title: "Selections cleared",
      description: "All symptoms have been cleared.",
    });
  };

  // Get top prediction and its confidence level
  const getTopPrediction = () => {
    if (predictionData.length === 0) return { prediction: "", confidence: 0 };
    const topResult = predictionData[0];
    return {
      prediction: topResult.disease,
      confidence: topResult.probability
    };
  };

  const { prediction, confidence } = getTopPrediction();

  return (
    <div className="min-h-screen bg-gray-50">
      <Header />
      
      <main className="container mx-auto px-4 py-8">
        <Disclaimer />
        
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-1">
            <SymptomSelector 
              availableSymptoms={availableSymptoms}
              selectedSymptoms={selectedSymptoms}
              onSymptomToggle={handleSymptomToggle}
              onClear={handleClearAll}
            />
          </div>
          
          <div className="lg:col-span-2 space-y-6">
            <DiseaseVisualization data={predictionData} />
            <PredictionResult 
              prediction={prediction} 
              confidence={confidence} 
              isLoading={isAnalyzing} 
            />
          </div>
        </div>
      </main>
      
      <footer className="bg-gray-100 py-6 mt-8">
        <div className="container mx-auto px-4 text-center text-sm text-gray-600">
          <p>MediPredictor - Disease Prediction Tool</p>
          <p className="mt-1">Disclaimer: For educational purposes only.</p>
        </div>
      </footer>
    </div>
  );
};

export default Index;

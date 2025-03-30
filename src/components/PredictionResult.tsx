
import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { CircleArrowUp } from 'lucide-react';

interface PredictionResultProps {
  prediction: string;
  confidence: number;
  isLoading: boolean;
}

const PredictionResult = ({ prediction, confidence, isLoading }: PredictionResultProps) => {
  if (isLoading) {
    return (
      <Card className="bg-gray-50 border-dashed border-gray-300">
        <CardContent className="py-8">
          <div className="flex flex-col items-center justify-center space-y-2">
            <div className="h-8 w-8 rounded-full bg-gray-200 animate-pulse"></div>
            <div className="h-6 w-40 bg-gray-200 rounded animate-pulse"></div>
            <div className="h-4 w-24 bg-gray-200 rounded animate-pulse"></div>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!prediction || prediction === "No symptoms selected") {
    return (
      <Card className="bg-gray-50 border-dashed border-gray-300">
        <CardContent className="py-8">
          <div className="flex flex-col items-center justify-center space-y-2 text-gray-500">
            <CircleArrowUp className="h-8 w-8" />
            <p className="text-lg font-semibold">No prediction available</p>
            <p className="text-sm">Select symptoms above to generate a prediction</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Determine confidence level text and color
  let confidenceText = "Low";
  let confidenceColor = "text-yellow-600";
  
  if (confidence >= 70) {
    confidenceText = "High";
    confidenceColor = "text-green-600";
  } else if (confidence >= 40) {
    confidenceText = "Moderate";
    confidenceColor = "text-blue-600";
  }

  return (
    <Card className="bg-blue-50 border-blue-200">
      <CardHeader className="pb-2">
        <CardTitle className="text-center">Prediction Result</CardTitle>
        <CardDescription className="text-center">
          Based on your selected symptoms
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col items-center justify-center py-4">
          <div className="text-2xl font-bold text-blue-800 mb-2">{prediction}</div>
          <div className={`font-medium ${confidenceColor}`}>
            Confidence: {confidenceText} ({confidence.toFixed(1)}%)
          </div>
          <div className="mt-4 text-sm text-gray-600 text-center max-w-md">
            <p className="mb-2">This is a preliminary analysis based on the symptoms you selected.</p>
            <p className="font-medium">Always consult a qualified healthcare provider for proper diagnosis and treatment.</p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default PredictionResult;

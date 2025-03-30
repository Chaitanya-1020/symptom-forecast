
import React from 'react';
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

const Disclaimer = () => {
  return (
    <Alert className="bg-amber-50 border-amber-200 mb-6">
      <AlertTitle className="text-amber-800">Important Medical Disclaimer</AlertTitle>
      <AlertDescription className="text-amber-700">
        <p className="text-sm">
          This tool is for educational purposes only and is not intended to replace professional medical advice, 
          diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider 
          with any questions you may have regarding a medical condition.
        </p>
      </AlertDescription>
    </Alert>
  );
};

export default Disclaimer;

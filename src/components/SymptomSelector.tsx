
import React from 'react';
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";

interface SymptomSelectorProps {
  availableSymptoms: string[];
  selectedSymptoms: string[];
  onSymptomToggle: (symptom: string) => void;
  onClear: () => void;
}

const SymptomSelector = ({ 
  availableSymptoms, 
  selectedSymptoms, 
  onSymptomToggle, 
  onClear 
}: SymptomSelectorProps) => {
  return (
    <div className="bg-white shadow-md rounded-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold text-gray-800">Select Your Symptoms</h2>
        <Button 
          variant="outline" 
          size="sm" 
          onClick={onClear}
          disabled={selectedSymptoms.length === 0}
        >
          Clear All
        </Button>
      </div>
      
      <div className="mb-4">
        <div className="bg-blue-50 text-blue-800 p-3 rounded-md text-sm">
          {selectedSymptoms.length === 0 ? (
            "Please select all symptoms you're experiencing for the most accurate prediction."
          ) : (
            <>Selected: <strong>{selectedSymptoms.length}</strong> symptoms</>
          )}
        </div>
      </div>
      
      <ScrollArea className="h-[300px] pr-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {availableSymptoms.map((symptom) => (
            <div key={symptom} className="flex items-start space-x-2 p-2 hover:bg-gray-50 rounded">
              <Checkbox 
                id={symptom} 
                checked={selectedSymptoms.includes(symptom)}
                onCheckedChange={() => onSymptomToggle(symptom)}
              />
              <Label 
                htmlFor={symptom} 
                className="cursor-pointer text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
              >
                {symptom}
              </Label>
            </div>
          ))}
        </div>
      </ScrollArea>
    </div>
  );
};

export default SymptomSelector;

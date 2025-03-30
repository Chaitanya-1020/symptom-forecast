
import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { InfoIcon, FileSpreadsheet, BarChart4, LineChart } from 'lucide-react';
import Header from '@/components/Header';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, 
         PieChart, Pie, Cell } from 'recharts';

const Analysis = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [statistics, setStatistics] = useState<any>(null);
  const [symptomFrequency, setSymptomFrequency] = useState<any[]>([]);
  const [diseasePrevalence, setDiseasePrevalence] = useState<any[]>([]);
  
  useEffect(() => {
    // This would normally load data from the Python analysis
    // For now, we'll use simulated data
    
    // Simulate loading time
    const timer = setTimeout(() => {
      try {
        // Sample statistics data (would come from the Python backend)
        setStatistics({
          disease_count: 12,
          unique_symptoms: 45,
          top_symptoms: ["Cough", "Fever", "Fatigue", "Headache", "Nausea"],
          mean_prevalence: 47.5,
          min_prevalence: 25,
          max_prevalence: 78
        });
        
        // Sample symptom frequency data
        setSymptomFrequency([
          { name: "Cough", frequency: 4 },
          { name: "Fever", frequency: 3 },
          { name: "Headache", frequency: 3 },
          { name: "Fatigue", frequency: 3 },
          { name: "Nausea", frequency: 2 },
          { name: "Vomiting", frequency: 2 },
          { name: "Shortness of Breath", frequency: 2 },
          { name: "Sore Throat", frequency: 1 },
          { name: "Body Aches", frequency: 1 },
          { name: "Diarrhea", frequency: 1 }
        ]);
        
        // Sample disease prevalence data
        setDiseasePrevalence([
          { name: "Common Cold", value: 78 },
          { name: "Influenza", value: 65 },
          { name: "COVID-19", value: 60 },
          { name: "Gastroenteritis", value: 55 },
          { name: "Food Poisoning", value: 50 },
          { name: "Allergic Rhinitis", value: 45 },
          { name: "Migraine", value: 42 },
          { name: "Sinusitis", value: 40 }
        ]);
        
        setLoading(false);
      } catch (err) {
        setError("Failed to load analysis data");
        setLoading(false);
      }
    }, 1000);
    
    return () => clearTimeout(timer);
  }, []);
  
  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#4ECDC4', '#C7F464', '#FF6B6B'];
  
  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50">
        <Header />
        <main className="container mx-auto px-4 py-8">
          <div className="flex items-center justify-center h-64">
            <div className="text-center">
              <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto"></div>
              <p className="mt-4 text-lg text-gray-700">Loading analysis results...</p>
            </div>
          </div>
        </main>
      </div>
    );
  }
  
  if (error) {
    return (
      <div className="min-h-screen bg-gray-50">
        <Header />
        <main className="container mx-auto px-4 py-8">
          <Alert variant="destructive" className="mb-6">
            <InfoIcon className="h-4 w-4" />
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>
              {error}. Please try again or check your data source.
            </AlertDescription>
          </Alert>
        </main>
      </div>
    );
  }
  
  return (
    <div className="min-h-screen bg-gray-50">
      <Header />
      
      <main className="container mx-auto px-4 py-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-800 mb-2">Medical Data Analysis</h1>
          <p className="text-gray-600">
            Exploratory Data Analysis results from our medical dataset
          </p>
        </div>
        
        {statistics && (
          <Card className="mb-8">
            <CardHeader>
              <CardTitle className="flex items-center">
                <FileSpreadsheet className="mr-2 h-5 w-5" />
                Dataset Statistics
              </CardTitle>
              <CardDescription>
                Summary statistics from the medical dataset
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-blue-50 p-4 rounded-lg">
                  <p className="text-sm text-blue-800">Total Diseases</p>
                  <p className="text-3xl font-bold text-blue-600">{statistics.disease_count}</p>
                </div>
                <div className="bg-green-50 p-4 rounded-lg">
                  <p className="text-sm text-green-800">Unique Symptoms</p>
                  <p className="text-3xl font-bold text-green-600">{statistics.unique_symptoms}</p>
                </div>
                <div className="bg-yellow-50 p-4 rounded-lg">
                  <p className="text-sm text-yellow-800">Avg. Prevalence</p>
                  <p className="text-3xl font-bold text-yellow-600">{statistics.mean_prevalence.toFixed(1)}%</p>
                </div>
              </div>
              
              <div className="mt-6">
                <h3 className="text-lg font-medium mb-2">Top Symptoms</h3>
                <div className="flex flex-wrap gap-2">
                  {statistics.top_symptoms.map((symptom: string) => (
                    <span key={symptom} className="bg-gray-100 text-gray-800 px-3 py-1 rounded-full text-sm">
                      {symptom}
                    </span>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        )}
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <BarChart4 className="mr-2 h-5 w-5" />
                Symptom Frequency Analysis
              </CardTitle>
              <CardDescription>
                Distribution of symptoms across all diseases
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="bar">
                <TabsList className="mb-4">
                  <TabsTrigger value="bar">Bar Chart</TabsTrigger>
                  <TabsTrigger value="pie">Pie Chart</TabsTrigger>
                </TabsList>
                
                <TabsContent value="bar">
                  <div className="h-[300px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={symptomFrequency}
                        margin={{ top: 5, right: 30, left: 20, bottom: 70 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis 
                          dataKey="name" 
                          tick={{ fontSize: 12 }}
                          angle={-45}
                          textAnchor="end"
                          height={70}
                        />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Bar dataKey="frequency" fill="#8884d8" name="Frequency" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </TabsContent>
                
                <TabsContent value="pie">
                  <div className="h-[300px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={symptomFrequency}
                          cx="50%"
                          cy="50%"
                          labelLine={false}
                          label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                          outerRadius={100}
                          fill="#8884d8"
                          dataKey="frequency"
                        >
                          {symptomFrequency.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                          ))}
                        </Pie>
                        <Tooltip formatter={(value) => [`${value}`, 'Frequency']} />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <LineChart className="mr-2 h-5 w-5" />
                Disease Prevalence
              </CardTitle>
              <CardDescription>
                Prevalence rates across different diseases
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="bar">
                <TabsList className="mb-4">
                  <TabsTrigger value="bar">Bar Chart</TabsTrigger>
                  <TabsTrigger value="pie">Pie Chart</TabsTrigger>
                </TabsList>
                
                <TabsContent value="bar">
                  <div className="h-[300px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={diseasePrevalence}
                        margin={{ top: 5, right: 30, left: 20, bottom: 70 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis 
                          dataKey="name" 
                          tick={{ fontSize: 12 }}
                          angle={-45}
                          textAnchor="end"
                          height={70}
                        />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Bar dataKey="value" fill="#4285F4" name="Prevalence (%)" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </TabsContent>
                
                <TabsContent value="pie">
                  <div className="h-[300px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={diseasePrevalence}
                          cx="50%"
                          cy="50%"
                          labelLine={false}
                          label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                          outerRadius={100}
                          fill="#8884d8"
                          dataKey="value"
                        >
                          {diseasePrevalence.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                          ))}
                        </Pie>
                        <Tooltip formatter={(value) => [`${value}%`, 'Prevalence']} />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>
        </div>
        
        <Card>
          <CardHeader>
            <CardTitle>Data Processing Workflow</CardTitle>
            <CardDescription>
              How the medical dataset was processed and analyzed
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-center">
              <div className="border rounded-md p-4 bg-gray-50">
                <div className="bg-blue-100 text-blue-800 w-8 h-8 rounded-full flex items-center justify-center mx-auto mb-2">1</div>
                <h3 className="font-medium">Data Loading</h3>
                <p className="text-sm text-gray-600">CSV data ingestion and parsing</p>
              </div>
              <div className="border rounded-md p-4 bg-gray-50">
                <div className="bg-blue-100 text-blue-800 w-8 h-8 rounded-full flex items-center justify-center mx-auto mb-2">2</div>
                <h3 className="font-medium">Cleaning</h3>
                <p className="text-sm text-gray-600">Handling missing values, standardizing formats</p>
              </div>
              <div className="border rounded-md p-4 bg-gray-50">
                <div className="bg-blue-100 text-blue-800 w-8 h-8 rounded-full flex items-center justify-center mx-auto mb-2">3</div>
                <h3 className="font-medium">Analysis</h3>
                <p className="text-sm text-gray-600">Statistical analysis and visualization</p>
              </div>
              <div className="border rounded-md p-4 bg-gray-50">
                <div className="bg-blue-100 text-blue-800 w-8 h-8 rounded-full flex items-center justify-center mx-auto mb-2">4</div>
                <h3 className="font-medium">Prediction</h3>
                <p className="text-sm text-gray-600">Machine learning model training</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </main>
    </div>
  );
};

export default Analysis;

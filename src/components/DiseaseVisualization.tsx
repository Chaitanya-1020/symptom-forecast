
import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { BarChart, XAxis, YAxis, CartesianGrid, Tooltip, Legend, Bar, PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';
import { ChartPie, ChartBar } from 'lucide-react';

interface DiseaseData {
  disease: string;
  probability: number;
  matchedSymptoms: number;
  totalSymptoms: number;
}

interface DiseaseVisualizationProps {
  data: DiseaseData[];
}

const DiseaseVisualization = ({ data }: DiseaseVisualizationProps) => {
  // Only take top 5 results for cleaner charts
  const topResults = data.slice(0, 5).filter(item => item.probability > 0);
  
  // Create data for the pie chart 
  const pieData = topResults.map(item => ({
    name: item.disease,
    value: item.probability
  }));

  // Colors for the charts
  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

  return (
    <Card className="mb-6">
      <CardHeader>
        <CardTitle>Disease Probability Analysis</CardTitle>
        <CardDescription>
          Visual representation of the most likely diagnoses based on your symptoms
        </CardDescription>
      </CardHeader>
      <CardContent>
        {topResults.length > 0 ? (
          <Tabs defaultValue="bar">
            <TabsList className="mb-4">
              <TabsTrigger value="bar" className="flex items-center">
                <ChartBar className="mr-2 h-4 w-4" /> Bar Chart
              </TabsTrigger>
              <TabsTrigger value="pie" className="flex items-center">
                <ChartPie className="mr-2 h-4 w-4" /> Pie Chart
              </TabsTrigger>
            </TabsList>
            
            <TabsContent value="bar" className="w-full">
              <ResponsiveContainer width="100%" height={300}>
                <BarChart
                  data={topResults}
                  margin={{ top: 20, right: 30, left: 20, bottom: 70 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="disease" 
                    angle={-45} 
                    textAnchor="end" 
                    height={70} 
                    tick={{ fontSize: 12 }}
                  />
                  <YAxis
                    label={{ value: 'Probability (%)', angle: -90, position: 'insideLeft' }}
                  />
                  <Tooltip 
                    formatter={(value) => [`${value}%`, 'Probability']}
                    labelFormatter={(label) => `Disease: ${label}`} 
                  />
                  <Legend />
                  <Bar 
                    dataKey="probability" 
                    fill="#4285F4" 
                    name="Probability" 
                    radius={[4, 4, 0, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            </TabsContent>
            
            <TabsContent value="pie">
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    labelLine={true}
                    outerRadius={100}
                    fill="#8884d8"
                    dataKey="value"
                    nameKey="name"
                    label={({ name, percent }) => 
                      `${name}: ${(percent * 100).toFixed(1)}%`
                    }
                  >
                    {pieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip 
                    formatter={(value) => [`${value}%`, 'Probability']}
                  />
                </PieChart>
              </ResponsiveContainer>
            </TabsContent>
          </Tabs>
        ) : (
          <div className="h-[300px] flex items-center justify-center text-gray-500">
            <p>Select symptoms to see probability charts</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default DiseaseVisualization;

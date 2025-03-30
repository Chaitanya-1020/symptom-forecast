
import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { 
  BarChart, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  Bar, 
  PieChart, 
  Pie, 
  Cell, 
  ResponsiveContainer,
  LineChart,
  Line,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Treemap,
  ScatterChart,
  Scatter,
  ZAxis
} from 'recharts';
import { 
  ChartPie, 
  ChartBar, 
  LineChart as LineChartIcon, 
  RadarIcon, 
  GridIcon, 
  ScatterChart as ScatterChartIcon 
} from 'lucide-react';

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
  // Only take top 8 results for cleaner charts (increased from 5)
  const topResults = data.slice(0, 8).filter(item => item.probability > 0);
  
  // Create data for the pie chart 
  const pieData = topResults.map(item => ({
    name: item.disease,
    value: item.probability
  }));

  // Create data for other charts
  const radarData = topResults.map(item => ({
    subject: item.disease,
    probability: item.probability,
    matchRate: (item.matchedSymptoms / item.totalSymptoms) * 100,
    fullMark: 100
  }));

  const scatterData = topResults.map(item => ({
    x: item.matchedSymptoms,
    y: item.probability,
    z: item.totalSymptoms,
    name: item.disease
  }));

  // Colors for the charts
  const COLORS = [
    '#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8',
    '#4ECDC4', '#C7F464', '#FF6B6B', '#845EC2', '#D65DB1'
  ];

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
                <ChartBar className="mr-2 h-4 w-4" /> Bar
              </TabsTrigger>
              <TabsTrigger value="pie" className="flex items-center">
                <ChartPie className="mr-2 h-4 w-4" /> Pie
              </TabsTrigger>
              <TabsTrigger value="line" className="flex items-center">
                <LineChartIcon className="mr-2 h-4 w-4" /> Line
              </TabsTrigger>
              <TabsTrigger value="radar" className="flex items-center">
                <RadarIcon className="mr-2 h-4 w-4" /> Radar
              </TabsTrigger>
              <TabsTrigger value="scatter" className="flex items-center">
                <ScatterChartIcon className="mr-2 h-4 w-4" /> Scatter
              </TabsTrigger>
              <TabsTrigger value="treemap" className="flex items-center">
                <GridIcon className="mr-2 h-4 w-4" /> Treemap
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

            <TabsContent value="line">
              <ResponsiveContainer width="100%" height={300}>
                <LineChart
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
                  <Line 
                    type="monotone" 
                    dataKey="probability" 
                    stroke="#8884d8" 
                    activeDot={{ r: 8 }} 
                    name="Probability"
                  />
                </LineChart>
              </ResponsiveContainer>
            </TabsContent>

            <TabsContent value="radar">
              <ResponsiveContainer width="100%" height={300}>
                <RadarChart cx="50%" cy="50%" outerRadius={100} data={radarData}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="subject" />
                  <PolarRadiusAxis angle={30} domain={[0, 100]} />
                  <Radar 
                    name="Probability" 
                    dataKey="probability" 
                    stroke="#8884d8" 
                    fill="#8884d8" 
                    fillOpacity={0.6} 
                  />
                  <Radar 
                    name="Match Rate (%)" 
                    dataKey="matchRate" 
                    stroke="#82ca9d" 
                    fill="#82ca9d" 
                    fillOpacity={0.6} 
                  />
                  <Legend />
                  <Tooltip/>
                </RadarChart>
              </ResponsiveContainer>
            </TabsContent>

            <TabsContent value="scatter">
              <ResponsiveContainer width="100%" height={300}>
                <ScatterChart
                  margin={{ top: 20, right: 30, left: 20, bottom: 70 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    type="number" 
                    dataKey="x" 
                    name="Matched Symptoms" 
                    label={{ 
                      value: 'Matched Symptoms', 
                      position: 'insideBottom', 
                      offset: -10 
                    }}
                  />
                  <YAxis 
                    type="number" 
                    dataKey="y" 
                    name="Probability" 
                    label={{ 
                      value: 'Probability (%)', 
                      angle: -90, 
                      position: 'insideLeft' 
                    }}
                  />
                  <ZAxis 
                    type="number" 
                    dataKey="z" 
                    range={[50, 400]} 
                    name="Total Symptoms" 
                  />
                  <Tooltip 
                    cursor={{ strokeDasharray: '3 3' }} 
                    formatter={(value, name, props) => {
                      if (name === 'Matched Symptoms') return [value, name];
                      if (name === 'Probability') return [`${value}%`, name];
                      return [value, name];
                    }}
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        const data = payload[0].payload;
                        return (
                          <div className="bg-white p-2 border rounded shadow-md">
                            <p className="font-bold">{data.name}</p>
                            <p>Matched: {data.x} symptoms</p>
                            <p>Total: {data.z} symptoms</p>
                            <p>Probability: {data.y}%</p>
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                  <Legend />
                  <Scatter 
                    name="Diseases" 
                    data={scatterData} 
                    fill="#8884d8"
                  >
                    {scatterData.map((entry, index) => (
                      <Cell 
                        key={`cell-${index}`} 
                        fill={COLORS[index % COLORS.length]} 
                      />
                    ))}
                  </Scatter>
                </ScatterChart>
              </ResponsiveContainer>
            </TabsContent>

            <TabsContent value="treemap">
              <ResponsiveContainer width="100%" height={300}>
                <Treemap
                  data={pieData}
                  dataKey="value"
                  nameKey="name"
                  stroke="#fff"
                  fill="#8884d8"
                  content={({ root, depth, x, y, width, height, index, payload, colors, rank, name }) => {
                    return (
                      <g>
                        <rect
                          x={x}
                          y={y}
                          width={width}
                          height={height}
                          style={{
                            fill: COLORS[index % COLORS.length],
                            stroke: '#fff',
                            strokeWidth: 2 / (depth + 1e-10),
                            strokeOpacity: 1 / (depth + 1e-10),
                          }}
                        />
                        {width > 30 && height > 30 && (
                          <text
                            x={x + width / 2}
                            y={y + height / 2}
                            textAnchor="middle"
                            fill="#fff"
                            fontSize={12}
                          >
                            {name}
                          </text>
                        )}
                        {width > 50 && height > 30 && (
                          <text
                            x={x + width / 2}
                            y={y + height / 2 + 12}
                            textAnchor="middle"
                            fill="#fff"
                            fontSize={10}
                          >
                            {`${pieData[index].value}%`}
                          </text>
                        )}
                      </g>
                    );
                  }}
                />
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

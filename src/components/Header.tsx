
import React from 'react';
import { Link } from 'react-router-dom';
import { ActivitySquare, ChartBar } from 'lucide-react';

const Header = () => {
  return (
    <header className="bg-blue-600 shadow-md">
      <div className="container mx-auto px-4 py-4">
        <div className="flex flex-col md:flex-row justify-between items-center">
          <div className="flex items-center mb-4 md:mb-0">
            <ActivitySquare className="h-7 w-7 text-white mr-2" />
            <h1 className="text-2xl font-bold text-white">MediPredictor</h1>
          </div>
          
          <nav>
            <ul className="flex space-x-6">
              <li>
                <Link 
                  to="/" 
                  className="text-white hover:text-blue-200 transition-colors font-medium flex items-center"
                >
                  <ActivitySquare className="h-4 w-4 mr-1" />
                  Predict
                </Link>
              </li>
              <li>
                <Link 
                  to="/analysis" 
                  className="text-white hover:text-blue-200 transition-colors font-medium flex items-center"
                >
                  <ChartBar className="h-4 w-4 mr-1" />
                  Analysis
                </Link>
              </li>
            </ul>
          </nav>
        </div>
      </div>
    </header>
  );
};

export default Header;

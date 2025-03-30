
import React from 'react';

const Header = () => {
  return (
    <header className="bg-gradient-to-r from-blue-600 to-blue-400 text-white py-4 px-6 shadow-md">
      <div className="container mx-auto flex flex-col items-center justify-between md:flex-row">
        <div className="flex items-center mb-4 md:mb-0">
          <h1 className="text-2xl font-bold">MediPredictor</h1>
          <span className="ml-2 bg-white text-blue-600 px-2 py-1 rounded-md text-xs font-medium">BETA</span>
        </div>
        <p className="text-sm opacity-90 text-center md:text-right">
          Predictive disease analysis based on symptoms
        </p>
      </div>
    </header>
  );
};

export default Header;

import React from 'react';
import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import LoadPreprocess from './components/LoadPreprocess';
import TrainLR from './components/TrainLR';
import ForecastARIMA from './components/ForecastARIMA';
import OptimizePortfolio from './components/OptimizePortfolio';

function App() {
  return (
    <BrowserRouter>
      <div>
        <nav>
          <ul>
            <li><Link to="/load-preprocess">Load & Preprocess Data</Link></li>
            <li><Link to="/train-lr">Train Linear Regression</Link></li>
            <li><Link to="/forecast-arima">Forecast ARIMA</Link></li>
            <li><Link to="/optimize-portfolio">Optimize Portfolio</Link></li>
          </ul>
        </nav>
        <Routes>
          <Route path="/load-preprocess" element={<LoadPreprocess />} />
          <Route path="/train-lr" element={<TrainLR />} />
          <Route path="/forecast-arima" element={<ForecastARIMA />} />
          <Route path="/optimize-portfolio" element={<OptimizePortfolio />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}

export default App;

import React, { useEffect } from 'react';
import { HashRouter as Router, Routes, Route } from 'react-router-dom';
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import Header from './components/Header';
import Footer from './components/Footer';
import Home from './pages/Home';
import Results from './pages/Results';
import Doctors from './pages/Doctors';
import { AnimatePresence } from 'framer-motion';
import { API_CONFIG, APP_CONFIG } from './config';

function App() {
  // Check backend connectivity on app start
  useEffect(() => {
    const checkBackendStatus = async () => {
      try {
        const response = await fetch(API_CONFIG.getUrl('HEALTH'), { 
          method: 'GET',
          headers: { 'Accept': 'application/json' },
          signal: AbortSignal.timeout(5000) // 5 second timeout
        });
        
        if (response.ok) {
          console.log('✓ Backend connection established');
        } else {
          console.warn('⚠ Backend returned non-200 status:', response.status);
        }
      } catch (error) {
        console.error('✖ Backend connection failed:', error.message);
      }
    };
    
    checkBackendStatus();
  }, []);

  return (
    <Router>
      <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-blue-50 flex flex-col">
        {APP_CONFIG.SHOW_DEMO_BANNER && (
          <div className="bg-gradient-to-r from-amber-500 to-amber-600 text-white text-center py-2 px-4 shadow-md">
            <strong>Demo Mode:</strong> Running with simulated data. Backend features are limited on GitHub Pages.
            {' '}
            <a 
              href={APP_CONFIG.REPO_URL} 
              className="underline font-medium hover:text-white/80"
              target="_blank" 
              rel="noopener noreferrer"
            >
              View on GitHub
            </a>
          </div>
        )}
        <Header />
        <main className="flex-grow">
          <AnimatePresence mode="wait">
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/results" element={<Results />} />
              <Route path="/doctors" element={<Doctors />} />
            </Routes>
          </AnimatePresence>
        </main>
        <ToastContainer 
          position="top-right" 
          autoClose={5000}
          hideProgressBar={false}
          newestOnTop
          closeOnClick
          rtl={false}
          pauseOnFocusLoss
          draggable
          pauseOnHover
          theme="light" 
        />
        <Footer />
      </div>
    </Router>
  );
}

export default App;

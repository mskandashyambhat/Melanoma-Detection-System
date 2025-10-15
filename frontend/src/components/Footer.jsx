import React from 'react';
import { FaGithub, FaLinkedin, FaEnvelope } from 'react-icons/fa';
import { Link } from 'react-router-dom';

const Footer = () => {
  const currentYear = new Date().getFullYear();
  
  return (
    <footer className="bg-white border-t mt-16">
      <div className="container mx-auto px-6 py-8">
        <div className="flex flex-col items-center md:flex-row md:justify-between">
          <div className="mb-4 md:mb-0">
            <Link to="/" className="flex items-center">
              <h3 className="text-lg font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
                Melanoma Detection System
              </h3>
            </Link>
            <p className="text-sm text-gray-500 mt-1">
              Advanced skin lesion analysis
            </p>
          </div>
          
          <div className="flex space-x-4">
            <a 
              href="#" 
              className="text-gray-500 hover:text-indigo-600 transition"
              aria-label="Github"
            >
              <FaGithub className="h-5 w-5" />
            </a>
            <a 
              href="#" 
              className="text-gray-500 hover:text-indigo-600 transition"
              aria-label="LinkedIn"
            >
              <FaLinkedin className="h-5 w-5" />
            </a>
            <a 
              href="mailto:contact@example.com" 
              className="text-gray-500 hover:text-indigo-600 transition"
              aria-label="Email"
            >
              <FaEnvelope className="h-5 w-5" />
            </a>
          </div>
        </div>
        
        <div className="border-t mt-6 pt-6">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <p className="text-sm text-gray-600 mb-4 md:mb-0">
              developed by <span className="font-semibold">M Skanda Shyam Bhat</span>, <span className="font-semibold">Kaushik</span>, <span className="font-semibold">Deepasree K</span>, <span className="font-semibold">Paavani K</span> @{currentYear}
            </p>
            
            <div className="flex items-center space-x-4">
              <Link to="/" className="text-sm text-gray-500 hover:text-indigo-600 transition">
                Privacy Policy
              </Link>
              <Link to="/" className="text-sm text-gray-500 hover:text-indigo-600 transition">
                Terms of Service
              </Link>
            </div>
          </div>
        </div>
        
        <div className="mt-6 text-center">
          <p className="text-xs text-gray-400">
            Disclaimer: This application is designed for educational purposes only and should not replace professional medical diagnosis.
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;

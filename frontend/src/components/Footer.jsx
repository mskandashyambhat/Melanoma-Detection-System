import React from 'react';
import { FaGithub, FaHeart } from 'react-icons/fa';

const Footer = () => {
  return (
    <footer className="bg-gradient-to-r from-slate-700 to-blue-800 text-white py-8 mt-auto">
      <div className="container mx-auto px-6">
        <div className="flex flex-col md:flex-row justify-between items-center">
          <div className="mb-4 md:mb-0">
            <p className="text-sm">
              Designed @2025 by M Skanda Shyam Bhat, Deepasree K, Kaushik, Paavani K
            </p>
          </div>
          
          <div className="flex items-center space-x-4">
            <a 
              href="https://github.com/mskandashyambhat/Melanoma-Detection-System" 
              target="_blank" 
              rel="noopener noreferrer"
              className="flex items-center text-sm hover:text-blue-200 transition-colors"
            >
              <FaGithub className="mr-2" />
              View on GitHub
            </a>
          </div>
        </div>
        

      </div>
    </footer>
  );
};

export default Footer;

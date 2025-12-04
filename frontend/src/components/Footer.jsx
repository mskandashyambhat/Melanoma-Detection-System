import React from 'react';
import { FaGithub, FaHeart } from 'react-icons/fa';

const Footer = () => {
  return (
    <footer className="bg-white dark:bg-black border-t border-gray-200 dark:border-gray-800 text-gray-700 dark:text-gray-300 py-8 mt-auto">
      <div className="container mx-auto px-6">
        <div className="flex flex-col md:flex-row justify-between items-center">
          <div className="mb-4 md:mb-0">
            <p className="text-sm">
              Designed @2025 by <span className="text-cyan-600 dark:text-cyan-400 font-medium">M Skanda Shyam Bhat</span>, Deepasree K, Kaushik, Paavani K
            </p>
          </div>
          
          <div className="flex items-center space-x-4">
            <a 
              href="https://github.com/mskandashyambhat/Melanoma-Detection-System" 
              target="_blank" 
              rel="noopener noreferrer"
              className="flex items-center text-sm hover:text-cyan-600 dark:hover:text-cyan-400 transition-colors group"
            >
              <FaGithub className="mr-2 group-hover:scale-110 transition-transform" />
              View on GitHub
            </a>
          </div>
        </div>
        

      </div>
    </footer>
  );
};

export default Footer;

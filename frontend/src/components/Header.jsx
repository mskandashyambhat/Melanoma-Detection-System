import React, { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { FaHeartbeat, FaStethoscope, FaBars, FaTimes } from 'react-icons/fa';
import { MdDashboard, MdHome } from 'react-icons/md';
import { motion } from 'framer-motion';
import { useTheme } from '../context/ThemeContext';

const Header = () => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const location = useLocation();
  const { isDark, toggleTheme } = useTheme();
  
  // Handle scrolling effect
  useEffect(() => {
    const handleScroll = () => {
      if (window.scrollY > 10) {
        setScrolled(true);
      } else {
        setScrolled(false);
      }
    };
    
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);
  
  // Close mobile menu when location changes
  useEffect(() => {
    setMobileMenuOpen(false);
  }, [location.pathname]);
  
  // Nav link animation
  const navLinkVariants = {
    initial: { y: -5, opacity: 0 },
    animate: { y: 0, opacity: 1 },
    hover: { scale: 1.05, color: '#1d4ed8' }
  };
  
  // Active link determination
  const isActive = (path) => location.pathname === path;
  
  return (
    <header 
      className={`sticky top-0 z-50 transition-all duration-300 ${
        scrolled 
          ? 'bg-white/95 dark:bg-black/95 backdrop-blur-md shadow-2xl shadow-cyan-500/10 border-b border-gray-200 dark:border-gray-800' 
          : 'bg-white dark:bg-black shadow-lg shadow-cyan-500/5 border-b border-gray-300 dark:border-gray-900'
      }`}
    >
      <div className="container mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          <Link to="/" className="flex items-center space-x-3 group">
            <div>
              <h1 className="text-2xl font-bold bg-gradient-to-r from-cyan-600 via-blue-600 to-purple-600 dark:from-cyan-400 dark:via-blue-500 dark:to-purple-500 bg-clip-text text-transparent hover:text-glow transition-all">
                Melanoma Detection System
              </h1>
              <p className="text-sm text-cyan-600/70 dark:text-cyan-400/70">Medical Diagnostics</p>
            </div>
          </Link>
          
          {/* Mobile Menu Button */}
          <div className="md:hidden">
            <button 
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="text-gray-700 dark:text-gray-300 hover:text-cyan-600 dark:hover:text-cyan-400 focus:outline-none transition"
            >
              {mobileMenuOpen ? 
                <FaTimes className="h-6 w-6" /> : 
                <FaBars className="h-6 w-6" />
              }
            </button>
          </div>
          
          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center space-x-8">
            <motion.div
              variants={navLinkVariants}
              initial="initial"
              animate="animate"
              transition={{ delay: 0.1 }}
            >
              <Link 
                to="/"
                className={`flex items-center space-x-1 px-3 py-2 rounded-md transition ${
                  isActive('/') 
                    ? 'text-cyan-600 dark:text-cyan-400 font-medium bg-cyan-500/10 shadow-lg shadow-cyan-500/20' 
                    : 'text-gray-700 dark:text-gray-300 hover:text-cyan-600 dark:hover:text-cyan-400'
                }`}
              >
                <MdHome className="text-lg" />
                <span>Home</span>
              </Link>
            </motion.div>
            
            <motion.div
              variants={navLinkVariants}
              initial="initial"
              animate="animate"
              transition={{ delay: 0.2 }}
            >
              <Link 
                to="/results"
                className={`flex items-center space-x-1 px-3 py-2 rounded-md transition ${
                  isActive('/results') 
                    ? 'text-cyan-600 dark:text-cyan-400 font-medium bg-cyan-500/10 shadow-lg shadow-cyan-500/20' 
                    : 'text-gray-700 dark:text-gray-300 hover:text-cyan-600 dark:hover:text-cyan-400'
                }`}
              >
                <MdDashboard className="text-lg" />
                <span>Results</span>
              </Link>
            </motion.div>
            
            <motion.div
              variants={navLinkVariants}
              initial="initial"
              animate="animate"
              transition={{ delay: 0.3 }}
            >
              <Link 
                to="/doctors"
                className={`flex items-center space-x-1 px-3 py-2 rounded-md transition ${
                  isActive('/doctors') 
                    ? 'text-cyan-600 dark:text-cyan-400 font-medium bg-cyan-500/10 shadow-lg shadow-cyan-500/20' 
                    : 'text-gray-700 dark:text-gray-300 hover:text-cyan-600 dark:hover:text-cyan-400'
                }`}
              >
                <FaStethoscope className="text-lg" />
                <span>Doctors</span>
              </Link>
            </motion.div>
            
            {/* Theme Toggle */}
            <motion.div
              variants={navLinkVariants}
              initial="initial"
              animate="animate"
              transition={{ delay: 0.4 }}
            >
              <button
                onClick={toggleTheme}
                className="relative px-4 py-2 rounded-md bg-gray-200 dark:bg-gray-800 hover:bg-gray-300 dark:hover:bg-gray-700 transition text-sm font-medium overflow-hidden"
              >
                <span className={`transition-all duration-300 ${isDark ? 'text-cyan-400' : 'text-gray-500 dark:text-gray-500'}`}>
                  Dark
                </span>
                <span className="mx-2 text-gray-400 dark:text-gray-600">|</span>
                <span className={`transition-all duration-300 ${!isDark ? 'text-cyan-600' : 'text-gray-500 dark:text-gray-500'}`}>
                  Light
                </span>
              </button>
            </motion.div>
          </nav>
        </div>
        
        {/* Mobile Navigation */}
        {mobileMenuOpen && (
          <motion.div 
            className="md:hidden mt-4 py-2 border-t border-gray-200 dark:border-gray-800 bg-gray-100/50 dark:bg-gray-900/50"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3 }}
          >
            <Link 
              to="/"
              className={`flex items-center space-x-2 py-3 px-4 rounded-md ${
                isActive('/') 
                  ? 'text-cyan-600 dark:text-cyan-400 font-medium bg-cyan-500/10' 
                  : 'text-gray-700 dark:text-gray-300'
              }`}
            >
              <MdHome className="text-lg" />
              <span>Home</span>
            </Link>
            <Link 
              to="/results"
              className={`flex items-center space-x-2 py-3 px-4 rounded-md ${
                isActive('/results') 
                  ? 'text-cyan-600 dark:text-cyan-400 font-medium bg-cyan-500/10' 
                  : 'text-gray-700 dark:text-gray-300'
              }`}
            >
              <MdDashboard className="text-lg" />
              <span>Results</span>
            </Link>
            <Link 
              to="/doctors"
              className={`flex items-center space-x-2 py-3 px-4 rounded-md ${
                isActive('/doctors') 
                  ? 'text-cyan-600 dark:text-cyan-400 font-medium bg-cyan-500/10' 
                  : 'text-gray-700 dark:text-gray-300'
              }`}
            >
              <FaStethoscope className="text-lg" />
              <span>Doctors</span>
            </Link>
            
            {/* Mobile Theme Toggle */}
            <button
              onClick={toggleTheme}
              className="py-3 px-4 rounded-md bg-gray-200 dark:bg-gray-800 hover:bg-gray-300 dark:hover:bg-gray-700 transition text-sm font-medium"
            >
              <span className={`transition-all duration-300 ${isDark ? 'text-cyan-400' : 'text-gray-500'}`}>
                Dark
              </span>
              <span className="mx-2 text-gray-400 dark:text-gray-600">|</span>
              <span className={`transition-all duration-300 ${!isDark ? 'text-cyan-600' : 'text-gray-500'}`}>
                Light
              </span>
            </button>
          </motion.div>
        )}
      </div>
    </header>
  );
};

export default Header;

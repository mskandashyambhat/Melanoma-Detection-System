import React, { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { FaHeartbeat, FaStethoscope, FaBars, FaTimes } from 'react-icons/fa';
import { MdDashboard, MdHome } from 'react-icons/md';
import { motion } from 'framer-motion';

const Header = () => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const location = useLocation();
  
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
    hover: { scale: 1.05, color: '#4F46E5' }
  };
  
  // Active link determination
  const isActive = (path) => location.pathname === path;
  
  return (
    <header 
      className={`sticky top-0 z-50 transition-all duration-300 ${
        scrolled ? 'bg-white/95 backdrop-blur-md shadow-lg' : 'bg-white shadow-md'
      }`}
    >
      <div className="container mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          <Link to="/" className="flex items-center space-x-3 group">
            <motion.div 
              className="bg-gradient-to-r from-indigo-600 to-purple-600 p-2 rounded-lg"
              whileHover={{ scale: 1.05, rotate: 5 }}
              transition={{ type: "spring", stiffness: 400 }}
            >
              <FaHeartbeat className="text-white text-3xl" />
            </motion.div>
            <div>
              <h1 className="text-2xl font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
                Melanoma Detection System
              </h1>
              <p className="text-sm text-gray-600">Medical Diagnostics</p>
            </div>
          </Link>
          
          {/* Mobile Menu Button */}
          <div className="md:hidden">
            <button 
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="text-gray-600 hover:text-indigo-600 focus:outline-none transition"
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
                  isActive('/') ? 'text-indigo-600 font-medium bg-indigo-50' : 'text-gray-700 hover:text-indigo-600'
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
                  isActive('/results') ? 'text-indigo-600 font-medium bg-indigo-50' : 'text-gray-700 hover:text-indigo-600'
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
                  isActive('/doctors') ? 'text-indigo-600 font-medium bg-indigo-50' : 'text-gray-700 hover:text-indigo-600'
                }`}
              >
                <FaStethoscope className="text-lg" />
                <span>Doctors</span>
              </Link>
            </motion.div>
          </nav>
        </div>
        
        {/* Mobile Navigation */}
        {mobileMenuOpen && (
          <motion.div 
            className="md:hidden mt-4 py-2 border-t"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3 }}
          >
            <Link 
              to="/"
              className={`flex items-center space-x-2 py-3 px-4 rounded-md ${
                isActive('/') ? 'text-indigo-600 font-medium bg-indigo-50' : 'text-gray-700'
              }`}
            >
              <MdHome className="text-lg" />
              <span>Home</span>
            </Link>
            <Link 
              to="/results"
              className={`flex items-center space-x-2 py-3 px-4 rounded-md ${
                isActive('/results') ? 'text-indigo-600 font-medium bg-indigo-50' : 'text-gray-700'
              }`}
            >
              <MdDashboard className="text-lg" />
              <span>Results</span>
            </Link>
            <Link 
              to="/doctors"
              className={`flex items-center space-x-2 py-3 px-4 rounded-md ${
                isActive('/doctors') ? 'text-indigo-600 font-medium bg-indigo-50' : 'text-gray-700'
              }`}
            >
              <FaStethoscope className="text-lg" />
              <span>Doctors</span>
            </Link>
          </motion.div>
        )}
      </div>
    </header>
  );
};

export default Header;

import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { FaCloudUploadAlt, FaImage, FaTimes, FaHeartbeat } from 'react-icons/fa';
import { motion } from 'framer-motion';
import { toast } from 'react-toastify';
import axios from 'axios';
import { API_CONFIG, APP_CONFIG } from '../config';
import { mockService } from '../mockService';

const Home = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const fileInputRef = useRef(null);
  const navigate = useNavigate();

  // Clear previous results when Home page loads
  useEffect(() => {
    localStorage.removeItem('analysisResult');
    localStorage.removeItem('reportFilename');
    localStorage.removeItem('patientInfo');
  }, []);

  const handleImageSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      if (file.size > 16 * 1024 * 1024) {
        toast.error('File size must be less than 16MB');
        return;
      }
      
      if (!['image/png', 'image/jpeg', 'image/jpg'].includes(file.type)) {
        toast.error('Please upload PNG, JPG, or JPEG image');
        return;
      }
      
      setSelectedImage(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) {
      const fakeEvent = { target: { files: [file] } };
      handleImageSelect(fakeEvent);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const removeImage = () => {
    setSelectedImage(null);
    setImagePreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleAnalyze = async () => {
    if (!selectedImage) {
      toast.error('Please upload an image first');
      return;
    }

    setLoading(true);
    
    try {
      let analysisResult;
      
      // Use mock service for GitHub Pages demo mode
      if (API_CONFIG.USE_MOCK) {
        analysisResult = await mockService.predictImage();
        
        // Store results and navigate
        localStorage.setItem('analysisResult', JSON.stringify(analysisResult));
      } else {
        // Normal API call to backend
        const formData = new FormData();
        formData.append('image', selectedImage);
        
        const response = await axios.post(API_CONFIG.getUrl('PREDICT'), formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
          timeout: 15000
        });
        
        // Store results and navigate
        localStorage.setItem('analysisResult', JSON.stringify({
          ...response.data,
          imagePreview: imagePreview
        }));
      }
      
      toast.success('Analysis completed successfully!');
      navigate('/results');
    } catch (error) {
      console.error('Error:', error);
      const msg = error.response?.data?.error || error.message || 'Failed to analyze image.';
      
      if (API_CONFIG.USE_MOCK) {
        toast.error('Error in demo mode. Please try again.');
      } else {
        toast.error(`${msg} Please ensure the backend is running and try a PNG/JPG under 16MB.`);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mx-auto px-6 py-12">
      {/* Hero Section */}
      <motion.div 
        className="text-center mb-16 max-w-4xl mx-auto"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
       
        <h1 className="text-5xl font-bold mb-6 bg-gradient-to-r from-slate-700 to-blue-700 bg-clip-text text-transparent">
          Advanced Melanoma Detection 
        </h1>
        <p className="text-xl text-gray-600 max-w-3xl mx-auto mb-8">
          Upload an image of your skin lesion for instant analysis
        </p>
        
        <div className="flex flex-wrap justify-center gap-6 mt-8">
          <div className="flex items-center bg-white rounded-lg shadow-md px-4 py-3">
            <div className="bg-indigo-100 p-2 rounded-full">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-indigo-600" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z" clipRule="evenodd" />
              </svg>
            </div>
            <span className="ml-2 text-gray-700">Results in seconds</span>
          </div>
          
          <div className="flex items-center bg-white rounded-lg shadow-md px-4 py-3">
            <div className="bg-blue-100 p-2 rounded-full">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-blue-700" viewBox="0 0 20 20" fill="currentColor">
                <path d="M9 6a3 3 0 11-6 0 3 3 0 016 0zM17 6a3 3 0 11-6 0 3 3 0 016 0zM12.93 17c.046-.327.07-.66.07-1a6.97 6.97 0 00-1.5-4.33A5 5 0 0119 16v1h-6.07zM6 11a5 5 0 015 5v1H1v-1a5 5 0 015-5z" />
              </svg>
            </div>
            <span className="ml-2 text-gray-700">Expert dermatologist consults</span>
          </div>
          
          <div className="flex items-center bg-white rounded-lg shadow-md px-4 py-3">
            <div className="bg-rose-100 p-2 rounded-full">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-rose-600" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M9 2a1 1 0 000 2h2a1 1 0 100-2H9z" clipRule="evenodd" />
                <path fillRule="evenodd" d="M4 5a2 2 0 012-2 3 3 0 003 3h2a3 3 0 003-3 2 2 0 012 2v11a2 2 0 01-2 2H6a2 2 0 01-2-2V5zm3 4a1 1 0 000 2h.01a1 1 0 100-2H7zm3 0a1 1 0 000 2h3a1 1 0 100-2h-3zm-3 4a1 1 0 100 2h.01a1 1 0 100-2H7zm3 0a1 1 0 100 2h3a1 1 0 100-2h-3z" clipRule="evenodd" />
              </svg>
            </div>
            <span className="ml-2 text-gray-700">
              Melanoma Detection
            </span>
          </div>
        </div>
      </motion.div>

      {/* Main Upload Section */}
      <motion.div 
        className="max-w-5xl mx-auto"
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5, delay: 0.2 }}
      >
        <div className="card bg-white rounded-2xl shadow-xl p-10 hover:shadow-2xl transition-shadow duration-300 border border-gray-100">
          {!imagePreview ? (
            <div
              className="upload-zone border-3 border-dashed border-blue-300 rounded-xl p-12 text-center cursor-pointer hover:border-blue-600 transition-all duration-300 bg-gradient-to-br from-white to-blue-50"
              onClick={() => fileInputRef.current.click()}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
            >
              <motion.div
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ type: "spring", stiffness: 300 }}
              >
                <div className="bg-blue-100 w-24 h-24 rounded-full flex items-center justify-center mx-auto mb-6">
                  <FaCloudUploadAlt className="text-5xl text-blue-700" />
                </div>
                                <h3 className="text-2xl font-semibold mb-4 bg-gradient-to-r from-slate-700 to-blue-700 bg-clip-text text-transparent">
                  Upload Your Image
                </h3>
                <p className="text-gray-600 mb-6 text-lg max-w-xl mx-auto">
                  Drag and drop your image here, or click to browse files from your device
                </p>
                <div className="bg-blue-50 py-3 px-5 rounded-lg inline-block mb-4">
                  <p className="text-sm text-blue-900 font-medium flex items-center">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M2.166 4.999A11.954 11.954 0 0010 1.944 11.954 11.954 0 0017.834 5c.11.65.166 1.32.166 2.001 0 5.225-3.34 9.67-8 11.317C5.34 16.67 2 12.225 2 7c0-.682.057-1.35.166-2.001zm11.541 3.708a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                    </svg>
                    Supported formats: PNG, JPG, JPEG (Max 16MB)
                  </p>
                </div>
                <button className="bg-blue-700 hover:bg-blue-800 text-white px-6 py-3 rounded-lg transition-colors duration-200 font-medium inline-flex items-center">
                  <FaImage className="mr-2" /> Browse Files
                </button>
              </motion.div>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/png,image/jpeg,image/jpg"
                onChange={handleImageSelect}
                className="hidden"
              />
            </div>
          ) : (
            <div className="relative">
              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.95 }}
                onClick={removeImage}
                className="absolute top-4 right-4 bg-red-500 text-white p-2 rounded-full hover:bg-red-600 transition shadow-md z-10"
                aria-label="Remove image"
              >
                <FaTimes />
              </motion.button>
              
              <div className="flex flex-col md:flex-row items-center gap-10">
                <div className="relative mb-8 md:mb-0 w-full max-w-md">
                  <div className="absolute inset-0 rounded-lg" style={{ 
                    background: 'radial-gradient(circle, rgba(99,102,241,0.1) 0%, rgba(147,51,234,0.05) 100%)' 
                  }}></div>
                  
                  <motion.div 
                    className="absolute inset-0 bg-gradient-to-r from-slate-600/20 to-blue-600/20 rounded-lg"
                    animate={{ opacity: [0, 0.5, 0] }}
                    transition={{ 
                      repeat: Infinity, 
                      duration: 2, 
                      ease: "easeInOut" 
                    }}
                  />
                  
                  <img
                    src={imagePreview}
                    alt="Selected skin lesion"
                    className="max-h-96 w-full object-contain rounded-lg shadow-lg"
                  />
                  
                  <div className="mt-3 text-center text-sm text-gray-500">
                    Selected image for analysis
                  </div>
                </div>
                
                <div className="w-full max-w-md">
                  <h3 className="text-xl font-semibold mb-4 text-slate-800">Ready for Analysis</h3>
                  
                  <div className="bg-blue-50 rounded-lg p-4 mb-6">
                    <div className="flex items-start">
                      <div className="bg-indigo-100 p-2 rounded-full">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-indigo-600" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                        </svg>
                      </div>
                      <p className="ml-3 text-sm text-indigo-700">
                        The analysis will include lesion classification, risk assessment, and segmentation using our ResNet50 and U-Net deep learning models.
                      </p>
                    </div>
                  </div>
                  
                  <motion.button
                    whileHover={{ scale: 1.03 }}
                    whileTap={{ scale: 0.97 }}
                    onClick={handleAnalyze}
                    disabled={loading}
                    className="w-full bg-gradient-to-r from-indigo-600 to-purple-600 text-white font-medium text-lg px-8 py-4 rounded-lg shadow-lg hover:shadow-xl transition-all duration-300 disabled:opacity-60 disabled:cursor-not-allowed"
                  >
                    {loading ? (
                      <motion.span 
                        className="flex items-center justify-center"
                        animate={{ opacity: [0.6, 1] }}
                        transition={{ repeat: Infinity, duration: 0.8, repeatType: "reverse" }}
                      >
                        <div className="animate-spin h-5 w-5 mr-3 border-2 border-white border-t-transparent rounded-full"></div>
                        Analyzing Image...
                      </motion.span>
                    ) : (
                      <span className="flex items-center justify-center">
                        <FaHeartbeat className="mr-2" />
                        Start Analysis
                      </span>
                    )}
                  </motion.button>
                  
                  <p className="text-sm text-gray-500 mt-3 text-center">
                    Analysis typically completes within 5-10 seconds
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      </motion.div>

      {/* Features Section */}
      <div className="mt-20">
        <motion.h2 
          className="text-3xl font-bold text-center mb-12 bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.3 }}
        >
          Advanced Features
        </motion.h2>
        
        <motion.div 
          className="grid md:grid-cols-2 gap-8 max-w-4xl mx-auto"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
        >
          <motion.div 
            className="bg-white rounded-xl shadow-lg p-8 hover:shadow-xl transition-all duration-300 hover:-translate-y-1 relative overflow-hidden"
            whileHover={{ scale: 1.03 }}
          >
            <div className="absolute top-0 left-0 w-full h-2 bg-gradient-to-r from-indigo-500 to-indigo-600"></div>
            <div className="bg-gradient-to-br from-indigo-50 to-indigo-100 w-20 h-20 rounded-full flex items-center justify-center mx-auto mb-6">
              <FaImage className="text-4xl text-indigo-600" />
            </div>
            <h3 className="text-xl font-semibold mb-4 text-center text-indigo-900">Advanced Detection</h3>
            <p className="text-gray-600 text-center leading-relaxed">
              Advanced CNN models (ResNet50 + UNet) for accurate skin condition identification with precise lesion segmentation
            </p>
            <div className="mt-6 flex justify-center">
              <span className="inline-flex items-center text-sm font-medium text-indigo-600">
                Learn more
                <svg className="ml-1 w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5l7 7-7 7"></path>
                </svg>
              </span>
            </div>
          </motion.div>

          <motion.div 
            className="bg-white rounded-xl shadow-lg p-8 hover:shadow-xl transition-all duration-300 hover:-translate-y-1 relative overflow-hidden"
            whileHover={{ scale: 1.03 }}
          >
            <div className="absolute top-0 left-0 w-full h-2 bg-gradient-to-r from-purple-500 to-purple-600"></div>
            <div className="bg-gradient-to-br from-purple-50 to-purple-100 w-20 h-20 rounded-full flex items-center justify-center mx-auto mb-6">
              <span className="text-3xl font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">97%+</span>
            </div>
            <h3 className="text-xl font-semibold mb-4 text-center text-indigo-900">High Accuracy</h3>
            <p className="text-gray-600 text-center leading-relaxed">
              Industry-leading accuracy in detecting melanoma and classifying multiple skin conditions for reliable results
            </p>
            <div className="mt-6 flex justify-center">
              <span className="inline-flex items-center text-sm font-medium text-purple-600">
                View studies
                <svg className="ml-1 w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5l7 7-7 7"></path>
                </svg>
              </span>
            </div>
          </motion.div>
        </motion.div>
      </div>

      {/* How It Works */}
      <motion.div
        className="mt-20" 
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.6, delay: 0.6 }}
      >
        <h2 className="text-3xl font-bold text-center mb-12 bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
          How It Works
        </h2>
        
        <div className="flex flex-col md:flex-row justify-center items-center space-y-8 md:space-y-0 md:space-x-10">
          <div className="flex flex-col items-center max-w-xs">
            <div className="bg-indigo-100 w-14 h-14 rounded-full flex items-center justify-center mb-4 shadow-md">
              <span className="text-xl font-bold text-indigo-700">1</span>
            </div>
            <h3 className="text-lg font-semibold mb-2 text-center">Upload Image</h3>
            <p className="text-gray-600 text-center">Upload a clear photo of your skin lesion or condition</p>
          </div>
          
          <div className="hidden md:block">
            <svg className="w-6 h-6 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
            </svg>
          </div>
          
          <div className="flex flex-col items-center max-w-xs">
            <div className="bg-purple-100 w-14 h-14 rounded-full flex items-center justify-center mb-4 shadow-md">
              <span className="text-xl font-bold text-purple-700">2</span>
            </div>
            <h3 className="text-lg font-semibold mb-2 text-center">Image Analysis</h3>
            <p className="text-gray-600 text-center">Our models analyze the image to detect and classify skin conditions</p>
          </div>
          
          <div className="hidden md:block">
            <svg className="w-6 h-6 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
            </svg>
          </div>
          
          <div className="flex flex-col items-center max-w-xs">
            <div className="bg-teal-100 w-14 h-14 rounded-full flex items-center justify-center mb-4 shadow-md">
              <span className="text-xl font-bold text-teal-700">3</span>
            </div>
            <h3 className="text-lg font-semibold mb-2 text-center">Get Results</h3>
            <p className="text-gray-600 text-center">View detailed results and connect with specialists if needed</p>
          </div>
        </div>
      </motion.div>

      {/* Disclaimer */}
      <motion.div 
        className="mt-16 max-w-4xl mx-auto"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.6, delay: 0.8 }}
      >
        <div className="bg-amber-50 border-l-4 border-amber-400 p-6 rounded-lg shadow-md">
          <div className="flex items-start">
            <svg className="w-6 h-6 text-amber-600 mt-0.5 mr-3 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            <div>
              <h4 className="text-lg font-semibold text-amber-800 mb-2">
                Important Medical Disclaimer
              </h4>
              <p className="text-amber-700">
                This tool is designed for preliminary screening purposes only and should not 
                replace professional medical diagnosis. The predictions are not a substitute for 
                consultation with qualified healthcare providers. Always consult with a dermatologist 
                or doctor for proper evaluation, diagnosis, and treatment of skin conditions.
              </p>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default Home;

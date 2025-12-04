import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { FaCloudUploadAlt, FaImage, FaTimes } from 'react-icons/fa';
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
        // Step 1: Validate the image first
        const formData = new FormData();
        formData.append('image', selectedImage);
        
        toast.info('Validating image...', { autoClose: 2000 });
        
        try {
          const validateResponse = await axios.post(API_CONFIG.getUrl('VALIDATE'), formData, {
            headers: {
              'Content-Type': 'multipart/form-data',
            },
            timeout: 10000
          });
          
          // If validation passes, show success and proceed to prediction
          if (validateResponse.data.valid) {
            toast.success('Image validated successfully! Analyzing...', { autoClose: 2000 });
            
            // Step 2: Now call predict endpoint
            const predictFormData = new FormData();
            predictFormData.append('image', selectedImage);
            
            const response = await axios.post(API_CONFIG.getUrl('PREDICT'), predictFormData, {
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
            
            toast.success('Analysis completed successfully!');
            navigate('/results');
          }
        } catch (validationError) {
          // If validation fails, show specific error
          if (validationError.response?.status === 400) {
            const errorMessage = validationError.response?.data?.error || 'Invalid image';
            const errorDetails = validationError.response?.data?.details || '';
            
            toast.error(
              <div>
                <div className="font-semibold">{errorMessage}</div>
                {errorDetails && <div className="text-sm mt-1">{errorDetails}</div>}
              </div>,
              {
                duration: 6000,
                style: {
                  maxWidth: '500px',
                }
              }
            );
          } else {
            // Network or other errors
            toast.error('Validation failed. Please ensure the backend is running.');
          }
          setLoading(false);
          return;
        }
      }
      
    } catch (error) {
      console.error('Error:', error);
      
      if (API_CONFIG.USE_MOCK) {
        toast.error('Error in demo mode. Please try again.');
      } else {
        // Extract detailed error message from backend
        const errorMessage = error.response?.data?.error || error.message || 'Failed to analyze image.';
        
        // If it's a prediction error (400), show the specific reason
        if (error.response?.status === 400) {
          toast.error(errorMessage, {
            duration: 5000,
            style: {
              maxWidth: '500px',
            }
          });
        } else {
          // Network or other errors
          toast.error(`${errorMessage} Please ensure the backend is running.`);
        }
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
       
        <h1 className="text-5xl font-bold mb-6 bg-gradient-to-r from-cyan-600 via-blue-600 to-purple-600 dark:from-cyan-400 dark:via-blue-500 dark:to-purple-500 bg-clip-text text-transparent">
          Melanoma Detection System
        </h1>
        <p className="text-xl text-gray-600 dark:text-gray-400 max-w-3xl mx-auto mb-8">
          Upload an image of your skin lesion for instant analysis
        </p>
      </motion.div>

      {/* Main Upload Section */}
      <motion.div 
        className="max-w-5xl mx-auto"
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5, delay: 0.2 }}
      >
        <div className="bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800 rounded-2xl shadow-2xl shadow-cyan-500/10 p-10 hover:shadow-cyan-500/20 transition-all duration-300 border border-gray-200 dark:border-gray-700">
          {!imagePreview ? (
            <div
              className="upload-zone border-3 border-dashed border-cyan-500/30 rounded-xl p-12 text-center cursor-pointer hover:border-cyan-400 hover:bg-cyan-500/5 transition-all duration-300 bg-gradient-to-br from-white to-gray-50 dark:from-gray-900 dark:to-gray-800"
              onClick={() => fileInputRef.current.click()}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
            >
              <motion.div
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ type: "spring", stiffness: 300 }}
              >
                <div className="bg-cyan-500/10 w-24 h-24 rounded-full flex items-center justify-center mx-auto mb-6 border border-cyan-500/30">
                  <FaCloudUploadAlt className="text-5xl text-cyan-400" />
                </div>
                                <h3 className="text-2xl font-semibold mb-4 bg-gradient-to-r from-cyan-600 to-blue-600 dark:from-cyan-400 dark:to-blue-500 bg-clip-text text-transparent">
                  Upload Your Image
                </h3>
                <p className="text-gray-600 dark:text-gray-300 mb-6 text-lg max-w-xl mx-auto">
                  Drag and drop your image here, or click to browse files from your device
                </p>
                <div className="bg-cyan-500/10 py-3 px-5 rounded-lg inline-block mb-4 border border-cyan-500/30">
                  <p className="text-sm text-cyan-700 dark:text-cyan-300 font-medium flex items-center">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M2.166 4.999A11.954 11.954 0 0010 1.944 11.954 11.954 0 0017.834 5c.11.65.166 1.32.166 2.001 0 5.225-3.34 9.67-8 11.317C5.34 16.67 2 12.225 2 7c0-.682.057-1.35.166-2.001zm11.541 3.708a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                    </svg>
                    Supported formats: PNG, JPG, JPEG (Max 16MB)
                  </p>
                </div>
                <button className="bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white px-6 py-3 rounded-lg transition-all duration-200 font-medium inline-flex items-center shadow-lg shadow-cyan-500/50 hover:shadow-cyan-400/60 hover:scale-105">
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
                  
                  <div className="mt-3 text-center text-sm text-gray-600 dark:text-gray-500">
                    Selected image for analysis
                  </div>
                </div>
                
                <div className="w-full max-w-md">
                  <h3 className="text-xl font-semibold mb-4 text-cyan-600 dark:text-cyan-400">Ready for Analysis</h3>
                  
                  <div className="bg-cyan-500/10 border border-cyan-500/30 rounded-lg p-4 mb-6">
                    <div className="flex items-start">
                      <div className="bg-cyan-400/20 p-2 rounded-full">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-cyan-400" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                        </svg>
                      </div>
                      <p className="ml-3 text-sm text-gray-700 dark:text-gray-300">
                        The analysis will include lesion classification, risk assessment, and segmentation using our ResNet50 and U-Net deep learning models.
                      </p>
                    </div>
                  </div>
                  
                  <motion.button
                    whileHover={{ scale: 1.03 }}
                    whileTap={{ scale: 0.97 }}
                    onClick={handleAnalyze}
                    disabled={loading}
                    className="w-full bg-gradient-to-r from-cyan-500 to-blue-600 text-white font-medium text-lg px-8 py-4 rounded-lg shadow-lg shadow-cyan-500/50 hover:shadow-2xl hover:shadow-cyan-400/60 transition-all duration-300 disabled:opacity-60 disabled:cursor-not-allowed"
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
                        Start Analysis
                      </span>
                    )}
                  </motion.button>
                  
                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-3 text-center">
                    Analysis typically completes within 5-10 seconds
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      </motion.div>

      {/* How It Works */}
      <motion.div
        className="mt-20" 
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.6, delay: 0.6 }}
      >
        <h2 className="text-3xl font-bold text-center mb-12 bg-gradient-to-r from-cyan-600 to-blue-600 dark:from-cyan-400 dark:to-blue-500 bg-clip-text text-transparent">
          How It Works
        </h2>
        
        <div className="flex flex-col md:flex-row justify-center items-center space-y-8 md:space-y-0 md:space-x-10">
          <div className="flex flex-col items-center max-w-xs">
            <div className="bg-cyan-500/20 border border-cyan-500/40 w-14 h-14 rounded-full flex items-center justify-center mb-4 shadow-lg shadow-cyan-500/30">
              <span className="text-xl font-bold text-cyan-400">1</span>
            </div>
            <h3 className="text-lg font-semibold mb-2 text-center text-gray-800 dark:text-gray-200">Upload Image</h3>
            <p className="text-gray-600 dark:text-gray-400 text-center">Upload a clear photo of your skin lesion or condition</p>
          </div>
          
          <div className="hidden md:block">
            <svg className="w-6 h-6 text-cyan-500/50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
            </svg>
          </div>
          
          <div className="flex flex-col items-center max-w-xs">
            <div className="bg-blue-500/20 border border-blue-500/40 w-14 h-14 rounded-full flex items-center justify-center mb-4 shadow-lg shadow-blue-500/30">
              <span className="text-xl font-bold text-blue-400">2</span>
            </div>
            <h3 className="text-lg font-semibold mb-2 text-center text-gray-800 dark:text-gray-200">Image Analysis</h3>
            <p className="text-gray-600 dark:text-gray-400 text-center">Our models analyze the image to detect and classify skin conditions</p>
          </div>
          
          <div className="hidden md:block">
            <svg className="w-6 h-6 text-cyan-500/50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
            </svg>
          </div>
          
          <div className="flex flex-col items-center max-w-xs">
            <div className="bg-purple-500/20 border border-purple-500/40 w-14 h-14 rounded-full flex items-center justify-center mb-4 shadow-lg shadow-purple-500/30">
              <span className="text-xl font-bold text-purple-400">3</span>
            </div>
            <h3 className="text-lg font-semibold mb-2 text-center text-gray-800 dark:text-gray-200">Get Results</h3>
            <p className="text-gray-600 dark:text-gray-400 text-center">View detailed results and connect with specialists if needed</p>
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
        <div className="bg-amber-500/10 border border-amber-500/30 border-l-4 border-l-amber-500 p-6 rounded-lg shadow-lg shadow-amber-500/10">
          <div className="flex items-start">
            <svg className="w-6 h-6 text-amber-400 mt-0.5 mr-3 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            <div>
              <h4 className="text-lg font-semibold text-amber-400 mb-2">
                Important Medical Disclaimer
              </h4>
              <p className="text-gray-300">
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

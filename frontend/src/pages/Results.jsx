import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { FaDownload, FaUserMd, FaExclamationTriangle, FaCheckCircle } from 'react-icons/fa';
import { toast } from 'react-toastify';
import axios from 'axios';
import MedicalChatbot from '../components/MedicalChatbot';

const Results = () => {
  const [result, setResult] = useState(null);
  const [showPatientForm, setShowPatientForm] = useState(false);
  const [patientInfo, setPatientInfo] = useState({
    name: '',
    age: '',
    gender: 'male',
    phone: '',
    email: '',
    medical_history: ''
  });
  const [generatingReport, setGeneratingReport] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    const storedResult = localStorage.getItem('analysisResult');
    if (storedResult) {
      setResult(JSON.parse(storedResult));
    } else {
      navigate('/');
    }
  }, [navigate]);

  const getSeverityClass = (severity) => {
    const classes = {
      'Critical': 'severity-critical',
      'High': 'severity-high',
      'Medium': 'severity-medium',
      'Low': 'severity-low',
      'None': 'severity-none'
    };
    return classes[severity] || 'severity-medium';
  };

  const handleInputChange = (e) => {
    setPatientInfo({
      ...patientInfo,
      [e.target.name]: e.target.value
    });
  };

  const handleGenerateReport = async () => {
    // Validate patient info
    if (!patientInfo.name || !patientInfo.age || !patientInfo.phone || !patientInfo.email) {
      toast.error('Please fill in all required fields');
      return;
    }

    setGeneratingReport(true);

    try {
      const response = await axios.post('http://localhost:5001/generate-report', {
        patient_info: patientInfo,
        prediction_data: result
      });      toast.success('Report generated successfully!');
      
      // Download the report
      const downloadUrl = `http://localhost:5001${response.data.download_url}`;
      window.open(downloadUrl, '_blank');
      
      // Store report filename for doctor consultation
      localStorage.setItem('reportFilename', response.data.report_filename);
      localStorage.setItem('patientInfo', JSON.stringify(patientInfo));
      
      setShowPatientForm(false);
    } catch (error) {
      console.error('Error:', error);
      toast.error('Failed to generate report. Please try again.');
    } finally {
      setGeneratingReport(false);
    }
  };

  const handleConsultDoctor = () => {
    if (!localStorage.getItem('reportFilename')) {
      toast.warning('Please generate a report first');
      setShowPatientForm(true);
      return;
    }
    navigate('/doctors');
  };

  if (!result) {
    return (
      <div className="container mx-auto px-6 py-12 flex flex-col items-center justify-center min-h-[40vh]">
        <div className="spinner mx-auto border-cyan-400 border-t-transparent"></div>
        <p className="mt-4 text-gray-700 dark:text-gray-300 font-medium">Loading analysis results...</p>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-6 py-10">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <h1 className="text-4xl font-bold text-center mb-8 bg-gradient-to-r from-cyan-600 to-blue-600 dark:from-cyan-400 dark:to-blue-500 bg-clip-text text-transparent">
          Analysis Results
        </h1>

        {/* Main Results Card */}
        <div className="max-w-6xl mx-auto">
          <div className="grid md:grid-cols-2 lg:grid-cols-5 gap-6 mb-8">
            {/* Image Display */}
            <div className="md:col-span-1 lg:col-span-2 bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800 rounded-xl shadow-2xl shadow-cyan-500/10 p-6 hover:shadow-cyan-500/20 transition-all border border-gray-200 dark:border-gray-700">
              <h3 className="text-xl font-semibold mb-4 flex items-center text-cyan-600 dark:text-cyan-400">
                <span className="inline-block w-2 h-2 rounded-full bg-cyan-400 mr-2"></span>
                Analyzed Image
              </h3>
              
              <div className="relative image-highlight rounded-lg overflow-hidden">
                <img
                  src={result.imagePreview}
                  alt="Analyzed skin lesion"
                  className="w-full rounded-lg shadow-md"
                />
                
                {result?.using_mock_prediction && (
                  <div className="absolute bottom-2 right-2 bg-yellow-500/20 border border-yellow-500/40 text-yellow-300 text-xs px-2 py-1 rounded-md shadow-sm">
                    Demo Mode
                  </div>
                )}
              </div>
              
              <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-600 dark:text-gray-400">Analysis Date:</span>
                  <span className="text-gray-800 dark:text-gray-200 font-medium">{new Date().toLocaleDateString()}</span>
                </div>
              </div>
            </div>

            {/* Detection Results */}
            <div className="md:col-span-1 lg:col-span-3 bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800 rounded-xl shadow-2xl shadow-blue-500/10 p-6 hover:shadow-blue-500/20 transition-all border border-gray-200 dark:border-gray-700">
              <h3 className="text-xl font-semibold mb-4 flex items-center text-blue-600 dark:text-blue-400">
                <span className="inline-block w-2 h-2 rounded-full bg-blue-400 mr-2"></span>
                Detection Results
              </h3>
              
              <div className="space-y-6">
                <div>
                  <label className="text-sm font-medium text-gray-600 dark:text-gray-400">Detected Condition</label>
                  <p className="text-3xl font-bold bg-gradient-to-r from-cyan-600 to-blue-600 dark:from-cyan-400 dark:to-blue-500 bg-clip-text text-transparent mt-1">
                    {result.disease}
                  </p>
                </div>
                


                <div>
                  <label className="text-sm font-medium text-gray-600 dark:text-gray-400">Confidence Level</label>
                  <div className="flex items-center mt-2">
                    <div className="flex-1 bg-gray-300 dark:bg-gray-700 rounded-full h-4 mr-3 overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${result.confidence}%` }}
                        transition={{ duration: 1, ease: "easeOut" }}
                        className="bg-gradient-to-r from-cyan-500 to-blue-600 h-4 rounded-full shadow-lg"
                      ></motion.div>
                    </div>
                    <span className="text-xl font-bold text-cyan-600 dark:text-cyan-400">{result.confidence}%</span>
                  </div>
                </div>

                <div className="mb-6">
                  <label className="text-sm font-medium text-gray-600 dark:text-gray-500 mb-2 block">Severity Level</label>
                  <div 
                    className={`inline-flex items-center px-4 py-2 rounded-lg font-semibold shadow-sm ${
                      result.severity === 'Critical' ? 'bg-red-100 text-red-800 border border-red-200' :
                      result.severity === 'High' ? 'bg-orange-100 text-orange-800 border border-orange-200' :
                      result.severity === 'Medium' ? 'bg-amber-100 text-amber-800 border border-amber-200' :
                      result.severity === 'Low' ? 'bg-green-100 text-green-800 border border-green-200' :
                      'bg-blue-100 text-blue-800 border border-blue-200'
                    }`}
                  >
                    {result.severity === 'Critical' || result.severity === 'High' ? (
                      <FaExclamationTriangle className="mr-2" />
                    ) : (
                      <FaCheckCircle className="mr-2" />
                    )}
                    {result.severity} Severity
                  </div>
                </div>
                
                <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
                  <p className="text-gray-700 dark:text-gray-300 leading-relaxed">{result.description}</p>
                </div>
              </div>
            </div>
          </div>

          {/* Recommendations */}
          <div className="bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800 rounded-xl shadow-2xl shadow-green-500/10 p-6 hover:shadow-green-500/20 transition-all border border-gray-200 dark:border-gray-700 mb-8">
            <h3 className="text-xl font-semibold mb-6 flex items-center text-green-600 dark:text-green-400">
              <span className="inline-block w-2 h-2 rounded-full bg-green-400 mr-2"></span>
              Professional Recommendations
            </h3>
            
            <div className="bg-cyan-500/10 border border-cyan-500/30 p-4 rounded-lg mb-6">
              <div className="flex items-center mb-3">
                <svg className="w-6 h-6 text-cyan-400 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <p className="text-gray-800 dark:text-gray-200 font-medium">Based on the analysis, we recommend the following steps:</p>
              </div>
            </div>
            
            <ul className="grid md:grid-cols-2 gap-4">
              {result.recommendations.map((rec, index) => (
                <li key={index} className="flex items-start bg-white dark:bg-gray-800/50 border border-gray-200 dark:border-gray-700 rounded-lg p-4 hover:border-cyan-500/50 hover:bg-cyan-500/5 transition-colors">
                  <div className="bg-gradient-to-r from-cyan-500 to-blue-600 text-white rounded-full w-6 h-6 flex items-center justify-center mr-3 mt-1 flex-shrink-0 shadow-lg shadow-cyan-500/30">
                    {index + 1}
                  </div>
                  <span className="text-gray-700 dark:text-gray-300">{rec}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Melanoma Stage Treatment Path */}
          {result.disease && result.disease.startsWith('Melanoma Stage') && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="mb-8"
            >
              <div className="bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800 rounded-xl shadow-2xl shadow-purple-500/10 p-6 hover:shadow-purple-500/20 transition-all border border-gray-200 dark:border-gray-700">
                <h3 className="text-xl font-semibold mb-6 flex items-center text-purple-600 dark:text-purple-400">
                  <span className="inline-block w-2 h-2 rounded-full bg-purple-400 mr-2"></span>
                  Melanoma Treatment Pathway
                </h3>
                
                <div className="relative">
                  <div className="absolute top-0 bottom-0 left-6 w-0.5 bg-gray-300 dark:bg-gray-700"></div>
                  
                  <div className="relative flex items-start mb-6">
                    <div className="flex items-center justify-center w-12 h-12 rounded-full border-4 border-cyan-500/30 bg-cyan-500/10 text-cyan-400 font-bold text-lg z-10">
                      1
                    </div>
                    <div className="ml-6">
                      <h4 className="text-lg font-semibold text-gray-800 dark:text-gray-200">Diagnosis Confirmation</h4>
                      <p className="text-gray-600 dark:text-gray-400 mt-1">A dermatologist will likely perform a biopsy to confirm the melanoma diagnosis and determine the exact characteristics.</p>
                    </div>
                  </div>
                  
                  <div className="relative flex items-start mb-6">
                    <div className="flex items-center justify-center w-12 h-12 rounded-full border-4 border-blue-500/30 bg-blue-500/10 text-blue-400 font-bold text-lg z-10">
                      2
                    </div>
                    <div className="ml-6">
                      <h4 className="text-lg font-semibold text-gray-800 dark:text-gray-200">Staging Assessment</h4>
                      <p className="text-gray-600 dark:text-gray-400 mt-1">Additional tests may be conducted to confirm the exact stage, which guides treatment decisions.</p>
                    </div>
                  </div>
                  
                  <div className="relative flex items-start mb-6">
                    <div className="flex items-center justify-center w-12 h-12 rounded-full border-4 border-purple-500/30 bg-purple-500/10 text-purple-400 font-bold text-lg z-10">
                      3
                    </div>
                    <div className="ml-6">
                      <h4 className="text-lg font-semibold text-gray-800 dark:text-gray-200">Treatment Plan</h4>
                      <p className="text-gray-600 dark:text-gray-400 mt-1">
                        {result.melanoma_stage === 1 && 'Surgery to remove the melanoma and a margin of normal skin is typically the main treatment for Stage 1.'}
                        {result.melanoma_stage === 2 && 'Wide excision surgery with potential sentinel lymph node biopsy to check if cancer has spread to lymph nodes.'}
                        {result.melanoma_stage === 3 && 'Surgery plus immunotherapy, targeted therapy, or clinical trials depending on specific characteristics.'}
                        {result.melanoma_stage === 4 && 'Combination of treatments including immunotherapy, targeted therapy, radiation, and potential clinical trials.'}
                      </p>
                    </div>
                  </div>
                  
                  <div className="relative flex items-start">
                    <div className="flex items-center justify-center w-12 h-12 rounded-full border-4 border-green-500/30 bg-green-500/10 text-green-400 font-bold text-lg z-10">
                      4
                    </div>
                    <div className="ml-6">
                      <h4 className="text-lg font-semibold text-gray-800 dark:text-gray-200">Follow-up Care</h4>
                      <p className="text-gray-600 dark:text-gray-400 mt-1">
                        {result.melanoma_stage <= 2 ? 
                          'Regular skin exams, imaging tests, and follow-up appointments to monitor for recurrence or new melanomas.' :
                          'Intensive monitoring and follow-up care with your oncology team to track response to treatment.'}
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
          
          {/* Warning for Severe Cases */}
          {(result?.severity === 'Critical' || result?.severity === 'High') && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="mb-8"
            >
              <div className="bg-red-50 border-l-4 border-red-500 p-5 rounded-lg shadow-md">
                <div className="flex">
                  <div className="flex-shrink-0">
                    <FaExclamationTriangle className="text-red-600 text-xl" />
                  </div>
                  <div className="ml-3">
                    <h3 className="text-lg font-semibold text-red-800">Medical Attention Recommended</h3>
                    <p className="text-red-700">
                      Based on our analysis, this condition requires prompt medical attention.
                      Please consult with a qualified dermatologist or healthcare professional as soon as possible.
                      {result.disease && result.disease.startsWith('Melanoma Stage') ? 
                        ` Early diagnosis and treatment are critical for melanoma, especially at Stage ${result.melanoma_stage}.` :
                        ' Early diagnosis and treatment are crucial for skin conditions with this severity level.'}
                    </p>
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {/* Action Buttons - Only show for valid medical images */}
          {result.disease !== 'Invalid Image' && (
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <motion.button
                whileHover={{ scale: 1.03 }}
                whileTap={{ scale: 0.97 }}
                onClick={() => setShowPatientForm(true)}
                className="flex items-center justify-center py-4 px-8 bg-gradient-to-r from-cyan-500 to-blue-600 text-white font-semibold rounded-lg shadow-lg shadow-cyan-500/50 hover:shadow-2xl hover:shadow-cyan-400/60 transition-all duration-300"
                disabled={generatingReport}
              >
                {generatingReport ? (
                  <>
                    <div className="spinner mr-2 border-2 border-white border-t-transparent"></div>
                    <span>Generating Report...</span>
                  </>
                ) : (
                  <>
                    <FaDownload className="mr-2" />
                    Download Medical Report
                  </>
                )}
              </motion.button>
              
              <motion.button
                whileHover={{ scale: 1.03 }}
                whileTap={{ scale: 0.97 }}
                onClick={handleConsultDoctor}
                className="flex items-center justify-center py-4 px-8 bg-gradient-to-r from-emerald-500 to-teal-500 text-white font-semibold rounded-lg shadow-lg shadow-emerald-500/50 hover:shadow-2xl hover:shadow-emerald-400/60 transition-all duration-300"
              >
                <FaUserMd className="mr-2" />
                Consult a Specialist
              </motion.button>
            </div>
          )}
          
          {/* For Invalid Images - Show "Try Again" button */}
          {result.disease === 'Invalid Image' && (
            <div className="flex justify-center">
              <motion.button
                whileHover={{ scale: 1.03 }}
                whileTap={{ scale: 0.97 }}
                onClick={() => navigate('/')}
                className="flex items-center justify-center py-4 px-8 bg-gradient-to-r from-blue-500 to-purple-600 text-white font-semibold rounded-lg shadow-lg shadow-blue-500/50 hover:shadow-2xl hover:shadow-blue-400/60 transition-all duration-300"
              >
                <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                </svg>
                Upload Another Image
              </motion.button>
            </div>
          )}
        </div>
      </motion.div>

      {/* Patient Information Modal */}
      {showPatientForm && (
        <div className="fixed inset-0 bg-black bg-opacity-70 flex items-center justify-center z-50 p-4 backdrop-blur-sm">
          <motion.div
            className="bg-white bg-opacity-95 rounded-xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto border border-indigo-100"
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
          >
            <div className="p-8">
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-2xl font-bold text-indigo-900">
                  <span className="mr-2">📋</span>
                  Patient Information
                </h2>
                <button 
                  onClick={() => setShowPatientForm(false)}
                  className="text-gray-500 hover:text-gray-700 transition-colors"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              
              <div className="space-y-5">
                <div className="bg-indigo-50 rounded-lg p-4 text-sm text-indigo-700 mb-5">
                  <div className="flex items-center">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                    </svg>
                    <span>Please provide accurate information for the medical report</span>
                  </div>
                </div>

                <div className="grid md:grid-cols-2 gap-5">
                  <div className="form-group">
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Full Name <span className="text-red-500">*</span>
                    </label>
                    <input
                      type="text"
                      name="name"
                      value={patientInfo.name}
                      onChange={handleInputChange}
                      className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:ring-2 focus:ring-indigo-400 focus:border-indigo-400 transition-all duration-200 outline-none bg-white text-gray-900"
                      placeholder="John Doe"
                    />
                  </div>
                  
                  <div className="form-group">
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Age <span className="text-red-500">*</span>
                    </label>
                    <input
                      type="number"
                      name="age"
                      value={patientInfo.age}
                      onChange={handleInputChange}
                      className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:ring-2 focus:ring-indigo-400 focus:border-indigo-400 transition-all duration-200 outline-none bg-white text-gray-900"
                      placeholder="35"
                    />
                  </div>
                </div>

                <div className="form-group">
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Gender <span className="text-red-500">*</span>
                  </label>
                  <select
                    name="gender"
                    value={patientInfo.gender}
                    onChange={handleInputChange}
                    className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:ring-2 focus:ring-indigo-400 focus:border-indigo-400 transition-all duration-200 outline-none bg-white text-gray-900"
                  >
                    <option value="">-- Select Gender --</option>
                    <option value="male">Male</option>
                    <option value="female">Female</option>
                    <option value="other">Other</option>
                  </select>
                </div>

                <div className="grid md:grid-cols-2 gap-5">
                  <div className="form-group">
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Phone Number <span className="text-red-500">*</span>
                    </label>
                    <input
                      type="tel"
                      name="phone"
                      value={patientInfo.phone}
                      onChange={handleInputChange}
                      className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:ring-2 focus:ring-indigo-400 focus:border-indigo-400 transition-all duration-200 outline-none bg-white text-gray-900"
                      placeholder="+1-555-0123"
                    />
                  </div>
                  
                  <div className="form-group">
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Email Address <span className="text-red-500">*</span>
                    </label>
                    <input
                      type="email"
                      name="email"
                      value={patientInfo.email}
                      onChange={handleInputChange}
                      className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:ring-2 focus:ring-indigo-400 focus:border-indigo-400 transition-all duration-200 outline-none bg-white text-gray-900"
                      placeholder="john@example.com"
                    />
                  </div>
                </div>

                <div className="form-group">
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Medical History <span className="text-gray-400">(Optional)</span>
                  </label>
                  <textarea
                    name="medical_history"
                    value={patientInfo.medical_history}
                    onChange={handleInputChange}
                    className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:ring-2 focus:ring-indigo-400 focus:border-indigo-400 transition-all duration-200 outline-none bg-white text-gray-900"
                    rows="3"
                    placeholder="Any relevant medical history, allergies, or current medications..."
                  ></textarea>
                </div>
              </div>

              <div className="flex gap-4 mt-8">
                <button
                  onClick={handleGenerateReport}
                  disabled={generatingReport}
                  className="flex-1 bg-gradient-to-r from-indigo-600 to-purple-600 text-white py-3 px-6 rounded-lg font-medium hover:shadow-lg transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
                >
                  {generatingReport ? (
                    <>
                      <div className="animate-spin h-5 w-5 mr-3 border-2 border-white border-t-transparent rounded-full"></div>
                      Generating...
                    </>
                  ) : (
                    <>
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                      Generate Report
                    </>
                  )}
                </button>
                <button
                  onClick={() => setShowPatientForm(false)}
                  className="px-6 py-3 border border-gray-300 rounded-lg text-gray-700 font-medium hover:bg-gray-50 transition-all duration-300"
                  disabled={generatingReport}
                >
                  Cancel
                </button>
              </div>
            </div>
          </motion.div>
        </div>
      )}

      {/* AI Medical Chatbot - Only show for valid medical images */}
      {result.disease !== 'Invalid Image' && (
        <MedicalChatbot result={result} />
      )}
    </div>
  );
};

export default Results;

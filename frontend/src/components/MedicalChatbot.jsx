import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FaComments, FaTimes, FaPaperPlane, FaRobot, FaUser } from 'react-icons/fa';
import axios from 'axios';
import { useTheme } from '../context/ThemeContext';

const MedicalChatbot = ({ result }) => {
  const { isDark } = useTheme();
  // Function to convert markdown to formatted HTML
  const formatMessage = (text) => {
    return text
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') // **bold**
      .replace(/\*(.*?)\*/g, '<em>$1</em>') // *italic*
      .replace(/\n/g, '<br/>') // line breaks
      .replace(/•/g, '•'); // bullet points
  };
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([
    {
      type: 'bot',
      text: result.disease === 'Melanoma' 
        ? `Hello! I'm your AI Medical Assistant. I understand this diagnosis may be concerning. I'm here to help answer your questions about melanoma and your next steps.\n\nPlease remember that while I can provide general information, it's crucial to consult with a dermatologist for personalized medical advice.\n\nWhat would you like to know?`
        : result.disease === 'Benign'
        ? `Hello! Good news - your skin lesion appears to be benign. I'm here to answer any questions you might have about maintaining healthy skin and monitoring your condition.\n\nWhat would you like to know?`
        : `Hello! I'm your AI Medical Assistant. I can help answer questions about your skin condition analysis. How can I assist you today?`,
      timestamp: new Date()
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const chatContainerRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage = {
      type: 'user',
      text: inputMessage,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const response = await axios.post('http://localhost:5001/chatbot', {
        message: inputMessage,
        context: {
          disease: result.disease,
          severity: result.severity,
          confidence: result.confidence,
          recommendations: result.recommendations,
          melanoma_stage: result.melanoma_stage
        }
      });

      const botMessage = {
        type: 'bot',
        text: response.data.response,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error:', error);
      const errorMessage = {
        type: 'bot',
        text: 'I apologize, but I encountered an error. Please try again or consult with a medical professional for assistance.',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const suggestedQuestions = [
    "What does this diagnosis mean?",
    "Should I see a doctor?",
    "What are my next steps?",
    "How can I protect my skin?",
    "What lifestyle changes should I make?"
  ];

  const handleSuggestedQuestion = (question) => {
    setInputMessage(question);
  };

  return (
    <>
      {/* Floating Chat Button */}
      <AnimatePresence>
        {!isOpen && (
          <motion.button
            initial={{ scale: 0, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0, opacity: 0 }}
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            onClick={() => setIsOpen(true)}
            className="fixed bottom-8 right-8 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-full p-5 shadow-2xl hover:shadow-3xl transition-all duration-300 z-40 flex items-center gap-3 group"
          >
            <FaComments className="text-2xl" />
            <span className="font-semibold pr-1">Ask AI Assistant</span>
            <motion.div
              animate={{ scale: [1, 1.2, 1] }}
              transition={{ repeat: Infinity, duration: 2 }}
              className="absolute -top-1 -right-1 w-4 h-4 bg-green-400 rounded-full border-2 border-white"
            />
          </motion.button>
        )}
      </AnimatePresence>

      {/* Chat Dock */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ x: '100%', opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: '100%', opacity: 0 }}
            transition={{ type: 'spring', damping: 25, stiffness: 200 }}
            className={`fixed right-0 top-0 h-full w-full md:w-1/4 min-w-[320px] max-w-[480px] shadow-2xl z-50 flex flex-col ${
              isDark ? 'bg-gray-900' : 'bg-white'
            }`}
            style={{ borderLeft: isDark ? '1px solid #374151' : '1px solid #e5e7eb' }}
          >
            {/* Header */}
            <div className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white p-4 flex items-center justify-between shadow-lg">
              <div className="flex items-center gap-3">
                <div className="relative">
                  <FaRobot className="text-2xl" />
                  <motion.div
                    animate={{ scale: [1, 1.3, 1] }}
                    transition={{ repeat: Infinity, duration: 2 }}
                    className="absolute -bottom-1 -right-1 w-3 h-3 bg-green-400 rounded-full border-2 border-white"
                  />
                </div>
                <div>
                  <h3 className="font-bold text-lg">AI Medical Assistant</h3>
                  <p className="text-xs text-blue-100">Always online</p>
                </div>
              </div>
              <button
                onClick={() => setIsOpen(false)}
                className="hover:bg-white/20 p-2 rounded-full transition-all duration-200"
              >
                <FaTimes className="text-xl" />
              </button>
            </div>

            {/* Disclaimer */}
            <div className="bg-amber-50 border-l-4 border-amber-400 p-3 text-xs text-amber-800">
              <strong>⚠️ Medical Disclaimer:</strong> This AI assistant provides general information only. Always consult qualified healthcare professionals for medical advice.
            </div>

            {/* Messages Container */}
            <div
              ref={chatContainerRef}
              className={`flex-1 overflow-y-auto p-4 space-y-4 ${isDark ? 'bg-gray-800' : 'bg-gray-50'}`}
              style={{ maxHeight: 'calc(100vh - 280px)' }}
            >
              {messages.map((message, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3 }}
                  className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div className={`flex gap-2 max-w-[85%] ${message.type === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
                    <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
                      message.type === 'user' 
                        ? 'bg-gradient-to-r from-purple-500 to-pink-500' 
                        : 'bg-gradient-to-r from-blue-500 to-indigo-500'
                    }`}>
                      {message.type === 'user' ? (
                        <FaUser className="text-white text-sm" />
                      ) : (
                        <FaRobot className="text-white text-sm" />
                      )}
                    </div>
                    <div>
                      <div className={`rounded-2xl p-3 shadow-sm ${
                        message.type === 'user'
                          ? 'bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-tr-none'
                          : isDark 
                            ? 'bg-gray-700 text-gray-100 rounded-tl-none border border-gray-600'
                            : 'bg-white text-gray-800 rounded-tl-none border border-gray-200'
                      }`}>
                        <div 
                          className="text-sm leading-relaxed whitespace-pre-wrap"
                          dangerouslySetInnerHTML={{ __html: formatMessage(message.text) }}
                        />
                      </div>
                      <p className={`text-xs text-gray-400 mt-1 ${message.type === 'user' ? 'text-right' : 'text-left'}`}>
                        {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </p>
                    </div>
                  </div>
                </motion.div>
              ))}

              {isLoading && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="flex justify-start"
                >
                  <div className="flex gap-2 max-w-[85%]">
                    <div className="flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center bg-gradient-to-r from-blue-500 to-indigo-500">
                      <FaRobot className="text-white text-sm" />
                    </div>
                    <div className={`rounded-2xl rounded-tl-none p-3 shadow-sm ${
                      isDark ? 'bg-gray-700 border border-gray-600' : 'bg-white border border-gray-200'
                    }`}>
                      <div className="flex gap-1">
                        <motion.div
                          animate={{ scale: [1, 1.2, 1] }}
                          transition={{ repeat: Infinity, duration: 0.6, delay: 0 }}
                          className="w-2 h-2 bg-blue-500 rounded-full"
                        />
                        <motion.div
                          animate={{ scale: [1, 1.2, 1] }}
                          transition={{ repeat: Infinity, duration: 0.6, delay: 0.2 }}
                          className="w-2 h-2 bg-blue-500 rounded-full"
                        />
                        <motion.div
                          animate={{ scale: [1, 1.2, 1] }}
                          transition={{ repeat: Infinity, duration: 0.6, delay: 0.4 }}
                          className="w-2 h-2 bg-blue-500 rounded-full"
                        />
                      </div>
                    </div>
                  </div>
                </motion.div>
              )}

              <div ref={messagesEndRef} />
            </div>

            {/* Suggested Questions */}
            {messages.length <= 1 && !isLoading && (
              <div className={`px-4 py-2 border-t ${
                isDark ? 'bg-gray-900 border-gray-700' : 'bg-white border-gray-200'
              }`}>
                <p className={`text-xs mb-2 font-semibold ${isDark ? 'text-gray-400' : 'text-gray-500'}`}>Suggested questions:</p>
                <div className="flex flex-wrap gap-2">
                  {suggestedQuestions.slice(0, 3).map((question, index) => (
                    <button
                      key={index}
                      onClick={() => handleSuggestedQuestion(question)}
                      className="text-xs bg-blue-50 hover:bg-blue-100 text-blue-600 px-3 py-1.5 rounded-full transition-all duration-200 border border-blue-200"
                    >
                      {question}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Input Area */}
            <div className={`p-4 border-t shadow-lg ${
              isDark ? 'bg-gray-900 border-gray-700' : 'bg-white border-gray-200'
            }`}>
              <div className="flex gap-2">
                <textarea
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Ask me anything about your condition..."
                  className={`flex-1 border rounded-xl px-4 py-2.5 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none text-sm ${
                    isDark 
                      ? 'bg-gray-800 border-gray-600 text-white placeholder-gray-400' 
                      : 'bg-white border-gray-300 text-gray-900 placeholder-gray-500'
                  }`}
                  rows="2"
                  disabled={isLoading}
                />
                <button
                  onClick={handleSendMessage}
                  disabled={!inputMessage.trim() || isLoading}
                  className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white px-5 rounded-xl hover:shadow-lg transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
                >
                  <FaPaperPlane className="text-lg" />
                </button>
              </div>
              <p className={`text-xs mt-2 text-center ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>
                Press Enter to send • Shift+Enter for new line
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
};

export default MedicalChatbot;

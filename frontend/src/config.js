// Configuration file for environment-specific settings

// Define the API base URL based on the environment
const isDevelopment = import.meta.env.MODE === 'development';

// For GitHub Pages deployment, we'll use mock services
// In a real production environment, you'd replace this with your actual backend URL
const isGitHubPages = window.location.hostname.includes('github.io');

// API configuration
export const API_CONFIG = {
  // Configuration for API access
  USE_MOCK: isGitHubPages,
  
  // Base URL (use localhost for dev, deployed URL for production)
  BASE_URL: isDevelopment 
    ? 'http://localhost:5001' 
    : 'https://your-backend-api-url.com', // Replace with your actual backend URL when you deploy it
  
  // Define endpoints
  ENDPOINTS: {
    VALIDATE: '/validate',
    PREDICT: '/predict',
    HEALTH: '/',
    DOCTORS: '/doctors',
  },
  
  // Helper function to get full endpoint URL
  getUrl: function(endpoint) {
    return `${this.BASE_URL}${this.ENDPOINTS[endpoint] || endpoint}`;
  }
};

// App configuration
export const APP_CONFIG = {
  // Flag to enable/disable features for demo mode
  DEMO_MODE: isGitHubPages,
  
  // GitHub repository information
  REPO_URL: 'https://github.com/mskandashyambhat/melanoma-detection',
  
  // Version information
  VERSION: '1.0.0',
  
  // Flag to show demo banner
  SHOW_DEMO_BANNER: isGitHubPages,
};
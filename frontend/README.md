# Melanoma Detection Frontend

This is the frontend component of the Melanoma Detection System, a web application that uses AI to detect and classify skin lesions.

## Features

- Upload and analyze skin lesion images
- View AI-based detection results with confidence levels
- Get medical recommendations based on analysis
- Connect with dermatologists and specialists
- Generate medical reports

## Technology Stack

- React 18
- Vite
- Tailwind CSS
- Framer Motion
- React Router
- Axios

## Development Setup

### Prerequisites

- Node.js (v16 or later)
- npm (v7 or later)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/mskandashyambhat/melanoma-detection.git
   cd melanoma-detection/frontend
   ```

2. Install dependencies:
   ```
   npm install
   ```

3. Start the development server:
   ```
   npm run dev
   ```

4. Open your browser and visit `http://localhost:3000`

## Deploying to GitHub Pages

### Setup

1. Update the `vite.config.js` file:
   - Set the base property to your repository name: `base: '/melanoma-detection/'`

2. Install gh-pages package:
   ```
   npm install --save-dev gh-pages
   ```

3. Add deployment scripts to package.json:
   ```json
   "predeploy": "npm run build",
   "deploy": "gh-pages -d dist"
   ```

### Manual Deployment

Run the deploy script:
```
npm run deploy
```

### Automatic Deployment with GitHub Actions

This repository is configured with GitHub Actions for automatic deployment. Any push to the main branch will trigger a build and deploy to GitHub Pages.

## Configuration

- `src/config.js`: Contains environment-specific settings
- Update `API_CONFIG.BASE_URL` in `config.js` for your backend API server address

## Backend Connection

The frontend communicates with the backend server for image analysis. By default, it connects to `http://localhost:5001` in development mode.

For the GitHub Pages deployment, the application runs in "Demo Mode" using mock data since GitHub Pages only supports static content.

## Demo Mode

When deployed to GitHub Pages, the application automatically switches to demo mode, using mock data instead of real API calls. This allows users to experience the application without needing to run the backend server.

## License

[MIT](./LICENSE)

## Contributors

- Skandashyam Bhat

## Acknowledgements

- This project was created as part of a Mini-Project for [Your Institution/Course]
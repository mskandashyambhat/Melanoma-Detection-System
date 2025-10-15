/**
 * Enhanced Mock Service for Melanoma Detection System
 * Provides realistic mock data when backend is unavailable
 * Used primarily for GitHub Pages demonstration
 */

// Use placeholder images from external sources instead of local files
const demoImages = [
  'https://www.skinvision.com/wp-content/uploads/2022/11/melanoma-header.webp',
  'https://www.skincancer.org/wp-content/uploads/Screen-Shot-2019-01-18-at-10.14.55-AM.jpg',
  'https://www.mayoclinic.org/-/media/kcms/gbs/patient-consumer/images/2013/08/26/10/51/ds00190_im00677_bn7_melanomaskin4thu_jpg.jpg'
];

// Get a random demo image
const getRandomDemoImage = () => {
  return demoImages[Math.floor(Math.random() * demoImages.length)];
};

// Various prediction patterns for different skin conditions
const PREDICTION_OPTIONS = [
  {
    class: 'Melanoma Stage 1',
    probability: 0.89,
    confidence_level: 'high',
    severity: 'moderate',
    stage: 1,
    recommendations: [
      'Schedule surgical removal (wide excision) within next 2 weeks',
      'Sentinel lymph node biopsy may be recommended',
      'Regular skin checks every 3-6 months for the next 2 years',
      'Daily sun protection with SPF 50+ sunscreen'
    ],
    similar_cases: [
      { class: 'Melanoma Stage 1', similarity: 0.94, count: 127 },
      { class: 'Melanoma Stage 2', similarity: 0.42, count: 38 },
      { class: 'Basal Cell Carcinoma', similarity: 0.31, count: 62 }
    ]
  },
  {
    class: 'Melanoma Stage 2',
    probability: 0.83,
    confidence_level: 'high',
    severity: 'high',
    stage: 2,
    recommendations: [
      'Immediate surgical removal with wider margins',
      'Sentinel lymph node biopsy strongly recommended',
      'Consider adjuvant therapy options',
      'Regular imaging and follow-ups every 3 months'
    ],
    similar_cases: [
      { class: 'Melanoma Stage 2', similarity: 0.92, count: 86 },
      { class: 'Melanoma Stage 3', similarity: 0.45, count: 31 },
      { class: 'Melanoma Stage 1', similarity: 0.38, count: 42 }
    ]
  },
  {
    class: 'Melanoma Stage 3',
    probability: 0.78,
    confidence_level: 'high',
    severity: 'very high',
    stage: 3,
    recommendations: [
      'Urgent surgical removal of primary tumor and affected lymph nodes',
      'Immunotherapy and/or targeted therapy strongly advised',
      'Radiation therapy may be necessary',
      'Clinical trial participation should be considered'
    ],
    similar_cases: [
      { class: 'Melanoma Stage 3', similarity: 0.89, count: 63 },
      { class: 'Melanoma Stage 4', similarity: 0.52, count: 27 },
      { class: 'Melanoma Stage 2', similarity: 0.34, count: 41 }
    ]
  },
  {
    class: 'Melanoma Stage 4',
    probability: 0.75,
    confidence_level: 'high',
    severity: 'critical',
    stage: 4,
    recommendations: [
      'Immediate oncology consultation with melanoma specialist',
      'Systemic therapies including immunotherapy and targeted therapy',
      'Comprehensive mutation testing for treatment planning',
      'Consider clinical trials for innovative treatment options'
    ],
    similar_cases: [
      { class: 'Melanoma Stage 4', similarity: 0.87, count: 42 },
      { class: 'Melanoma Stage 3', similarity: 0.61, count: 38 },
      { class: 'Metastatic Cancer', similarity: 0.43, count: 24 }
    ]
  },
  {
    class: 'Basal Cell Carcinoma',
    probability: 0.83,
    confidence_level: 'high',
    severity: 'moderate',
    recommendations: [
      'Schedule an appointment with a dermatologist',
      'Protect the area from sun exposure',
      'Consider biopsy for confirmation',
      'Review treatment options with your doctor'
    ],
    similar_cases: [
      { class: 'Basal Cell Carcinoma', similarity: 0.89, count: 178 },
      { class: 'Melanoma', similarity: 0.31, count: 42 },
      { class: 'Acne', similarity: 0.16, count: 24 }
    ]
  },
  {
    class: 'Normal Skin',
    probability: 0.91,
    confidence_level: 'very high',
    severity: 'none',
    recommendations: [
      'Continue regular skin self-examinations',
      'Use sun protection with SPF 30+',
      'Schedule annual skin check with a dermatologist'
    ],
    similar_cases: [
      { class: 'Normal Skin', similarity: 0.94, count: 531 },
      { class: 'Eczema', similarity: 0.11, count: 17 },
      { class: 'Psoriasis', similarity: 0.09, count: 12 }
    ]
  },
  {
    class: 'Acne',
    probability: 0.87,
    confidence_level: 'high',
    severity: 'mild',
    recommendations: [
      'Use non-comedogenic skin products',
      'Consider over-the-counter treatments with salicylic acid or benzoyl peroxide',
      'Consult a dermatologist if condition persists or worsens'
    ],
    similar_cases: [
      { class: 'Acne', similarity: 0.95, count: 412 },
      { class: 'Ringworm', similarity: 0.22, count: 38 },
      { class: 'Normal Skin', similarity: 0.18, count: 53 }
    ]
  },
  {
    class: 'Eczema',
    probability: 0.85,
    confidence_level: 'high',
    severity: 'mild',
    recommendations: [
      'Avoid known triggers (stress, allergens, harsh soaps)',
      'Apply prescribed corticosteroid creams as directed',
      'Keep skin well moisturized',
      'Consult dermatologist if symptoms worsen'
    ],
    similar_cases: [
      { class: 'Eczema', similarity: 0.93, count: 287 },
      { class: 'Psoriasis', similarity: 0.41, count: 75 },
      { class: 'Contact Dermatitis', similarity: 0.32, count: 54 }
    ]
  }
];

// Mock doctors data
const MOCK_DOCTORS = [
  {
    id: 'dr1',
    name: 'Dr. Sarah Johnson',
    specialization: 'Dermatology, Skin Cancer',
    experience: '12 years',
    rating: 4.9,
    reviews: 127,
    image: 'https://randomuser.me/api/portraits/women/22.jpg',
    available: true,
    hospital: 'Memorial Skin Center',
    nextAvailable: 'Today, 4:30 PM',
    education: 'MD, Harvard Medical School',
    bio: 'Dr. Johnson specializes in early melanoma detection and has published numerous research papers on advanced diagnostic techniques.'
  },
  {
    id: 'dr2',
    name: 'Dr. Michael Chen',
    specialization: 'Oncology, Melanoma Treatment',
    experience: '15 years',
    rating: 4.8,
    reviews: 96,
    image: 'https://randomuser.me/api/portraits/men/32.jpg',
    available: true,
    hospital: 'University Medical Center',
    nextAvailable: 'Tomorrow, 10:00 AM',
    education: 'MD, Johns Hopkins University',
    bio: 'Dr. Chen is a leading oncologist specializing in melanoma treatment, with expertise in immunotherapy and targeted treatments.'
  },
  {
    id: 'dr3',
    name: 'Dr. Elena Rodriguez',
    specialization: 'Dermatopathology',
    experience: '9 years',
    rating: 4.7,
    reviews: 83,
    image: 'https://randomuser.me/api/portraits/women/65.jpg',
    available: false,
    hospital: 'City Skin Institute',
    nextAvailable: 'Oct 25, 2:15 PM',
    education: 'MD, Stanford University',
    bio: 'Dr. Rodriguez focuses on dermatopathology and precise diagnosis of complex skin conditions through microscopic tissue analysis.'
  }
];

/**
 * Enhanced Mock Service Class
 * Provides more realistic and varied predictions
 */
class MockService {
  // Simulate a delay for API calls
  async delay(ms = 1500) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  // Mock predict API with improved realism and melanoma staging
  async predictImage() {
    await this.delay();
    
    // Select a random prediction pattern from our options
    // This makes the demo mode more realistic by showing different possible diagnoses
    const selectedPrediction = PREDICTION_OPTIONS[Math.floor(Math.random() * PREDICTION_OPTIONS.length)];
    
    // Add some randomness to the values to make each prediction unique
    const confidenceVariation = (Math.random() * 0.1) - 0.05; // +/- 5%
    
    // Determine if this is a melanoma prediction
    const isMelanoma = selectedPrediction.class.startsWith('Melanoma Stage');
    const melanomaStage = isMelanoma ? parseInt(selectedPrediction.class.split(' ').pop()) : 0;
    
    // Adjust segmentation and risk factors based on melanoma stage
    let borderIrregularity, colorVariation, riskFactors;
    
    if (isMelanoma) {
      // More advanced stages have more severe characteristics
      borderIrregularity = melanomaStage >= 3 ? 'severe' : (melanomaStage === 2 ? 'high' : 'moderate');
      colorVariation = melanomaStage >= 3 ? 'extreme' : (melanomaStage === 2 ? 'significant' : 'moderate');
      
      riskFactors = {
        age: melanomaStage >= 3 ? 'high' : 'moderate',
        sun_exposure: 'high',
        family_history: melanomaStage >= 2 ? 'high' : 'moderate',
        overall: melanomaStage >= 2 ? 'high' : 'moderate'
      };
    } else {
      // Non-melanoma conditions
      borderIrregularity = selectedPrediction.class === 'Normal Skin' ? 'low' : 
                           (selectedPrediction.class === 'Basal Cell Carcinoma' ? 'moderate' : 'low');
      colorVariation = selectedPrediction.class === 'Normal Skin' ? 'minimal' : 
                       (selectedPrediction.class === 'Basal Cell Carcinoma' ? 'moderate' : 'low');
      
      riskFactors = {
        age: selectedPrediction.class === 'Basal Cell Carcinoma' ? 'moderate' : 'low',
        sun_exposure: selectedPrediction.class === 'Basal Cell Carcinoma' ? 'high' : 'moderate',
        family_history: 'low',
        overall: selectedPrediction.class === 'Basal Cell Carcinoma' ? 'moderate' : 'low'
      };
    }
    
    // Calculate features for consistent staging
    const features = {
      // ABCDE criteria: Asymmetry, Border irregularity, Color variation, Diameter, Evolution
      asymmetry: isMelanoma ? (0.5 + (melanomaStage * 0.1)) : 0.2,
      border_irregularity: isMelanoma ? (0.4 + (melanomaStage * 0.15)) : 0.3,
      color_variation: isMelanoma ? (0.4 + (melanomaStage * 0.12)) : 0.25,
      diameter_mm: isMelanoma ? (5 + (melanomaStage * 2)) : 4,
      evolution_rate: isMelanoma ? (melanomaStage * 0.2) : 0.1
    };
    
    // Construct result using the selected prediction template with slight variations
    return {
      prediction: {
        class: selectedPrediction.class,
        probability: Math.min(0.99, Math.max(0.5, selectedPrediction.probability + confidenceVariation)),
        confidence_level: selectedPrediction.confidence_level,
        severity: selectedPrediction.severity,
        recommendations: selectedPrediction.recommendations,
        stage: isMelanoma ? melanomaStage : null
      },
      similar_cases: selectedPrediction.similar_cases,
      segmentation: {
        area: 120 + Math.random() * 40 + (isMelanoma ? melanomaStage * 15 : 0),
        diameter: features.diameter_mm,
        border_irregularity: borderIrregularity,
        color_variation: colorVariation
      },
      risk_factors: riskFactors,
      features: features,
      timestamp: new Date().toISOString(),
      imagePreview: getRandomDemoImage(),
      melanoma_detection: {
        is_melanoma: isMelanoma,
        stage: isMelanoma ? melanomaStage : null,
        prognosis: isMelanoma ? 
          (melanomaStage === 1 ? 'Excellent (>95% 5-year survival)' :
          (melanomaStage === 2 ? 'Good (65-90% 5-year survival)' :
          (melanomaStage === 3 ? 'Moderate (45-75% 5-year survival)' : 
                                 'Guarded (10-30% 5-year survival)'))) : 'N/A'
      }
    };
  }

  // Mock doctors API
  async getDoctors() {
    await this.delay(1000);
    return MOCK_DOCTORS;
  }

  // Mock health check
  async checkHealth() {
    await this.delay(500);
    return { status: 'ok', mode: 'demo' };
  }
}

// Export the mock service
export const mockService = new MockService();
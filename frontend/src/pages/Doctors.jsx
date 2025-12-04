import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { FaStar, FaMapMarkerAlt, FaClock, FaEnvelope, FaPhone, FaCheckCircle } from 'react-icons/fa';
import { toast } from 'react-toastify';
import axios from 'axios';

const Doctors = () => {
  const [doctors, setDoctors] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedDoctor, setSelectedDoctor] = useState(null);
  const [sending, setSending] = useState(false);

  useEffect(() => {
    fetchDoctors();
  }, []);

  const fetchDoctors = async () => {
    try {
      const response = await axios.get('http://localhost:5001/doctors', { timeout: 8000 });
      const apiDoctors = response?.data?.doctors || [];
      if (apiDoctors.length === 0) {
        throw new Error('Empty doctor list');
      }
      setDoctors(apiDoctors);
    } catch (error) {
      console.error('Error fetching doctors:', error);
      // Fallback dummy doctors
      const dummyDoctors = [
        {
          id: 101,
          name: 'Dr. Sarah Johnson',
          specialization: 'Dermatology & Skin Cancer',
          experience: '15 years',
          rating: 4.9,
          location: 'New York Medical Center',
          availability: 'Mon-Fri, 9 AM - 5 PM',
          image: 'https://images.unsplash.com/photo-1559839734-2b71ea197ec2?w=300&h=300&fit=crop&crop=faces',
          email: 'sarah.johnson@hospital.com',
          phone: '+1-555-0101'
        },
        {
          id: 102,
          name: 'Dr. Michael Chen',
          specialization: 'Dermatologist',
          experience: '12 years',
          rating: 4.8,
          location: 'City Dermatology Clinic',
          availability: 'Mon-Sat, 10 AM - 6 PM',
          image: 'https://images.unsplash.com/photo-1612349317150-e413f6a5b16d?w=300&h=300&fit=crop&crop=faces',
          email: 'michael.chen@clinic.com',
          phone: '+1-555-0102'
        },
        {
          id: 103,
          name: 'Dr. Emily Rodriguez',
          specialization: 'Oncologist & Dermatology',
          experience: '18 years',
          rating: 5.0,
          location: 'Cancer Care Institute',
          availability: 'Tue-Sat, 8 AM - 4 PM',
          image: 'https://images.unsplash.com/photo-1594824476967-48c8b964273f?w=300&h=300&fit=crop&crop=faces',
          email: 'emily.rodriguez@cci.com',
          phone: '+1-555-0103'
        },
        {
          id: 104,
          name: 'Dr. James Williams',
          specialization: 'Dermatology',
          experience: '10 years',
          rating: 4.7,
          location: 'Skin Health Center',
          availability: 'Mon-Fri, 11 AM - 7 PM',
          image: 'https://images.unsplash.com/photo-1622253692010-333f2da6031d?w=300&h=300&fit=crop&crop=faces',
          email: 'james.williams@skincenter.com',
          phone: '+1-555-0104'
        },
        {
          id: 105,
          name: 'Dr. Priya Patel',
          specialization: 'Pediatric Dermatology',
          experience: '8 years',
          rating: 4.9,
          location: "Children's Skin Clinic",
          availability: 'Mon-Thu, 9 AM - 5 PM',
          image: 'https://images.unsplash.com/photo-1651008376811-b90baee60c1f?w=300&h=300&fit=crop&crop=faces',
          email: 'priya.patel@childclinic.com',
          phone: '+1-555-0105'
        },
        {
          id: 106,
          name: 'Dr. Robert Anderson',
          specialization: 'Mohs Surgery Specialist',
          experience: '20 years',
          rating: 4.8,
          location: 'Advanced Dermatology',
          availability: 'Wed-Sun, 8 AM - 3 PM',
          image: 'https://images.unsplash.com/photo-1537368910025-700350fe46c7?w=300&h=300&fit=crop&crop=faces',
          email: 'robert.anderson@advderm.com',
          phone: '+1-555-0106'
        },
        {
          id: 107,
          name: 'Dr. Shyam',
          specialization: 'MD Dermatology, MBBS',
          experience: '14 years',
          rating: 4.9,
          location: 'Digital Health Institute',
          availability: 'Mon-Fri, 10 AM - 6 PM',
          image: 'https://images.unsplash.com/photo-1582750433449-648ed127bb54?w=300&h=300&fit=crop&crop=faces',
          email: 'skandashyam102@gmail.com',
          phone: '+1-555-0107',
          realEmail: true
        },
        {
          id: 108,
          name: 'Dr. Kaushik',
          specialization: 'Surgical Oncology & Dermatology',
          experience: '16 years',
          rating: 4.8,
          location: 'Advanced Cancer Care',
          availability: 'Tue-Sat, 9 AM - 5 PM',
          image: 'https://images.unsplash.com/photo-1612349317150-e413f6a5b16d?w=300&h=300&fit=crop&crop=faces',
          email: 'kaushikmuliya@gmail.com',
          phone: '+1-555-0108',
          realEmail: true
        },
        {
          id: 109,
          name: 'Dr. Paavani',
          specialization: 'Cosmetic & Medical Dermatology',
          experience: '11 years',
          rating: 5.0,
          location: 'Skin Wellness Center',
          availability: 'Mon-Thu, 11 AM - 7 PM',
          image: 'https://images.unsplash.com/photo-1594824476967-48c8b964273f?w=300&h=300&fit=crop&crop=faces',
          email: 'kpaavani20@gmail.com',
          phone: '+1-555-0109',
          realEmail: true
        },
        {
          id: 110,
          name: 'Dr. Deepa',
          specialization: 'Dermatopathology & Melanoma Research',
          experience: '19 years',
          rating: 4.9,
          location: 'Research Dermatology Clinic',
          availability: 'Mon-Fri, 8 AM - 4 PM',
          image: 'https://images.unsplash.com/photo-1559839734-2b71ea197ec2?w=300&h=300&fit=crop&crop=faces',
          email: 'deepamk725@gmail.com',
          phone: '+1-555-0110',
          realEmail: true
        }
      ];
      setDoctors(dummyDoctors);
      toast.warn('Loaded dummy doctors due to API issue');
    } finally {
      setLoading(false);
    }
  };

  const handleSelectDoctor = async (doctor) => {
    const reportFilename = localStorage.getItem('reportFilename');
    const patientInfo = localStorage.getItem('patientInfo');

    if (!reportFilename) {
      toast.warning('Please generate a report first from the Results page');
      return;
    }

    setSelectedDoctor(doctor);
    setSending(true);

    try {
      const response = await axios.post('http://localhost:5001/consult-doctor', {
        doctor_id: doctor.id,
        report_filename: reportFilename,
        patient_info: JSON.parse(patientInfo || '{}')
      });

      toast.success(
        <div>
          <p className="font-semibold">Report sent successfully!</p>
          <p className="text-sm">Dr. {doctor.name} will review your case shortly.</p>
          <p className="text-xs mt-1">Consultation ID: {response.data.consultation_id}</p>
        </div>,
        { autoClose: 5000 }
      );
      
      setSelectedDoctor(null);
    } catch (error) {
      console.error('Error:', error);
      toast.error('Failed to send report. Please try again.');
    } finally {
      setSending(false);
    }
  };

  if (loading) {
    return (
      <div className="container mx-auto px-6 py-12 text-center">
        <div className="spinner mx-auto border-cyan-400 border-t-transparent"></div>
        <p className="mt-4 text-gray-700 dark:text-gray-300">Loading doctors...</p>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-6 py-12">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-cyan-600 to-blue-600 dark:from-cyan-400 dark:to-blue-500 bg-clip-text text-transparent">
            Consult Our Expert Doctors
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
            Connect with board-certified dermatologists and oncologists for professional consultation
          </p>
        </div>

        {/* Doctors Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          {doctors.map((doctor, index) => (
            <motion.div
              key={doctor.id}
              className="bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800 rounded-xl shadow-2xl shadow-cyan-500/10 p-6 hover:shadow-cyan-500/20 hover:scale-105 transition-all duration-300 border border-gray-200 dark:border-gray-700"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
            >
              {/* Doctor Info Header */}
              <div className="mb-4 pb-4 border-b border-gray-200 dark:border-gray-700">
                <h3 className="text-xl font-bold text-cyan-600 dark:text-cyan-400 mb-2">{doctor.name}</h3>
                <p className="text-blue-600 dark:text-blue-400 font-medium">{doctor.specialization}</p>
              </div>

              {/* Doctor Details */}
              <div className="space-y-3 mb-6">
                <div className="flex items-center text-gray-700 dark:text-gray-300">
                  <FaStar className="text-yellow-400 mr-2" />
                  <span className="font-semibold">{doctor.rating}</span>
                  <span className="text-gray-500 dark:text-gray-500 ml-1">/5.0 Rating</span>
                </div>

                <div className="flex items-start text-gray-700 dark:text-gray-300">
                  <FaMapMarkerAlt className="text-cyan-400 mr-2 mt-1 flex-shrink-0" />
                  <span className="text-sm">{doctor.location}</span>
                </div>

                <div className="flex items-start text-gray-700 dark:text-gray-300">
                  <FaClock className="text-green-400 mr-2 mt-1 flex-shrink-0" />
                  <span className="text-sm">{doctor.availability}</span>
                </div>

                <div className="flex items-center text-gray-700 dark:text-gray-300">
                  <FaCheckCircle className="text-purple-400 mr-2" />
                  <span className="text-sm font-medium">{doctor.experience} Experience</span>
                </div>
              </div>

              {/* Contact Info */}
              <div className="border-t border-gray-200 dark:border-gray-700 pt-4 mb-4 space-y-2">
                <div className="flex items-center text-sm text-gray-600 dark:text-gray-400">
                  <FaEnvelope className="mr-2 text-cyan-400" />
                  <span className="truncate">{doctor.email}</span>
                </div>
                <div className="flex items-center text-sm text-gray-600 dark:text-gray-400">
                  <FaPhone className="mr-2 text-cyan-400" />
                  <span>{doctor.phone}</span>
                </div>
              </div>

              {/* Action Button */}
              <button
                onClick={() => handleSelectDoctor(doctor)}
                disabled={sending && selectedDoctor?.id === doctor.id}
                className="w-full bg-gradient-to-r from-cyan-500 to-blue-600 text-white py-3 rounded-lg font-semibold shadow-lg shadow-cyan-500/50 hover:shadow-2xl hover:shadow-cyan-400/60 hover:scale-105 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {sending && selectedDoctor?.id === doctor.id ? (
                  <span className="flex items-center justify-center">
                    <div className="spinner mr-2 border-2 border-white border-t-transparent"></div>
                    Sending...
                  </span>
                ) : (
                  'Send Report & Consult'
                )}
              </button>
            </motion.div>
          ))}
        </div>

        {/* Information Box */}
        <motion.div
          className="mt-12 max-w-4xl mx-auto"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.5 }}
        >
          <div className="bg-cyan-500/10 border border-cyan-500/30 border-l-4 border-l-cyan-500 p-6 rounded-lg shadow-lg shadow-cyan-500/10">
            <h4 className="text-lg font-semibold text-cyan-600 dark:text-cyan-400 mb-2">
              📋 How It Works
            </h4>
            <ol className="list-decimal list-inside space-y-2 text-gray-700 dark:text-gray-300">
              <li>Select a doctor from the list above</li>
              <li>Your medical report will be automatically sent to them</li>
              <li>The doctor will review your case and contact you within 24-48 hours</li>
              <li>You'll receive consultation via email or phone based on your preference</li>
            </ol>
            <p className="mt-4 text-sm text-gray-600 dark:text-gray-400">
              <strong className="text-cyan-600 dark:text-cyan-400">Note:</strong> Make sure you have generated your medical report from the Results page before consulting a doctor.
            </p>
          </div>
        </motion.div>
      </motion.div>
    </div>
  );
};

export default Doctors;

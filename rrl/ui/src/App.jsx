// RRL Training Dashboard - Main App
// src/App.jsx

import React from 'react';
import { BrowserRouter, Routes, Route, Link, useLocation } from 'react-router-dom';
import { Activity, Database, Play, Settings, BarChart3, Upload, BookOpen } from 'lucide-react';

// 🔥 NEW: Framer Motion
import { motion, AnimatePresence } from 'framer-motion';

import Dashboard from './pages/Dashboard';
import Models from './pages/Models';
import Training from './pages/Training';
import Evaluation from './pages/Evaluation';
import Inference from './pages/Inference';
import DataUpload from './pages/DataUpload';
import RAG from './pages/RAG';

function App() {
  return (
    <BrowserRouter>
      <AnimatedLayout />
    </BrowserRouter>
  );
}


function AnimatedLayout() {
  const location = useLocation();

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Navigation */}
      <nav className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex">
              <div className="flex-shrink-0 flex items-center">
                <Activity className="h-8 w-8 text-blue-600" />
                <span className="ml-2 text-xl font-bold text-gray-900">RRL Training</span>
              </div>
              <div className="hidden sm:ml-6 sm:flex sm:space-x-8">
                <NavLink to="/" icon={BarChart3}>Dashboard</NavLink>
                <NavLink to="/models" icon={Database}>Models</NavLink>
                <NavLink to="/training" icon={Play}>Training</NavLink>
                <NavLink to="/evaluation" icon={Settings}>Evaluation</NavLink>
                <NavLink to="/inference" icon={Activity}>Inference</NavLink>
                <NavLink to="/rag" icon={BookOpen}>RAG</NavLink>
                <NavLink to="/upload" icon={Upload}>Upload</NavLink>
              </div>
            </div>
          </div>
        </div>
      </nav>

      {/* Animated Page Transitions */}
      <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        <AnimatePresence mode="wait">
          <motion.div
            key={location.pathname}
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.98 }}
            transition={{ duration: 0.25, ease: 'easeOut' }}
          >
            <Routes location={location}>
              <Route path="/" element={<Dashboard />} />
              <Route path="/models" element={<Models />} />
              <Route path="/training" element={<Training />} />
              <Route path="/evaluation" element={<Evaluation />} />
              <Route path="/inference" element={<Inference />} />
              <Route path="/rag" element={<RAG />} />
              <Route path="/upload" element={<DataUpload />} />
            </Routes>
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  );
}

function NavLink({ to, icon: Icon, children }) {
  return (
    <Link
      to={to}
      className="inline-flex items-center px-1 pt-1 border-b-2 border-transparent text-sm font-medium
                 text-gray-500 hover:text-gray-700 hover:border-gray-300 transition-colors"
    >
      <Icon className="h-4 w-4 mr-2" />
      {children}
    </Link>
  );
}

export default App;

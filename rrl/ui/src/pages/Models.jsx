// Models Page - Browse available models
// src/pages/Models.jsx

import React, { useState, useEffect } from 'react';
import { Database, Info } from 'lucide-react';
import api from '../api';

export default function Models() {
  const [models, setModels] = useState([]);

  useEffect(() => {
    loadModels();
  }, []);

  const loadModels = async () => {
    try {
      const data = await api.listModels();
      setModels(data.models);
    } catch (error) {
      console.error('Failed to load models:', error);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold text-gray-900">Available Models</h1>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {models.map(model => (
          <ModelCard key={model.name} model={model} />
        ))}
      </div>
    </div>
  );
}

function ModelCard({ model }) {
  return (
    <div className="bg-white shadow rounded-lg p-6 hover:shadow-lg transition-shadow">
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-3">
          <Database className="h-8 w-8 text-blue-600" />
          <div>
            <h3 className="text-lg font-semibold text-gray-900">{model.name}</h3>
            <p className="text-sm text-gray-500">{model.architecture}</p>
          </div>
        </div>
      </div>

      <div className="mt-4 space-y-2">
        <InfoRow label="Size" value={model.size} />
        <InfoRow label="Hidden Size" value={model.hidden_size} />
        <InfoRow label="Recommended Rank" value={model.recommended_rank} />
      </div>
    </div>
  );
}

function InfoRow({ label, value }) {
  return (
    <div className="flex justify-between text-sm">
      <span className="text-gray-500">{label}:</span>
      <span className="font-medium text-gray-900">{value}</span>
    </div>
  );
}



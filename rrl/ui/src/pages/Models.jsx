// Models Page - Manage embedding and generator models
// src/pages/Models.jsx

import React, { useState, useEffect } from 'react';
import { Database, Cpu, Plus, Trash2, RefreshCw, CheckCircle, Loader2 } from 'lucide-react';
import api from '../api';

export default function Models() {
  const [embeddingModels, setEmbeddingModels] = useState([]);
  const [generatorModels, setGeneratorModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Add model form state
  const [showAddForm, setShowAddForm] = useState(false);
  const [newModelName, setNewModelName] = useState('');
  const [newModelPath, setNewModelPath] = useState('');
  const [addLoading, setAddLoading] = useState(false);

  useEffect(() => {
    loadAllModels();
  }, []);

  const loadAllModels = async () => {
    setLoading(true);
    try {
      const [embeddingData, generatorData] = await Promise.all([
        api.listModels(),
        api.listGeneratorModels(),
      ]);
      setEmbeddingModels(embeddingData.models || []);
      setGeneratorModels(generatorData.models || []);
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleAddModel = async (e) => {
    e.preventDefault();
    if (!newModelName.trim() || !newModelPath.trim()) {
      setError('Please provide both name and path');
      return;
    }

    setAddLoading(true);
    setError(null);
    try {
      await api.addGeneratorModel(newModelName.trim(), newModelPath.trim());
      const data = await api.listGeneratorModels();
      setGeneratorModels(data.models || []);
      setShowAddForm(false);
      setNewModelName('');
      setNewModelPath('');
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setAddLoading(false);
    }
  };

  const handleRemoveModel = async (modelName) => {
    if (!confirm(`Remove model "${modelName}"?`)) return;
    try {
      await api.removeGeneratorModel(modelName);
      const data = await api.listGeneratorModels();
      setGeneratorModels(data.models || []);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    }
  };

  const handleScanFinetuned = async () => {
    try {
      await api.scanFinetunedModels('./output');
      const data = await api.listGeneratorModels();
      setGeneratorModels(data.models || []);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-8 w-8 animate-spin text-blue-600" />
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <div className="flex items-center gap-3">
        <Database className="h-8 w-8 text-blue-600" />
        <h1 className="text-3xl font-bold text-gray-900">Models</h1>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-800">
          {error}
        </div>
      )}

      {/* Generator Models (LLMs) Section */}
      <section>
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Cpu className="h-6 w-6 text-purple-600" />
            <h2 className="text-xl font-semibold text-gray-900">Generator Models (LLMs)</h2>
          </div>
          <div className="flex gap-2">
            <button
              onClick={handleScanFinetuned}
              className="flex items-center gap-2 px-3 py-2 text-sm border border-gray-300 rounded-lg hover:bg-gray-50"
              title="Scan for fine-tuned models"
            >
              <RefreshCw className="h-4 w-4" />
              Scan Fine-tuned
            </button>
            <button
              onClick={() => setShowAddForm(!showAddForm)}
              className="flex items-center gap-2 px-4 py-2 text-sm bg-purple-600 text-white rounded-lg hover:bg-purple-700"
            >
              <Plus className="h-4 w-4" />
              Add Model
            </button>
          </div>
        </div>

        {/* Add Model Form */}
        {showAddForm && (
          <form onSubmit={handleAddModel} className="bg-purple-50 border border-purple-200 rounded-lg p-4 mb-4">
            <h3 className="font-medium text-purple-900 mb-3">Add New Generator Model</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Model Name</label>
                <input
                  type="text"
                  value={newModelName}
                  onChange={(e) => setNewModelName(e.target.value)}
                  placeholder="e.g., Qwen2.5-7B-Instruct"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Model Path</label>
                <input
                  type="text"
                  value={newModelPath}
                  onChange={(e) => setNewModelPath(e.target.value)}
                  placeholder="~/models/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/..."
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg"
                />
              </div>
            </div>
            <div className="flex gap-2 mt-4">
              <button
                type="submit"
                disabled={addLoading}
                className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:bg-gray-400"
              >
                {addLoading ? 'Adding...' : 'Add Model'}
              </button>
              <button
                type="button"
                onClick={() => setShowAddForm(false)}
                className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50"
              >
                Cancel
              </button>
            </div>
          </form>
        )}

        {/* Generator Models Grid */}
        {generatorModels.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {generatorModels.map(model => (
              <GeneratorModelCard
                key={model.name}
                model={model}
                onRemove={() => handleRemoveModel(model.name)}
              />
            ))}
          </div>
        ) : (
          <div className="bg-gray-50 border border-gray-200 rounded-lg p-8 text-center">
            <Cpu className="h-12 w-12 text-gray-400 mx-auto mb-3" />
            <p className="text-gray-600">No generator models added yet.</p>
            <p className="text-sm text-gray-500 mt-1">
              Click "Add Model" to register a local model or HuggingFace model.
            </p>
          </div>
        )}
      </section>

      {/* Embedding Models Section */}
      <section>
        <div className="flex items-center gap-2 mb-4">
          <Database className="h-6 w-6 text-blue-600" />
          <h2 className="text-xl font-semibold text-gray-900">Embedding Models</h2>
          <span className="text-xs bg-gray-100 text-gray-600 px-2 py-1 rounded">Built-in</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {embeddingModels.map(model => (
            <EmbeddingModelCard key={model.name} model={model} />
          ))}
        </div>
      </section>
    </div>
  );
}

function GeneratorModelCard({ model, onRemove }) {
  return (
    <div className="bg-white shadow rounded-lg p-4 hover:shadow-md transition-shadow border border-gray-200">
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-3">
          <Cpu className="h-6 w-6 text-purple-600" />
          <div>
            <h3 className="font-semibold text-gray-900">{model.name}</h3>
            <div className="flex items-center gap-2 mt-1">
              <span className="text-xs text-gray-500">{model.architecture}</span>
              {model.size !== 'unknown' && (
                <span className="text-xs text-gray-400">{model.size}</span>
              )}
              {model.is_finetuned && (
                <span className="text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded flex items-center gap-1">
                  <CheckCircle className="h-3 w-3" />
                  Fine-tuned
                </span>
              )}
            </div>
          </div>
        </div>
        <button
          onClick={onRemove}
          className="text-gray-400 hover:text-red-500 p-1"
          title="Remove model"
        >
          <Trash2 className="h-4 w-4" />
        </button>
      </div>

      <div className="mt-3 pt-3 border-t border-gray-100">
        <p className="text-xs text-gray-500 truncate" title={model.path}>
          {model.path}
        </p>
        {model.checkpoint_path && (
          <p className="text-xs text-green-600 truncate mt-1" title={model.checkpoint_path}>
            Checkpoint: {model.checkpoint_path}
          </p>
        )}
      </div>
    </div>
  );
}

function EmbeddingModelCard({ model }) {
  return (
    <div className="bg-white shadow rounded-lg p-4 hover:shadow-md transition-shadow border border-gray-200">
      <div className="flex items-center gap-3">
        <Database className="h-6 w-6 text-blue-600" />
        <div>
          <h3 className="font-semibold text-gray-900">{model.name}</h3>
          <p className="text-xs text-gray-500">{model.architecture}</p>
        </div>
      </div>

      <div className="mt-3 pt-3 border-t border-gray-100 space-y-1">
        <InfoRow label="Size" value={model.size} />
        <InfoRow label="Hidden" value={model.hidden_size} />
        <InfoRow label="LoRA Rank" value={model.recommended_rank} />
      </div>
    </div>
  );
}

function InfoRow({ label, value }) {
  return (
    <div className="flex justify-between text-xs">
      <span className="text-gray-500">{label}:</span>
      <span className="font-medium text-gray-700">{value}</span>
    </div>
  );
}

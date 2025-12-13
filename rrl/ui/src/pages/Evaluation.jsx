// src/pages/Evaluation.jsx
import React, { useState } from 'react';
import { BarChart3, Play, CheckCircle } from 'lucide-react';
import api from '../api';

export default function Evaluation() {
  const [config, setConfig] = useState({
    model_name: 'bert-base-uncased',
    checkpoint_path: '',
    data_path: '',
    lora_rank: 8,
    lora_alpha: 16.0,
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleEvaluate = async () => {
    if (!config.data_path) {
      setError('Please provide a test data path');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await api.runEvaluation(config);
      setResult(data);
    } catch (err) {
      console.error('Evaluation failed:', err);
      setError(err.message || 'Evaluation failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <BarChart3 className="h-8 w-8 text-blue-600" />
        <h1 className="text-3xl font-bold text-gray-900">Model Evaluation</h1>
      </div>

      <div className="bg-white shadow rounded-lg p-6 space-y-6">
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Model Name
            </label>
            <input
              type="text"
              placeholder="bert-base-uncased"
              value={config.model_name}
              onChange={(e) => setConfig({...config, model_name: e.target.value})}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
            <p className="mt-1 text-sm text-gray-500">
              Enter the model identifier (e.g., bert-base-uncased, roberta-base)
            </p>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Test Data Path *
            </label>
            <input
              type="text"
              placeholder="./data/test.jsonl"
              value={config.data_path}
              onChange={(e) => setConfig({...config, data_path: e.target.value})}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
            <p className="mt-1 text-sm text-gray-500">
              Path to your test dataset (JSONL format)
            </p>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Checkpoint Path (Optional)
            </label>
            <input
              type="text"
              placeholder="./outputs/checkpoint-500/lora_weights.safetensors"
              value={config.checkpoint_path}
              onChange={(e) => setConfig({...config, checkpoint_path: e.target.value})}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
            <p className="mt-1 text-sm text-gray-500">
              Leave empty to evaluate base model without fine-tuning
            </p>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                LoRA Rank
              </label>
              <input
                type="number"
                value={config.lora_rank}
                onChange={(e) => setConfig({...config, lora_rank: parseInt(e.target.value)})}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                LoRA Alpha
              </label>
              <input
                type="number"
                step="0.1"
                value={config.lora_alpha}
                onChange={(e) => setConfig({...config, lora_alpha: parseFloat(e.target.value)})}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>
        </div>

        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4">
            <p className="text-red-800 text-sm">{error}</p>
          </div>
        )}

        <button
          onClick={handleEvaluate}
          disabled={loading || !config.data_path}
          className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed font-medium transition-colors"
        >
          {loading ? (
            <>
              <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
              Evaluating...
            </>
          ) : (
            <>
              <Play className="h-5 w-5" />
              Run Evaluation
            </>
          )}
        </button>
      </div>

      {result && (
        <div className="bg-white shadow rounded-lg p-6">
          <div className="flex items-center gap-2 mb-4">
            <CheckCircle className="h-6 w-6 text-green-600" />
            <h2 className="text-xl font-semibold">Evaluation Results</h2>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {result.accuracy !== null && result.accuracy !== undefined && (
              <div className="text-center p-6 bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg">
                <p className="text-5xl font-bold text-blue-600">
                  {result.accuracy?.toFixed(2)}%
                </p>
                <p className="text-sm text-gray-600 mt-2 font-medium">Accuracy</p>
                <p className="text-xs text-gray-500 mt-1">
                  Percentage of correct predictions
                </p>
              </div>
            )}
            
            {result.mrr !== null && result.mrr !== undefined && (
              <div className="text-center p-6 bg-gradient-to-br from-green-50 to-green-100 rounded-lg">
                <p className="text-5xl font-bold text-green-600">
                  {result.mrr?.toFixed(4)}
                </p>
                <p className="text-sm text-gray-600 mt-2 font-medium">MRR</p>
                <p className="text-xs text-gray-500 mt-1">
                  Mean Reciprocal Rank
                </p>
              </div>
            )}
          </div>

          {result.output && (
            <div className="mt-6">
              <h3 className="text-sm font-medium text-gray-700 mb-2">Full Output</h3>
              <pre className="bg-gray-900 text-green-400 p-4 rounded-lg overflow-auto text-xs max-h-64">
                {result.output}
              </pre>
            </div>
          )}
        </div>
      )}

      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <h3 className="text-sm font-medium text-blue-900 mb-2">💡 Tips</h3>
        <ul className="text-sm text-blue-700 space-y-1 list-disc list-inside">
          <li>Ensure your test data is in JSONL format</li>
          <li>Use the same LoRA configuration as training</li>
          <li>Checkpoint path should point to the .safetensors file</li>
          <li>Base model evaluation runs without checkpoint</li>
        </ul>
      </div>
    </div>
  );
}
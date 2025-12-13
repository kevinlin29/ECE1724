// src/pages/Inference.jsx
import React, { useState } from 'react';
import { Zap, Send, Loader2 } from 'lucide-react';
import api from '../api';

export default function Inference() {
  const [query, setQuery] = useState('');
  const [modelName, setModelName] = useState('bert-base-uncased');
  const [checkpointPath, setCheckpointPath] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleInference = async () => {
    if (!query.trim()) {
      setError('Please enter a query');
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      const data = await api.runInference({
        model_name: modelName,
        checkpoint_path: checkpointPath || null,
        queries: [query],
      });
      setResults(data.results);
    } catch (err) {
      console.error('Inference failed:', err);
      setError(err.message || 'Inference failed');
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && e.ctrlKey) {
      handleInference();
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <Zap className="h-8 w-8 text-blue-600" />
        <h1 className="text-3xl font-bold text-gray-900">Inference Playground</h1>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column - Input */}
        <div className="lg:col-span-2 space-y-6">
          <div className="bg-white shadow rounded-lg p-6 space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Query Text
              </label>
              <textarea
                placeholder="Enter your text here...&#10;&#10;Example: &quot;What is machine learning?&quot;"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyPress={handleKeyPress}
                rows={8}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 font-mono text-sm"
              />
              <p className="mt-1 text-xs text-gray-500">
                Press Ctrl+Enter to run inference
              </p>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Model
                </label>
                <select
                  value={modelName}
                  onChange={(e) => setModelName(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                >
                  <option value="bert-base-uncased">BERT Base</option>
                  <option value="roberta-base">RoBERTa Base</option>
                  <option value="BAAI/bge-base-en-v1.5">BGE Base</option>
                  <option value="BAAI/bge-large-en-v1.5">BGE Large</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Checkpoint (Optional)
                </label>
                <input
                  type="text"
                  placeholder="./outputs/checkpoint-500"
                  value={checkpointPath}
                  onChange={(e) => setCheckpointPath(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                />
              </div>
            </div>

            {error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-3">
                <p className="text-red-800 text-sm">{error}</p>
              </div>
            )}
            
            <button
              onClick={handleInference}
              disabled={loading || !query.trim()}
              className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed font-medium transition-colors"
            >
              {loading ? (
                <>
                  <Loader2 className="h-5 w-5 animate-spin" />
                  Processing...
                </>
              ) : (
                <>
                  <Send className="h-5 w-5" />
                  Run Inference
                </>
              )}
            </button>
          </div>
        </div>

        {/* Right Column - Results */}
        <div className="space-y-6">
          <div className="bg-white shadow rounded-lg p-6">
            <h2 className="text-lg font-semibold mb-4">Results</h2>
            
            {results.length === 0 ? (
              <div className="text-center py-8">
                <Zap className="h-12 w-12 text-gray-300 mx-auto mb-3" />
                <p className="text-gray-500 text-sm">
                  Results will appear here
                </p>
              </div>
            ) : (
              <div className="space-y-4">
                {results.map((result, idx) => (
                  <div key={idx} className="border rounded-lg p-4">
                    <h3 className="text-sm font-medium text-gray-700 mb-2">
                      Query {idx + 1}
                    </h3>
                    <p className="text-xs text-gray-600 mb-3 italic">
                      &quot;{result.query}&quot;
                    </p>
                    
                    {result.embedding && (
                      <div>
                        <p className="text-xs font-medium text-gray-700 mb-1">
                          Embedding Vector
                        </p>
                        <div className="bg-gray-50 rounded p-2 max-h-32 overflow-auto">
                          <p className="text-xs font-mono text-gray-600">
                            [{result.embedding.slice(0, 10).map(v => v.toFixed(4)).join(', ')}...]
                          </p>
                          <p className="text-xs text-gray-500 mt-1">
                            Dimension: {result.embedding.length}
                          </p>
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <h3 className="text-sm font-medium text-blue-900 mb-2">💡 Usage</h3>
            <ul className="text-xs text-blue-700 space-y-1 list-disc list-inside">
              <li>Enter text for embedding</li>
              <li>Choose a model variant</li>
              <li>Optionally use fine-tuned checkpoint</li>
              <li>Get embedding vectors</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Examples */}
      <div className="bg-white shadow rounded-lg p-6">
        <h2 className="text-lg font-semibold mb-4">Example Queries</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {[
            "What is artificial intelligence?",
            "How do neural networks work?",
            "Explain transformer architecture",
          ].map((example, idx) => (
            <button
              key={idx}
              onClick={() => setQuery(example)}
              className="text-left p-3 border border-gray-200 rounded-lg hover:border-blue-500 hover:bg-blue-50 transition-colors"
            >
              <p className="text-sm text-gray-700">{example}</p>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
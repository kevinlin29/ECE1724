// src/pages/Inference.jsx
import React, { useState, useEffect } from 'react';
import { Zap, Send, Loader2, Sparkles, FileText } from 'lucide-react';
import api from '../api';

export default function Inference() {
  const [query, setQuery] = useState('');
  const [modelName, setModelName] = useState('BAAI/bge-base-en-v1.5');
  const [checkpointPath, setCheckpointPath] = useState('');
  const [checkpoints, setCheckpoints] = useState([]);
  const [inferenceType, setInferenceType] = useState('generation'); // 'generation' or 'embedding'
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadCheckpoints();
  }, []);

  const loadCheckpoints = async () => {
    try {
      const data = await api.listCheckpoints();
      setCheckpoints(data.checkpoints || []);
    } catch (err) {
      console.error('Failed to load checkpoints:', err);
    }
  };

  const handleInference = async () => {
    if (!query.trim()) {
      setError('Please enter a query');
      return;
    }
    
    setLoading(true);
    setError(null);
    setResults(null);
    
    try {
      const data = await api.runInference({
        model_name: modelName,
        checkpoint_path: checkpointPath || null,
        queries: [query],
        inference_type: inferenceType,
      });
      setResults(data);
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
        <h1 className="text-3xl font-bold text-gray-900">LLM Inference</h1>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column - Input */}
        <div className="lg:col-span-2 space-y-6">
          <div className="bg-white shadow rounded-lg p-6 space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Query / Prompt
              </label>
              <textarea
                placeholder="Enter your query or prompt here...&#10;&#10;For generation: &quot;Explain the benefits of RAG systems&quot;&#10;For embedding: &quot;What is machine learning?&quot;"
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

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Model
                </label>
                <select
                  value={modelName}
                  onChange={(e) => setModelName(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                >
                  <option value="BAAI/bge-base-en-v1.5">BGE Base</option>
                  <option value="BAAI/bge-large-en-v1.5">BGE Large</option>
                  <option value="bert-base-uncased">BERT Base</option>
                  <option value="roberta-base">RoBERTa Base</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Inference Type
                </label>
                <select
                  value={inferenceType}
                  onChange={(e) => setInferenceType(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                >
                  <option value="generation">Text Generation</option>
                  <option value="embedding">Embedding Only</option>
                </select>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Fine-tuned Checkpoint (Optional)
              </label>
              <select
                value={checkpointPath}
                onChange={(e) => setCheckpointPath(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              >
                <option value="">Base Model (No Fine-tuning)</option>
                {checkpoints.map((ckpt, idx) => (
                  <option key={idx} value={ckpt.path}>
                    {ckpt.name} ({new Date(ckpt.created_at).toLocaleDateString()})
                  </option>
                ))}
              </select>
              <p className="mt-1 text-xs text-gray-500">
                Select a checkpoint to use your fine-tuned model
              </p>
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
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Sparkles className="h-5 w-5 text-blue-600" />
              Results
            </h2>
            
            {!results && !loading && (
              <div className="text-center py-8">
                <Zap className="h-12 w-12 text-gray-300 mx-auto mb-3" />
                <p className="text-gray-500 text-sm">
                  Results will appear here
                </p>
              </div>
            )}

            {loading && (
              <div className="text-center py-8">
                <Loader2 className="h-12 w-12 text-blue-600 mx-auto mb-3 animate-spin" />
                <p className="text-gray-600 text-sm">
                  Generating response...
                </p>
              </div>
            )}

            {results && (
              <div className="space-y-4">
                {/* Generation Response */}
                {inferenceType === 'generation' && results.generated_text && (
                  <div className="border rounded-lg p-4 bg-gradient-to-br from-blue-50 to-indigo-50">
                    <h3 className="text-sm font-medium text-gray-700 mb-2 flex items-center gap-2">
                      <FileText className="h-4 w-4 text-blue-600" />
                      Generated Response
                    </h3>
                    <div className="bg-white rounded p-3 border border-blue-100">
                      <p className="text-sm text-gray-800 whitespace-pre-wrap">
                        {results.generated_text}
                      </p>
                    </div>
                    {results.generation_metrics && (
                      <div className="mt-3 pt-3 border-t border-blue-100">
                        <div className="grid grid-cols-2 gap-2 text-xs">
                          {results.generation_metrics.tokens && (
                            <div>
                              <span className="text-gray-600">Tokens:</span>
                              <span className="ml-1 font-medium">{results.generation_metrics.tokens}</span>
                            </div>
                          )}
                          {results.generation_metrics.time && (
                            <div>
                              <span className="text-gray-600">Time:</span>
                              <span className="ml-1 font-medium">{results.generation_metrics.time}s</span>
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* Embedding Vector */}
                {results.embedding && (
                  <div className="border rounded-lg p-4">
                    <h3 className="text-sm font-medium text-gray-700 mb-2">
                      Embedding Vector
                    </h3>
                    <div className="bg-gray-50 rounded p-3 max-h-32 overflow-auto">
                      <p className="text-xs font-mono text-gray-600">
                        [{results.embedding.slice(0, 10).map(v => v.toFixed(4)).join(', ')}...]
                      </p>
                      <p className="text-xs text-gray-500 mt-2">
                        Dimension: {results.embedding.length}
                      </p>
                    </div>
                  </div>
                )}

                {/* Model Info */}
                <div className="border rounded-lg p-4 bg-gray-50">
                  <h3 className="text-sm font-medium text-gray-700 mb-2">
                    Model Information
                  </h3>
                  <div className="space-y-1 text-xs">
                    <div className="flex justify-between">
                      <span className="text-gray-600">Model:</span>
                      <span className="font-medium text-gray-900">{modelName}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Checkpoint:</span>
                      <span className="font-medium text-gray-900">
                        {checkpointPath ? checkpointPath.split('/').pop() : 'Base Model'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Type:</span>
                      <span className="font-medium text-gray-900">
                        {inferenceType === 'generation' ? 'Generation' : 'Embedding'}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Similarity Scores (if available) */}
                {results.similarity_scores && (
                  <div className="border rounded-lg p-4">
                    <h3 className="text-sm font-medium text-gray-700 mb-2">
                      Similarity Scores
                    </h3>
                    <div className="space-y-2">
                      {results.similarity_scores.map((score, idx) => (
                        <div key={idx} className="flex items-center gap-2">
                          <div className="flex-1 bg-gray-200 rounded-full h-2">
                            <div
                              className="bg-blue-600 h-2 rounded-full transition-all"
                              style={{ width: `${score * 100}%` }}
                            />
                          </div>
                          <span className="text-xs font-medium text-gray-700 w-12">
                            {(score * 100).toFixed(1)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <h3 className="text-sm font-medium text-blue-900 mb-2">💡 Usage Tips</h3>
            <ul className="text-xs text-blue-700 space-y-1 list-disc list-inside">
              <li>Use generation for text responses</li>
              <li>Use embedding for vector representation</li>
              <li>Select checkpoint to test fine-tuned models</li>
              <li>Base model runs without fine-tuning</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Example Queries */}
      <div className="bg-white shadow rounded-lg p-6">
        <h2 className="text-lg font-semibold mb-4">Example Queries</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {[
            {
              text: "Explain the benefits of using RAG systems in production",
              type: "generation"
            },
            {
              text: "What are the key differences between BERT and RoBERTa?",
              type: "generation"
            },
            {
              text: "How does LoRA fine-tuning work?",
              type: "embedding"
            },
          ].map((example, idx) => (
            <button
              key={idx}
              onClick={() => {
                setQuery(example.text);
                setInferenceType(example.type);
              }}
              className="text-left p-3 border border-gray-200 rounded-lg hover:border-blue-500 hover:bg-blue-50 transition-colors"
            >
              <p className="text-sm text-gray-700">{example.text}</p>
              <p className="text-xs text-gray-500 mt-1">
                Type: {example.type === 'generation' ? '🤖 Generation' : '📊 Embedding'}
              </p>
            </button>
          ))}
        </div>
      </div>

      {/* Model Comparison (if checkpoints available) */}
      {checkpoints.length > 0 && (
        <div className="bg-gradient-to-r from-purple-50 to-pink-50 border border-purple-200 rounded-lg p-6">
          <h2 className="text-lg font-semibold mb-3 text-purple-900">
            🔬 Compare Models
          </h2>
          <p className="text-sm text-purple-700 mb-3">
            You have {checkpoints.length} fine-tuned checkpoint{checkpoints.length > 1 ? 's' : ''} available. 
            Try the same query with different checkpoints to see how fine-tuning affects the output!
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
            {checkpoints.slice(0, 3).map((ckpt, idx) => (
              <button
                key={idx}
                onClick={() => setCheckpointPath(ckpt.path)}
                className={`p-2 rounded border text-left text-xs transition-colors ${
                  checkpointPath === ckpt.path
                    ? 'border-purple-500 bg-white shadow'
                    : 'border-purple-200 hover:border-purple-400 bg-white/50'
                }`}
              >
                <div className="font-medium text-purple-900">{ckpt.name}</div>
                <div className="text-purple-600 mt-1">
                  {new Date(ckpt.created_at).toLocaleDateString()}
                </div>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
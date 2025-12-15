// src/pages/Evaluation.jsx
import React, { useState, useEffect, useRef } from 'react';
import { BarChart3, Play, CheckCircle, XCircle, Clock, Database, StopCircle } from 'lucide-react';
import api from '../api';

const SAMPLE_OPTIONS = [
  { value: null, label: 'Full Dataset (10,047 queries)' },
  { value: 1000, label: '1,000 queries' },
  { value: 500, label: '500 queries' },
  { value: 100, label: '100 queries (Quick test)' },
];

export default function Evaluation() {
  // Evaluation mode: 'standard' | 'msmarco'
  const [evalMode, setEvalMode] = useState('msmarco');

  // Standard evaluation config
  const [config, setConfig] = useState({
    model_name: 'bert-base-uncased',
    checkpoint_path: '',
    data_path: '',
    lora_rank: 8,
    lora_alpha: 16.0,
  });

  // MS MARCO evaluation config
  const [msmarcoConfig, setMsmarcoConfig] = useState({
    model_name: 'bert-base-uncased',
    checkpoint_path: '',
    data_path: 'data/msmarco_validation.jsonl',
    sample_size: 100, // Default to quick test
    lora_rank: 8,
    lora_alpha: 16.0,
    device: 'auto',
  });

  // State
  const [result, setResult] = useState(null);
  const [msmarcoResult, setMsmarcoResult] = useState(null);
  const [msmarcoProgress, setMsmarcoProgress] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [jobId, setJobId] = useState(null);

  // WebSocket ref
  const wsRef = useRef(null);

  // Cleanup WebSocket on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  // Standard evaluation handler
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

  // MS MARCO evaluation handler
  const handleMsmarcoEvaluate = async () => {
    setLoading(true);
    setError(null);
    setMsmarcoResult(null);
    setMsmarcoProgress(null);

    try {
      // Start evaluation
      const response = await api.startMsmarcoEval(msmarcoConfig);
      const newJobId = response.job_id;
      setJobId(newJobId);

      // Connect WebSocket for progress updates
      wsRef.current = api.connectMsmarcoWebSocket(
        newJobId,
        // onProgress
        (progress) => {
          setMsmarcoProgress(progress);
        },
        // onComplete
        (result) => {
          setMsmarcoResult(result);
          setMsmarcoProgress(null);
          setLoading(false);
          if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
          }
        },
        // onError
        (error) => {
          setError(typeof error === 'string' ? error : 'Evaluation failed');
          setLoading(false);
          if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
          }
        }
      );
    } catch (err) {
      console.error('MS MARCO evaluation failed:', err);
      setError(err.message || 'Failed to start evaluation');
      setLoading(false);
    }
  };

  // Cancel MS MARCO evaluation
  const handleCancelEvaluation = async () => {
    if (jobId) {
      try {
        await api.cancelMsmarcoEval(jobId);
        setLoading(false);
        setMsmarcoProgress(null);
        if (wsRef.current) {
          wsRef.current.close();
          wsRef.current = null;
        }
      } catch (err) {
        console.error('Failed to cancel evaluation:', err);
      }
    }
  };

  // Calculate progress percentage
  const progressPercent = msmarcoProgress
    ? Math.round((msmarcoProgress.processed / msmarcoProgress.total) * 100)
    : 0;

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <BarChart3 className="h-8 w-8 text-blue-600" />
        <h1 className="text-3xl font-bold text-gray-900">Model Evaluation</h1>
      </div>

      {/* Mode Toggle Tabs */}
      <div className="bg-white shadow rounded-lg overflow-hidden">
        <div className="flex border-b border-gray-200">
          <button
            onClick={() => setEvalMode('msmarco')}
            className={`flex-1 px-4 py-3 text-sm font-medium transition-colors ${
              evalMode === 'msmarco'
                ? 'bg-blue-50 text-blue-700 border-b-2 border-blue-600'
                : 'text-gray-500 hover:text-gray-700 hover:bg-gray-50'
            }`}
          >
            <Database className="inline-block w-4 h-4 mr-2" />
            MS MARCO v1.1 Evaluation
          </button>
          <button
            onClick={() => setEvalMode('standard')}
            className={`flex-1 px-4 py-3 text-sm font-medium transition-colors ${
              evalMode === 'standard'
                ? 'bg-blue-50 text-blue-700 border-b-2 border-blue-600'
                : 'text-gray-500 hover:text-gray-700 hover:bg-gray-50'
            }`}
          >
            <BarChart3 className="inline-block w-4 h-4 mr-2" />
            Standard Evaluation
          </button>
        </div>

        <div className="p-6 space-y-6">
          {evalMode === 'msmarco' ? (
            // MS MARCO Evaluation Form
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Model Name
                  </label>
                  <input
                    type="text"
                    placeholder="bert-base-uncased"
                    value={msmarcoConfig.model_name}
                    onChange={(e) => setMsmarcoConfig({...msmarcoConfig, model_name: e.target.value})}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Sample Size
                  </label>
                  <select
                    value={msmarcoConfig.sample_size ?? ''}
                    onChange={(e) => setMsmarcoConfig({
                      ...msmarcoConfig,
                      sample_size: e.target.value ? parseInt(e.target.value) : null
                    })}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  >
                    {SAMPLE_OPTIONS.map((opt) => (
                      <option key={opt.label} value={opt.value ?? ''}>
                        {opt.label}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  LoRA Checkpoint Path (Optional)
                </label>
                <input
                  type="text"
                  placeholder="./outputs/checkpoint-500/lora_weights.safetensors"
                  value={msmarcoConfig.checkpoint_path}
                  onChange={(e) => setMsmarcoConfig({...msmarcoConfig, checkpoint_path: e.target.value})}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
                <p className="mt-1 text-sm text-gray-500">
                  Leave empty to evaluate base model without fine-tuning
                </p>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Validation Data Path
                </label>
                <input
                  type="text"
                  placeholder="data/msmarco_validation.jsonl"
                  value={msmarcoConfig.data_path}
                  onChange={(e) => setMsmarcoConfig({...msmarcoConfig, data_path: e.target.value})}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>

              <div className="grid grid-cols-3 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    LoRA Rank
                  </label>
                  <input
                    type="number"
                    value={msmarcoConfig.lora_rank}
                    onChange={(e) => setMsmarcoConfig({...msmarcoConfig, lora_rank: parseInt(e.target.value)})}
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
                    value={msmarcoConfig.lora_alpha}
                    onChange={(e) => setMsmarcoConfig({...msmarcoConfig, lora_alpha: parseFloat(e.target.value)})}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Device
                  </label>
                  <select
                    value={msmarcoConfig.device}
                    onChange={(e) => setMsmarcoConfig({...msmarcoConfig, device: e.target.value})}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="auto">Auto</option>
                    <option value="cuda">CUDA</option>
                    <option value="cpu">CPU</option>
                  </select>
                </div>
              </div>
            </div>
          ) : (
            // Standard Evaluation Form
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
          )}

          {/* Error Display */}
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-start gap-3">
              <XCircle className="h-5 w-5 text-red-500 flex-shrink-0 mt-0.5" />
              <p className="text-red-800 text-sm">{error}</p>
            </div>
          )}

          {/* Progress Display (MS MARCO only) */}
          {evalMode === 'msmarco' && loading && msmarcoProgress && (
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Clock className="h-5 w-5 text-blue-600 animate-pulse" />
                  <span className="text-sm font-medium text-blue-900">
                    Evaluating... {msmarcoProgress.processed} / {msmarcoProgress.total} queries
                  </span>
                </div>
                <span className="text-sm text-blue-700">
                  ETA: {Math.round(msmarcoProgress.eta_seconds)}s
                </span>
              </div>

              {/* Progress Bar */}
              <div className="w-full bg-blue-200 rounded-full h-3">
                <div
                  className="bg-blue-600 h-3 rounded-full transition-all duration-300"
                  style={{ width: `${progressPercent}%` }}
                />
              </div>

              <div className="flex justify-between text-sm text-blue-700">
                <span>{progressPercent}% complete</span>
                <span>Running MRR@10: {msmarcoProgress.current_mrr?.toFixed(4) || 'N/A'}</span>
              </div>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex gap-3">
            <button
              onClick={evalMode === 'msmarco' ? handleMsmarcoEvaluate : handleEvaluate}
              disabled={loading || (evalMode === 'standard' && !config.data_path)}
              className="flex-1 flex items-center justify-center gap-2 px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed font-medium transition-colors"
            >
              {loading ? (
                <>
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                  Evaluating...
                </>
              ) : (
                <>
                  <Play className="h-5 w-5" />
                  Run {evalMode === 'msmarco' ? 'MS MARCO' : ''} Evaluation
                </>
              )}
            </button>

            {loading && evalMode === 'msmarco' && (
              <button
                onClick={handleCancelEvaluation}
                className="px-4 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 font-medium transition-colors flex items-center gap-2"
              >
                <StopCircle className="h-5 w-5" />
                Cancel
              </button>
            )}
          </div>
        </div>
      </div>

      {/* MS MARCO Results */}
      {evalMode === 'msmarco' && msmarcoResult && (
        <div className="bg-white shadow rounded-lg p-6">
          <div className="flex items-center gap-2 mb-6">
            <CheckCircle className="h-6 w-6 text-green-600" />
            <h2 className="text-xl font-semibold">MS MARCO Evaluation Results</h2>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center p-4 bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg">
              <p className="text-3xl font-bold text-blue-600">
                {msmarcoResult.mrr_at_10?.toFixed(4)}
              </p>
              <p className="text-sm text-gray-600 mt-1 font-medium">MRR@10</p>
              <p className="text-xs text-gray-500 mt-1">Mean Reciprocal Rank</p>
            </div>

            <div className="text-center p-4 bg-gradient-to-br from-green-50 to-green-100 rounded-lg">
              <p className="text-3xl font-bold text-green-600">
                {msmarcoResult.ndcg_at_10?.toFixed(4)}
              </p>
              <p className="text-sm text-gray-600 mt-1 font-medium">NDCG@10</p>
              <p className="text-xs text-gray-500 mt-1">Normalized DCG</p>
            </div>

            <div className="text-center p-4 bg-gradient-to-br from-purple-50 to-purple-100 rounded-lg">
              <p className="text-3xl font-bold text-purple-600">
                {msmarcoResult.recall_at_10?.toFixed(4)}
              </p>
              <p className="text-sm text-gray-600 mt-1 font-medium">Recall@10</p>
              <p className="text-xs text-gray-500 mt-1">Top-10 Recall</p>
            </div>

            <div className="text-center p-4 bg-gradient-to-br from-orange-50 to-orange-100 rounded-lg">
              <p className="text-3xl font-bold text-orange-600">
                {msmarcoResult.recall_at_100?.toFixed(4)}
              </p>
              <p className="text-sm text-gray-600 mt-1 font-medium">Recall@100</p>
              <p className="text-xs text-gray-500 mt-1">Top-100 Recall</p>
            </div>
          </div>

          <div className="mt-6 pt-4 border-t border-gray-200 flex justify-between text-sm text-gray-500">
            <span>Queries evaluated: {msmarcoResult.num_queries}</span>
            <span>Time: {msmarcoResult.elapsed_seconds?.toFixed(1)}s</span>
          </div>
        </div>
      )}

      {/* Standard Evaluation Results */}
      {evalMode === 'standard' && result && (
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
              </div>
            )}

            {result.mrr !== null && result.mrr !== undefined && (
              <div className="text-center p-6 bg-gradient-to-br from-green-50 to-green-100 rounded-lg">
                <p className="text-5xl font-bold text-green-600">
                  {result.mrr?.toFixed(4)}
                </p>
                <p className="text-sm text-gray-600 mt-2 font-medium">MRR</p>
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

      {/* Tips */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <h3 className="text-sm font-medium text-blue-900 mb-2">Tips</h3>
        <ul className="text-sm text-blue-700 space-y-1 list-disc list-inside">
          {evalMode === 'msmarco' ? (
            <>
              <li>MRR@10 is the primary metric for MS MARCO passage ranking</li>
              <li>Use a smaller sample size (100-500) for quick iteration</li>
              <li>Full evaluation (10K queries) takes longer but gives accurate results</li>
              <li>Compare baseline vs fine-tuned model to measure improvement</li>
            </>
          ) : (
            <>
              <li>Ensure your test data is in JSONL format</li>
              <li>Use the same LoRA configuration as training</li>
              <li>Checkpoint path should point to the .safetensors file</li>
              <li>Base model evaluation runs without checkpoint</li>
            </>
          )}
        </ul>
      </div>
    </div>
  );
}

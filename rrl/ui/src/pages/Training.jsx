// RRL Training Page
// src/pages/Training.jsx

import React, { useState, useEffect, useRef } from 'react';
import { Play, Square, Terminal, Settings, Wifi, WifiOff } from 'lucide-react';
import api from '../api';

export default function Training() {
  const [models, setModels] = useState([]);
  const [config, setConfig] = useState({
    model_name: 'bert-base-uncased',
    model_path: '',  // For generator models that use path
    dataset_path: '',
    output_dir: './outputs',
    output_name: '',  // Name for the fine-tuned model
    epochs: 3,
    batch_size: 32,
    learning_rate: 5e-5,
    lora_rank: 8,
    lora_alpha: 16.0,
    device: 'auto',
    max_seq_length: 512,
    gradient_accumulation: 1,
    warmup_ratio: 0.1,
    save_steps: 500,
    logging_steps: 10,
  });
  const [currentJob, setCurrentJob] = useState(null);
  const [logs, setLogs] = useState([]);
  const [isTraining, setIsTraining] = useState(false);
  const [metrics, setMetrics] = useState(null);
  const [wsConnected, setWsConnected] = useState(false);
  const logsEndRef = useRef(null);
  const wsRef = useRef(null);

  useEffect(() => {
    loadModels();
    connectWebSocket();

    // Cleanup on unmount
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  const connectWebSocket = () => {
    wsRef.current = api.connectTrainingWebSocket(
      // onLog callback - receives real-time logs
      (log, logMetrics, jobId, isError) => {
        // Format error logs differently
        const formattedLog = isError ? `❌ ${log}` : log;
        setLogs(prev => [...prev, formattedLog]);
        if (logMetrics) {
          setMetrics(logMetrics);
        }
      },
      // onComplete callback
      (jobId, status, error) => {
        setIsTraining(false);
        if (status === 'completed') {
          setLogs(prev => [...prev, `\n✅ Training completed successfully!`]);
        } else if (error) {
          setLogs(prev => [...prev, `\n❌ Training failed: ${error}`]);
        } else {
          setLogs(prev => [...prev, `\n⚠️ Training ${status}`]);
        }
      },
      // onError callback
      (error) => {
        console.error('WebSocket error:', error);
        setWsConnected(false);
      }
    );

    // Track connection state
    const originalOnOpen = wsRef.current.ws?.onopen;
    const ws = wsRef.current;
    setTimeout(() => setWsConnected(true), 500); // Assume connected after 500ms
  };

  const loadModels = async () => {
    try {
      const [embeddingData, generatorData] = await Promise.all([
        api.listModels(),
        api.listGeneratorModels(),
      ]);
      // Combine both lists with type marker
      const allModels = [
        ...(embeddingData.models || []).map(m => ({...m, type: 'embedding'})),
        ...(generatorData.models || []).map(m => ({...m, type: 'generator', name: m.name})),
      ];
      setModels(allModels);
    } catch (error) {
      console.error('Failed to load models:', error);
    }
  };

  const handleStartTraining = async () => {
    try {
      setIsTraining(true);
      setLogs(['Starting training job...']);
      setMetrics(null);

      const result = await api.startTraining(config);
      setCurrentJob(result.job_id);
      setLogs(prev => [...prev, `Job started: ${result.job_id}`]);

      // WebSocket will handle live log updates automatically
      // Fallback polling in case WebSocket fails
      if (!wsConnected) {
        const interval = setInterval(async () => {
          try {
            const logsData = await api.getTrainingLogs(result.job_id);
            setLogs(logsData.logs);

            const status = await api.getTrainingStatus(result.job_id);
            if (status.status !== 'running') {
              clearInterval(interval);
              setIsTraining(false);
            }
          } catch (error) {
            console.error('Failed to fetch logs:', error);
          }
        }, 1000);
      }

    } catch (error) {
      console.error('Failed to start training:', error);
      setLogs(prev => [...prev, `Error: ${error.message}`]);
      setIsTraining(false);
    }
  };

  const handleStopTraining = async () => {
    if (!currentJob) return;
    
    try {
      await api.stopTraining(currentJob);
      setIsTraining(false);
    } catch (error) {
      console.error('Failed to stop training:', error);
    }
  };

  const selectedModel = models.find(m => m.name === config.model_name);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold text-gray-900">Start Training</h1>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Configuration Form */}
        <div className="bg-white shadow rounded-lg p-6 space-y-6">
          <div className="flex items-center gap-2 mb-4">
            <Settings className="h-5 w-5 text-gray-600" />
            <h2 className="text-xl font-semibold">Training Configuration</h2>
          </div>

          {/* Model Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Model
            </label>
            <select
              value={config.model_name}
              onChange={(e) => {
                const model = models.find(m => m.name === e.target.value);
                setConfig({
                  ...config,
                  model_name: e.target.value,
                  model_path: model?.path || '',
                  lora_rank: model?.recommended_rank || 8,
                });
              }}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            >
              <optgroup label="Embedding Models">
                {models.filter(m => m.type === 'embedding').map(model => (
                  <option key={model.name} value={model.name}>
                    {model.name} ({model.architecture}, {model.size})
                  </option>
                ))}
              </optgroup>
              {models.filter(m => m.type === 'generator').length > 0 && (
                <optgroup label="Generator Models (LLMs)">
                  {models.filter(m => m.type === 'generator').map(model => (
                    <option key={model.name} value={model.name}>
                      {model.name} ({model.architecture}{model.size !== 'unknown' ? `, ${model.size}` : ''})
                      {model.is_finetuned ? ' [Fine-tuned]' : ''}
                    </option>
                  ))}
                </optgroup>
              )}
            </select>
            {selectedModel && (
              <div className="mt-2 text-sm text-gray-500 space-y-1">
                {selectedModel.recommended_rank && (
                  <p>Recommended LoRA rank: {selectedModel.recommended_rank}</p>
                )}
                {selectedModel.path && (
                  <p className="truncate" title={selectedModel.path}>Path: {selectedModel.path}</p>
                )}
                {selectedModel.type === 'generator' && (
                  <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-purple-100 text-purple-800">
                    LLM/Generator
                  </span>
                )}
              </div>
            )}
          </div>

          {/* Output Name */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Output Name
            </label>
            <input
              type="text"
              value={config.output_name}
              onChange={(e) => setConfig({...config, output_name: e.target.value})}
              placeholder="my-finetuned-model"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            />
            <p className="mt-1 text-sm text-gray-500">
              Checkpoint will be saved to: <code className="bg-gray-100 px-1 rounded">{config.output_dir}/{config.output_name || 'model'}/</code>
            </p>
          </div>

          {/* Dataset Path */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Dataset Path
            </label>
            <input
              type="text"
              value={config.dataset_path}
              onChange={(e) => setConfig({...config, dataset_path: e.target.value})}
              placeholder="./data/train.jsonl"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            />
          </div>

          {/* Training Parameters */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Epochs
              </label>
              <input
                type="number"
                value={config.epochs}
                onChange={(e) => setConfig({...config, epochs: parseInt(e.target.value)})}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Batch Size
              </label>
              <input
                type="number"
                value={config.batch_size}
                onChange={(e) => setConfig({...config, batch_size: parseInt(e.target.value)})}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Learning Rate
              </label>
              <input
                type="number"
                step="0.00001"
                value={config.learning_rate}
                onChange={(e) => setConfig({...config, learning_rate: parseFloat(e.target.value)})}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                LoRA Rank
              </label>
              <input
                type="number"
                value={config.lora_rank}
                onChange={(e) => setConfig({...config, lora_rank: parseInt(e.target.value)})}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg"
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
                className="w-full px-3 py-2 border border-gray-300 rounded-lg"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Device
              </label>
              <select
                value={config.device}
                onChange={(e) => setConfig({...config, device: e.target.value})}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg"
              >
                <option value="auto">Auto</option>
                <option value="cuda">CUDA</option>
                <option value="cpu">CPU</option>
                <option value="metal">Metal</option>
              </select>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-3 pt-4">
            <button
              onClick={handleStartTraining}
              disabled={isTraining || !config.dataset_path}
              className="flex-1 flex items-center justify-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
            >
              <Play className="h-5 w-5" />
              {isTraining ? 'Training...' : 'Start Training'}
            </button>
            
            {isTraining && (
              <button
                onClick={handleStopTraining}
                className="flex items-center gap-2 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700"
              >
                <Square className="h-5 w-5" />
                Stop
              </button>
            )}
          </div>
        </div>

        {/* Live Logs */}
        <div className="bg-gray-900 shadow rounded-lg p-6 h-[600px] flex flex-col">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Terminal className="h-5 w-5 text-green-400" />
              <h2 className="text-xl font-semibold text-white">Training Logs</h2>
            </div>
            <div className="flex items-center gap-2">
              {wsConnected ? (
                <span className="flex items-center gap-1 text-green-400 text-sm">
                  <Wifi className="h-4 w-4" />
                  Live
                </span>
              ) : (
                <span className="flex items-center gap-1 text-yellow-400 text-sm">
                  <WifiOff className="h-4 w-4" />
                  Polling
                </span>
              )}
            </div>
          </div>

          {/* Real-time Metrics Panel */}
          {metrics && (
            <div className="grid grid-cols-4 gap-2 mb-4 p-3 bg-gray-800 rounded-lg">
              <div className="text-center">
                <div className="text-xs text-gray-400">Step</div>
                <div className="text-lg font-bold text-blue-400">{metrics.current_step || '-'}</div>
              </div>
              <div className="text-center">
                <div className="text-xs text-gray-400">Epoch</div>
                <div className="text-lg font-bold text-purple-400">{metrics.current_epoch || '-'}</div>
              </div>
              <div className="text-center">
                <div className="text-xs text-gray-400">Loss</div>
                <div className="text-lg font-bold text-yellow-400">
                  {metrics.current_loss?.toFixed(4) || '-'}
                </div>
              </div>
              <div className="text-center">
                <div className="text-xs text-gray-400">LR</div>
                <div className="text-lg font-bold text-green-400">
                  {metrics.learning_rate?.toExponential(2) || '-'}
                </div>
              </div>
            </div>
          )}

          <div className="flex-1 overflow-y-auto font-mono text-sm bg-black rounded p-4">
            {logs.length === 0 ? (
              <div className="text-gray-500">
                Waiting for training to start...
              </div>
            ) : (
              logs.map((log, i) => {
                // Check if this is an error line
                const isError = log.startsWith('❌') || log.toLowerCase().includes('error') ||
                                log.toLowerCase().includes('failed') || log.toLowerCase().includes('oom');
                return (
                  <div
                    key={i}
                    className={`mb-1 whitespace-pre-wrap ${isError ? 'text-red-400 font-bold' : 'text-green-400'}`}
                  >
                    {log}
                  </div>
                );
              })
            )}
            <div ref={logsEndRef} />
          </div>
        </div>
      </div>
    </div>
  );
}
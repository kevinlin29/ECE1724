// API Client for RRL Backend
// src/api.js

import axios from 'axios';

const API_URL = 'http://localhost:8000';

const api = {
  // Models
  listModels: async () => {
    const response = await axios.get(`${API_URL}/models`);
    return response.data;
  },

  getModelInfo: async (modelName) => {
    const response = await axios.get(`${API_URL}/models/${modelName}`);
    return response.data;
  },

  // Training
  startTraining: async (config) => {
    const response = await axios.post(`${API_URL}/train`, config);
    return response.data;
  },

  getTrainingStatus: async (jobId) => {
    const response = await axios.get(`${API_URL}/train/${jobId}`);
    return response.data;
  },

  getTrainingLogs: async (jobId, lines = 100) => {
    const response = await axios.get(`${API_URL}/train/${jobId}/logs`, {
      params: { lines }
    });
    return response.data;
  },

  stopTraining: async (jobId) => {
    const response = await axios.post(`${API_URL}/train/${jobId}/stop`);
    return response.data;
  },

  listTrainings: async () => {
    const response = await axios.get(`${API_URL}/trainings`);
    return response.data;
  },

  // Evaluation
  runEvaluation: async (config) => {
    const response = await axios.post(`${API_URL}/eval`, config);
    return response.data;
  },

  // Inference
  runInference: async (request) => {
    const response = await axios.post(`${API_URL}/inference`, request);
    return response.data;
  },

  // Checkpoints
  listCheckpoints: async (outputDir = './outputs') => {
    const response = await axios.get(`${API_URL}/checkpoints`, {
      params: { output_dir: outputDir }
    });
    return response.data;
  },

  // Upload
  uploadDataset: async (file, onProgress) => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await axios.post(`${API_URL}/upload/dataset`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        const percentCompleted = Math.round(
          (progressEvent.loaded * 100) / progressEvent.total
        );
        if (onProgress) onProgress(percentCompleted);
      },
    });

    return response.data;
  },

  // RAG Endpoints
  runIngest: async (config) => {
    const response = await axios.post(`${API_URL}/rag/ingest`, config);
    return response.data;
  },

  generateEmbeddings: async (config) => {
    const response = await axios.post(`${API_URL}/rag/embed`, config);
    return response.data;
  },

  buildIndex: async (config) => {
    const response = await axios.post(`${API_URL}/rag/index`, config);
    return response.data;
  },

  queryRAG: async (config) => {
    const response = await axios.post(`${API_URL}/rag/query`, config);
    return response.data;
  },

  // Full RAG with LLM generation
  ragGenerate: async (config) => {
    const response = await axios.post(`${API_URL}/rag/generate`, config, {
      timeout: 300000,  // 5 min timeout for LLM generation
    });
    return response.data;
  },

  // Full RAG pipeline: ingest → embed → index
  ragPipeline: async (config) => {
    const response = await axios.post(`${API_URL}/rag/pipeline`, config, {
      timeout: 600000,  // 10 min timeout for full pipeline
    });
    return response.data;
  },

  // Generator Model Management
  listGeneratorModels: async () => {
    const response = await axios.get(`${API_URL}/generator-models`);
    return response.data;
  },

  addGeneratorModel: async (name, path, architecture = 'auto') => {
    const response = await axios.post(`${API_URL}/generator-models`, {
      name,
      path,
      architecture,
    });
    return response.data;
  },

  removeGeneratorModel: async (modelName) => {
    const response = await axios.delete(`${API_URL}/generator-models/${encodeURIComponent(modelName)}`);
    return response.data;
  },

  getGeneratorModel: async (modelName) => {
    const response = await axios.get(`${API_URL}/generator-models/${encodeURIComponent(modelName)}`);
    return response.data;
  },

  scanFinetunedModels: async (outputDir = './output') => {
    const response = await axios.post(`${API_URL}/generator-models/scan`, null, {
      params: { output_dir: outputDir }
    });
    return response.data;
  },

  // Health
  health: async () => {
    const response = await axios.get(`${API_URL}/health`);
    return response.data;
  },

  // WebSocket for live training updates
  connectTrainingWebSocket: (onLog, onComplete, onError) => {
    const WS_URL = API_URL.replace('http', 'ws');
    const ws = new WebSocket(`${WS_URL}/ws`);

    ws.onopen = () => {
      console.log('WebSocket connected for training updates');
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'training_log') {
          onLog(data.log, data.metrics, data.job_id, data.is_error);
        } else if (data.type === 'training_complete') {
          onComplete(data.job_id, data.status, data.error);
        }
      } catch (e) {
        console.error('Failed to parse WebSocket message:', e);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      if (onError) onError(error);
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
    };

    // Return control object
    return {
      close: () => ws.close(),
      send: (msg) => ws.send(JSON.stringify(msg)),
    };
  },
};

export default api;
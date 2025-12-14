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

  // Health
  health: async () => {
    const response = await axios.get(`${API_URL}/health`);
    return response.data;
  },
};

export default api;
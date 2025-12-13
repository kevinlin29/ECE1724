import axios from 'axios';

const API_URL = 'http://localhost:8000';

const api = {
  listModels: async () => {
    const response = await axios.get(`${API_URL}/models`);
    return response.data;
  },
  
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
  
  runEvaluation: async (config) => {
    const response = await axios.post(`${API_URL}/eval`, config);
    return response.data;
  },
  
  runInference: async (request) => {
    const response = await axios.post(`${API_URL}/inference`, request);
    return response.data;
  },
  
  uploadDataset: async (file, onProgress) => {
    const formData = new FormData();
    formData.append('file', file);
    const response = await axios.post(`${API_URL}/upload/dataset`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: (progressEvent) => {
        const percentCompleted = Math.round(
          (progressEvent.loaded * 100) / progressEvent.total
        );
        if (onProgress) onProgress(percentCompleted);
      },
    });
    return response.data;
  },
};

export default api;

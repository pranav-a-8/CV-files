// frontend/services/api.js
import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export const generateProject = async (description) => {
  try {
    const response = await axios.post(`${API_URL}/api/generate`, { description });
    return response.data; // { id, file_paths, description, created_at }
  } catch (error) {
    throw new Error(error.response?.data?.detail || 'Generation failed');
  }
};

export const checkStatus = async (projectId) => {
  try {
    const response = await axios.get(`${API_URL}/api/preview/${projectId}`);
    return response.data; // { preview_url }
  } catch (error) {
    throw new Error(error.response?.data?.detail || 'Status check failed');
  }
};

export const downloadZip = (projectId) => {
  window.location.href = `${API_URL}/api/download/${projectId}`; // Correct endpoint
};

export const stopPreview = async (projectId) => {
  try {
    const response = await axios.post(`${API_URL}/api/stop-preview/${projectId}`);
    return response.data; // { message }
  } catch (error) {
    throw new Error(error.response?.data?.detail || 'Failed to stop preview');
  }
};
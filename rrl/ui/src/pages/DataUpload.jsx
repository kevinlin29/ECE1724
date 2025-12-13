// src/pages/DataUpload.jsx
import React, { useState } from 'react';
import { Upload, CheckCircle, File, X, AlertCircle } from 'lucide-react';
import api from '../api';

export default function DataUpload() {
  const [progress, setProgress] = useState(0);
  const [uploadResult, setUploadResult] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelect = (file) => {
    if (!file) return;

    // Validate file type
    const validTypes = ['.jsonl', '.json', '.txt'];
    const fileName = file.name.toLowerCase();
    const isValid = validTypes.some(type => fileName.endsWith(type));
    
    if (!isValid) {
      setError('Invalid file type. Please upload .jsonl, .json, or .txt files');
      return;
    }

    // Validate file size (max 100MB)
    if (file.size > 100 * 1024 * 1024) {
      setError('File too large. Maximum size is 100MB');
      return;
    }

    setSelectedFile(file);
    setError(null);
    setUploadResult(null);
  };

  const handleFileInputChange = (event) => {
    const file = event.target.files[0];
    handleFileSelect(file);
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setUploading(true);
    setProgress(0);
    setError(null);
    setUploadResult(null);

    try {
      const result = await api.uploadDataset(selectedFile, setProgress);
      setUploadResult(result);
      setSelectedFile(null);
    } catch (err) {
      console.error('Upload failed:', err);
      setError(err.message || 'Upload failed');
    } finally {
      setUploading(false);
      setProgress(0);
    }
  };

  const clearSelection = () => {
    setSelectedFile(null);
    setError(null);
    setUploadResult(null);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <Upload className="h-8 w-8 text-blue-600" />
        <h1 className="text-3xl font-bold text-gray-900">Upload Dataset</h1>
      </div>

      {/* Upload Area */}
      <div className="bg-white shadow rounded-lg p-6">
        <div
          className={`relative border-2 border-dashed rounded-lg p-12 text-center transition-colors ${
            dragActive 
              ? 'border-blue-500 bg-blue-50' 
              : selectedFile 
                ? 'border-green-500 bg-green-50'
                : 'border-gray-300 hover:border-gray-400'
          }`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          {selectedFile ? (
            <div className="space-y-4">
              <File className="mx-auto h-16 w-16 text-green-600" />
              <div>
                <p className="text-lg font-medium text-gray-900">{selectedFile.name}</p>
                <p className="text-sm text-gray-500 mt-1">
                  {(selectedFile.size / 1024).toFixed(2)} KB
                </p>
              </div>
              <div className="flex justify-center gap-3">
                <button
                  onClick={handleUpload}
                  disabled={uploading}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 font-medium"
                >
                  {uploading ? 'Uploading...' : 'Upload File'}
                </button>
                <button
                  onClick={clearSelection}
                  disabled={uploading}
                  className="px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 font-medium"
                >
                  Cancel
                </button>
              </div>
            </div>
          ) : (
            <>
              <Upload className={`mx-auto h-12 w-12 ${dragActive ? 'text-blue-600' : 'text-gray-400'}`} />
              <div className="mt-4">
                <label
                  htmlFor="file-upload"
                  className="cursor-pointer inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 transition-colors"
                >
                  <span>Choose File</span>
                  <input
                    id="file-upload"
                    type="file"
                    onChange={handleFileInputChange}
                    accept=".jsonl,.json,.txt"
                    className="sr-only"
                  />
                </label>
                <p className="mt-2 text-sm text-gray-500">
                  or drag and drop
                </p>
              </div>
              <p className="mt-2 text-xs text-gray-500">
                JSONL, JSON, or TXT files (max 100MB)
              </p>
            </>
          )}
        </div>
        
        {/* Progress Bar */}
        {uploading && (
          <div className="mt-4">
            <div className="flex justify-between text-sm text-gray-600 mb-2">
              <span>Uploading...</span>
              <span>{progress}%</span>
            </div>
            <div className="bg-gray-200 rounded-full h-2 overflow-hidden">
              <div
                className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
        )}

        {/* Error Message */}
        {error && (
          <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg flex items-start gap-3">
            <AlertCircle className="h-5 w-5 text-red-600 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-red-800 font-medium">Upload Failed</p>
              <p className="text-red-700 text-sm mt-1">{error}</p>
            </div>
          </div>
        )}

        {/* Success Message */}
        {uploadResult && (
          <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <CheckCircle className="h-5 w-5 text-green-600" />
              <p className="text-green-800 font-medium">Upload Successful!</p>
            </div>
            <div className="text-sm text-gray-700 space-y-1 ml-7">
              <p><span className="font-medium">Filename:</span> {uploadResult.filename}</p>
              <p><span className="font-medium">Path:</span> <code className="bg-white px-2 py-1 rounded">{uploadResult.path}</code></p>
              <p><span className="font-medium">Size:</span> {(uploadResult.size / 1024).toFixed(2)} KB</p>
            </div>
            <div className="mt-3 ml-7">
              <p className="text-xs text-green-700">
                ✓ You can now use this dataset in the Training page
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Dataset Format Guide */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-blue-900 mb-3">📝 Dataset Format</h3>
        <p className="text-sm text-blue-800 mb-4">
          Upload datasets in JSONL format with the following structure:
        </p>
        
        <div className="space-y-4">
          <div>
            <p className="text-sm font-medium text-blue-900 mb-2">Training Data (JSONL):</p>
            <pre className="text-xs bg-white p-3 rounded border border-blue-100 overflow-auto">
{`{"query": "What is machine learning?", "positive": "Machine learning is a subset of AI..."}
{"query": "Explain neural networks", "positive": "Neural networks are computing systems..."}`}
            </pre>
          </div>

          <div>
            <p className="text-sm font-medium text-blue-900 mb-2">Evaluation Data (JSON):</p>
            <pre className="text-xs bg-white p-3 rounded border border-blue-100 overflow-auto">
{`[
  {
    "query": "What is AI?",
    "candidates": [
      "Artificial Intelligence is...",
      "AI stands for...",
      "Machine learning is..."
    ],
    "correct": 0
  }
]`}
            </pre>
          </div>
        </div>

        <div className="mt-4 bg-white rounded-lg p-4 border border-blue-100">
          <h4 className="text-sm font-medium text-blue-900 mb-2">📋 Requirements:</h4>
          <ul className="text-xs text-blue-800 space-y-1 list-disc list-inside">
            <li>Each line must be valid JSON (for JSONL files)</li>
            <li>Required fields: <code className="bg-blue-100 px-1 rounded">query</code> and <code className="bg-blue-100 px-1 rounded">positive</code></li>
            <li>Optional fields: <code className="bg-blue-100 px-1 rounded">negative</code>, <code className="bg-blue-100 px-1 rounded">metadata</code></li>
            <li>UTF-8 encoding recommended</li>
            <li>Maximum file size: 100MB</li>
          </ul>
        </div>
      </div>

      {/* Recent Uploads */}
      <div className="bg-white shadow rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4">Recent Uploads</h3>
        <div className="text-center py-8 text-gray-500 text-sm">
          <File className="h-12 w-12 text-gray-300 mx-auto mb-3" />
          <p>Upload history will appear here</p>
        </div>
      </div>
    </div>
  );
}
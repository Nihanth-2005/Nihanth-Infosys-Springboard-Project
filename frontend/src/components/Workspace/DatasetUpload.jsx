import React, { useState, useEffect } from 'react';
import axios from 'axios';
import toast from 'react-hot-toast';
import { motion } from 'framer-motion';
import { Upload, FileText, Target, Loader } from 'lucide-react';
import { API_ENDPOINTS } from '../../config/api';

function DatasetUpload({ workspaceId }) {
  const [file, setFile] = useState(null);
  const [targetColumn, setTargetColumn] = useState('');
  const [columns, setColumns] = useState([]);
  const [loading, setLoading] = useState(false);
  const [training, setTraining] = useState(false);
  const [results, setResults] = useState(null);
  const [progress, setProgress] = useState(0);
  const [progressMessage, setProgressMessage] = useState('');

  const handleFileChange = async (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) return;

    const isCsv = selectedFile.name.toLowerCase().endsWith('.csv');
    const isJson = selectedFile.name.toLowerCase().endsWith('.json');
    if (!isCsv && !isJson) {
      toast.error('Please upload a CSV or JSON file');
      return;
    }

    setFile(selectedFile);

    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const text = event.target.result;
        if (isCsv) {
          const lines = text.split('\n');
          if (lines.length > 0) {
            const cols = lines[0].split(',').map((col) => col.trim());
            setColumns(cols);
            if (cols.length > 0) setTargetColumn(cols[cols.length - 1]);
          }
        } else if (isJson) {
          let data = null;
          try {
            data = JSON.parse(text);
          } catch (e) {}
          if (Array.isArray(data) && data.length > 0 && typeof data[0] === 'object') {
            const cols = Object.keys(data[0]);
            setColumns(cols);
            if (cols.length > 0) setTargetColumn(cols[cols.length - 1]);
          } else {
            setColumns([]);
          }
        }
      } catch (_) {
        setColumns([]);
      }
    };
    reader.readAsText(selectedFile);
  };

  const handleUpload = async () => {
    if (!file || !targetColumn) {
      toast.error('Please select a file and target column');
      return;
    }

    setTraining(true);
    setLoading(true);

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('workspace_id', workspaceId);
      formData.append('target_column', targetColumn);

      const response = await axios.post(API_ENDPOINTS.DATASET_UPLOAD, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setProgress(100);
      setProgressMessage('Training complete');

      const normalized = response.data.metrics ? response.data.metrics : response.data;
      setResults(normalized);
      toast.success('Model trained successfully!');
    } catch (error) {
      console.error('Error training model:', error);
      toast.error(error.response?.data?.error || 'Failed to train model');
    } finally {
      setTraining(false);
      setLoading(false);
    }
  };

  useEffect(() => {
    let interval = null;
    const fetchProgress = async () => {
      try {
        const res = await axios.get(`${API_ENDPOINTS.PROGRESS}/${workspaceId}`);
        const data = res.data;
        setProgress(data.progress ?? 0);
        setProgressMessage(data.message ?? '');
        if (data.status === 'done' || data.status === 'error') {
          clearInterval(interval);
        }
      } catch (e) {}
    };

    if (training) {
      fetchProgress();
      interval = setInterval(fetchProgress, 2000);
    }

    return () => clearInterval(interval);
  }, [training, workspaceId]);

  return (
    <div
      className="min-h-screen w-full bg-cover bg-center bg-no-repeat bg-fixed p-8"
      style={{
        backgroundImage:
          "url('https://img.freepik.com/premium-photo/artificial-intelligence-concept-abstract-futuristic-electronic-circuit-technology-background-with-logo-ai_154730-4513.jpg')",
        backgroundSize: 'cover',
        backgroundAttachment: 'fixed',
        backgroundPosition: 'center',
        backgroundColor: 'rgba(0, 0, 0, 0.5)',
        backgroundBlendMode: 'overlay',
      }}
    >
      <div className="max-w-6xl mx-auto space-y-6">
        <div className="glass-card rounded-2xl p-8 bg-slate-900/40 backdrop-blur-xl shadow-xl border border-slate-700">
          <h2 className="text-2xl font-bold mb-6 text-white flex items-center gap-3">
            <Upload className="w-6 h-6 text-blue-400" />
            Upload Dataset & Training
          </h2>

          {/* File Upload */}
          <div className="mb-6">
            <label className="block text-sm font-medium mb-2 text-gray-200">
              Select CSV or JSON File
            </label>
            <div className="relative">
              <input
                type="file"
                accept=".csv,.json"
                onChange={handleFileChange}
                className="hidden"
                id="file-upload"
                disabled={loading}
              />
              <label
                htmlFor="file-upload"
                className="flex items-center justify-center gap-3 p-8 border-2 border-dashed border-slate-500 rounded-xl cursor-pointer hover:border-blue-400 transition-colors bg-slate-800/40"
              >
                <FileText className="w-8 h-8 text-blue-400" />
                <div>
                  <span className="text-blue-400 font-semibold">Click to upload</span>
                  <span className="text-gray-300"> or drag and drop</span>
                </div>
              </label>
              {file && (
                <div className="mt-4 p-4 bg-slate-800/60 rounded-lg text-gray-200">
                  <p className="text-sm">
                    Selected: <span className="font-semibold text-blue-300">{file.name}</span>
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Target Column Selection */}
          {columns.length > 0 && (
            <div className="mb-6">
              <label className="block text-sm font-medium mb-2 text-gray-200 flex items-center gap-2">
                <Target className="w-4 h-4 text-blue-400" />
                Target Column
              </label>
              <select
                value={targetColumn}
                onChange={(e) => setTargetColumn(e.target.value)}
                className="w-full px-4 py-3 rounded-lg border border-slate-600 bg-slate-800 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                disabled={loading}
              >
                {columns.map((col) => (
                  <option key={col} value={col}>
                    {col}
                  </option>
                ))}
              </select>
            </div>
          )}

          {/* Upload Button */}
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={handleUpload}
            disabled={!file || !targetColumn || loading}
            className="w-full flex items-center justify-center gap-3 px-6 py-4 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-xl font-semibold hover:shadow-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {training ? (
              <>
                <Loader className="w-5 h-5 animate-spin" />
                <span>Training Model...</span>
              </>
            ) : (
              <>
                <Upload className="w-5 h-5" />
                <span>Train Model</span>
              </>
            )}
          </motion.button>

          {/* Progress display */}
          {training && (
            <div className="mt-4">
              <p className="text-sm text-gray-300 mb-2">{progressMessage}</p>
              <div className="w-full bg-slate-700 h-2 rounded-full overflow-hidden">
                <div
                  style={{ width: `${progress}%` }}
                  className="h-2 bg-blue-500 transition-all"
                />
              </div>
              <p className="text-xs text-gray-400 mt-1">{progress}%</p>
            </div>
          )}
        </div>

        {/* Results */}
        {results && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="glass-card rounded-2xl p-8 bg-slate-900/50 backdrop-blur-xl border border-slate-700"
          >
            <h3 className="text-xl font-bold mb-4 text-white">Training Results</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {results?.accuracy != null && (
                <div className="p-4 bg-blue-900/30 rounded-lg">
                  <p className="text-sm text-gray-300 mb-1">Accuracy</p>
                  <p className="text-2xl font-bold text-blue-400">
                    {((results.accuracy * 100) || 0).toFixed(2)}%
                  </p>
                </div>
              )}
              {results?.f1_score != null && (
                <div className="p-4 bg-purple-900/30 rounded-lg">
                  <p className="text-sm text-gray-300 mb-1">F1 Score</p>
                  <p className="text-2xl font-bold text-purple-400">
                    {Number.isFinite(results.f1_score)
                      ? results.f1_score.toFixed(3)
                      : String(results.f1_score)}
                  </p>
                </div>
              )}
              {results?.mse != null && (
                <div className="p-4 bg-green-900/30 rounded-lg">
                  <p className="text-sm text-gray-300 mb-1">MSE</p>
                  <p className="text-2xl font-bold text-green-400">
                    {Number.isFinite(results.mse)
                      ? results.mse.toFixed(3)
                      : String(results.mse)}
                  </p>
                </div>
              )}
              {results?.r2_score != null && (
                <div className="p-4 bg-yellow-900/30 rounded-lg">
                  <p className="text-sm text-gray-300 mb-1">R² Score</p>
                  <p className="text-2xl font-bold text-yellow-400">
                    {Number.isFinite(results.r2_score)
                      ? results.r2_score.toFixed(3)
                      : String(results.r2_score)}
                  </p>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </div>
    </div>
  );
}

export default DatasetUpload;

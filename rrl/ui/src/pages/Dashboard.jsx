// RRL Dashboard - Training Monitor
// src/pages/Dashboard.jsx

import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Activity, TrendingDown, Zap, Clock, CheckCircle, XCircle } from 'lucide-react';
import api from '../api';

export default function Dashboard() {
  const [trainings, setTrainings] = useState([]);
  const [selectedJob, setSelectedJob] = useState(null);
  const [liveMetrics, setLiveMetrics] = useState([]);
  const [ws, setWs] = useState(null);

  useEffect(() => {
    loadTrainings();
    connectWebSocket();
    
    const interval = setInterval(loadTrainings, 5000);
    return () => {
      clearInterval(interval);
      if (ws) ws.close();
    };
  }, []);

  const loadTrainings = async () => {
    try {
      const data = await api.listTrainings();
      setTrainings(data.trainings);
    } catch (error) {
      console.error('Failed to load trainings:', error);
    }
  };

  const connectWebSocket = () => {
    const websocket = new WebSocket('ws://localhost:8000/ws');
    
    websocket.onmessage = (event) => {
      const message = JSON.parse(event.data);
      
      if (message.type === 'training_log' && message.metrics) {
        setLiveMetrics(prev => [...prev, {
          step: message.metrics.current_step,
          loss: message.metrics.current_loss,
          epoch: message.metrics.current_epoch,
          lr: message.metrics.learning_rate,
          timestamp: Date.now()
        }].slice(-100)); // Keep last 100 points
      }
    };
    
    setWs(websocket);
  };

  const activeTrainings = trainings.filter(t => t.status === 'running');
  const completedTrainings = trainings.filter(t => t.status === 'completed');

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold text-gray-900">Training Dashboard</h1>
        <button
          onClick={loadTrainings}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
        >
          Refresh
        </button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-4">
        <StatCard
          icon={Activity}
          label="Active Trainings"
          value={activeTrainings.length}
          color="blue"
        />
        <StatCard
          icon={CheckCircle}
          label="Completed"
          value={completedTrainings.length}
          color="green"
        />
        <StatCard
          icon={TrendingDown}
          label="Current Loss"
          value={liveMetrics[liveMetrics.length - 1]?.loss?.toFixed(4) || 'N/A'}
          color="purple"
        />
        <StatCard
          icon={Zap}
          label="Total Jobs"
          value={trainings.length}
          color="yellow"
        />
      </div>

      {/* Live Training Chart */}
      {liveMetrics.length > 0 && (
        <div className="bg-white shadow rounded-lg p-6">
          <h2 className="text-xl font-semibold mb-4">Live Training Metrics</h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={liveMetrics}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="step" label={{ value: 'Step', position: 'insideBottom', offset: -5 }} />
              <YAxis label={{ value: 'Loss', angle: -90, position: 'insideLeft' }} />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="loss" stroke="#8884d8" name="Training Loss" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Active Trainings */}
      <div className="bg-white shadow rounded-lg overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-200">
          <h2 className="text-xl font-semibold">Active Trainings</h2>
        </div>
        <div className="divide-y divide-gray-200">
          {activeTrainings.length === 0 ? (
            <div className="px-6 py-8 text-center text-gray-500">
              No active trainings
            </div>
          ) : (
            activeTrainings.map(job => (
              <TrainingRow key={job.id} job={job} onSelect={setSelectedJob} />
            ))
          )}
        </div>
      </div>

      {/* Recent Trainings */}
      <div className="bg-white shadow rounded-lg overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-200">
          <h2 className="text-xl font-semibold">Recent Trainings</h2>
        </div>
        <div className="divide-y divide-gray-200">
          {trainings.slice(0, 5).map(job => (
            <TrainingRow key={job.id} job={job} onSelect={setSelectedJob} />
          ))}
        </div>
      </div>
    </div>
  );
}

function StatCard({ icon: Icon, label, value, color }) {
  const colors = {
    blue: 'bg-blue-50 text-blue-700',
    green: 'bg-green-50 text-green-700',
    purple: 'bg-purple-50 text-purple-700',
    yellow: 'bg-yellow-50 text-yellow-700',
  };

  return (
    <div className="bg-white overflow-hidden shadow rounded-lg">
      <div className="p-5">
        <div className="flex items-center">
          <div className={`flex-shrink-0 rounded-md p-3 ${colors[color]}`}>
            <Icon className="h-6 w-6" />
          </div>
          <div className="ml-5 w-0 flex-1">
            <dl>
              <dt className="text-sm font-medium text-gray-500 truncate">{label}</dt>
              <dd className="text-2xl font-semibold text-gray-900">{value}</dd>
            </dl>
          </div>
        </div>
      </div>
    </div>
  );
}

function TrainingRow({ job, onSelect }) {
  const statusColors = {
    running: 'bg-blue-100 text-blue-800',
    completed: 'bg-green-100 text-green-800',
    failed: 'bg-red-100 text-red-800',
    stopped: 'bg-gray-100 text-gray-800',
  };

  const StatusIcon = job.status === 'completed' ? CheckCircle : 
                     job.status === 'failed' ? XCircle : 
                     Activity;

  return (
    <div 
      className="px-6 py-4 hover:bg-gray-50 cursor-pointer"
      onClick={() => onSelect(job)}
    >
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <div className="flex items-center gap-3">
            <StatusIcon className={`h-5 w-5 ${job.status === 'running' ? 'text-blue-600 animate-pulse' : ''}`} />
            <div>
              <p className="text-sm font-medium text-gray-900">{job.config.model_name}</p>
              <p className="text-sm text-gray-500">Job ID: {job.id.slice(0, 8)}</p>
            </div>
          </div>
        </div>
        
        <div className="flex items-center gap-4">
          {job.current_loss && (
            <div className="text-right">
              <p className="text-sm font-medium text-gray-900">Loss: {job.current_loss.toFixed(4)}</p>
              <p className="text-xs text-gray-500">Step {job.current_step}</p>
            </div>
          )}
          
          <span className={`px-3 py-1 rounded-full text-xs font-medium ${statusColors[job.status]}`}>
            {job.status}
          </span>
        </div>
      </div>
    </div>
  );
}
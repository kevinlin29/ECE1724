// src/pages/RAG.jsx
import React, { useState } from 'react';
import { FileText, Database, Search, Layers, Upload, Play, CheckCircle, Loader2 } from 'lucide-react';
import api from '../api';

export default function RAG() {
  const [activeTab, setActiveTab] = useState('ingest'); // ingest, embed, index, query
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <Database className="h-8 w-8 text-blue-600" />
        <h1 className="text-3xl font-bold text-gray-900">RAG System</h1>
      </div>

      {/* Tab Navigation */}
      <div className="bg-white shadow rounded-lg">
        <div className="border-b border-gray-200">
          <nav className="flex -mb-px">
            <TabButton
              active={activeTab === 'ingest'}
              onClick={() => setActiveTab('ingest')}
              icon={FileText}
            >
              1. Ingest Documents
            </TabButton>
            <TabButton
              active={activeTab === 'embed'}
              onClick={() => setActiveTab('embed')}
              icon={Layers}
            >
              2. Generate Embeddings
            </TabButton>
            <TabButton
              active={activeTab === 'index'}
              onClick={() => setActiveTab('index')}
              icon={Database}
            >
              3. Build Index
            </TabButton>
            <TabButton
              active={activeTab === 'query'}
              onClick={() => setActiveTab('query')}
              icon={Search}
            >
              4. Query
            </TabButton>
          </nav>
        </div>

        <div className="p-6">
          {activeTab === 'ingest' && (
            <IngestPanel loading={loading} setLoading={setLoading} result={result} setResult={setResult} error={error} setError={setError} />
          )}
          {activeTab === 'embed' && (
            <EmbedPanel loading={loading} setLoading={setLoading} result={result} setResult={setResult} error={error} setError={setError} />
          )}
          {activeTab === 'index' && (
            <IndexPanel loading={loading} setLoading={setLoading} result={result} setResult={setResult} error={error} setError={setError} />
          )}
          {activeTab === 'query' && (
            <QueryPanel loading={loading} setLoading={setLoading} result={result} setResult={setResult} error={error} setError={setError} />
          )}
        </div>
      </div>

      {/* Workflow Guide */}
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-blue-900 mb-3">📚 RAG Workflow</h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <WorkflowStep number="1" title="Ingest" description="Load and chunk documents" />
          <WorkflowStep number="2" title="Embed" description="Generate embeddings" />
          <WorkflowStep number="3" title="Index" description="Build vector index" />
          <WorkflowStep number="4" title="Query" description="Search and retrieve" />
        </div>
      </div>
    </div>
  );
}

function TabButton({ active, onClick, icon: Icon, children }) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-2 px-6 py-4 text-sm font-medium border-b-2 transition-colors ${
        active
          ? 'border-blue-600 text-blue-600'
          : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
      }`}
    >
      <Icon className="h-5 w-5" />
      {children}
    </button>
  );
}

function WorkflowStep({ number, title, description }) {
  return (
    <div className="text-center">
      <div className="inline-flex items-center justify-center w-10 h-10 rounded-full bg-blue-600 text-white font-bold mb-2">
        {number}
      </div>
      <p className="font-medium text-blue-900">{title}</p>
      <p className="text-sm text-blue-700">{description}</p>
    </div>
  );
}

// Ingest Panel
function IngestPanel({ loading, setLoading, result, setResult, error, setError }) {
  const [config, setConfig] = useState({
    input: './docs',
    output: './data/chunks.json',
    chunk_size: 512,
    chunk_overlap: 50,
  });

  const handleIngest = async () => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await api.runIngest(config);
      setResult(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Input Directory
          </label>
          <input
            type="text"
            value={config.input}
            onChange={(e) => setConfig({...config, input: e.target.value})}
            placeholder="./docs"
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
          />
          <p className="mt-1 text-sm text-gray-500">
            Path to directory containing documents (.txt, .md, .pdf)
          </p>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Output Path
          </label>
          <input
            type="text"
            value={config.output}
            onChange={(e) => setConfig({...config, output: e.target.value})}
            placeholder="./data/chunks.json"
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
          />
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Chunk Size
            </label>
            <input
              type="number"
              value={config.chunk_size}
              onChange={(e) => setConfig({...config, chunk_size: parseInt(e.target.value)})}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Chunk Overlap
            </label>
            <input
              type="number"
              value={config.chunk_overlap}
              onChange={(e) => setConfig({...config, chunk_overlap: parseInt(e.target.value)})}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg"
            />
          </div>
        </div>
      </div>

      {error && <ErrorMessage message={error} />}
      {result && <SuccessMessage message={`Ingested ${result.chunks_count || 0} chunks`} />}

      <button
        onClick={handleIngest}
        disabled={loading}
        className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 font-medium"
      >
        {loading ? <Loader2 className="h-5 w-5 animate-spin" /> : <Play className="h-5 w-5" />}
        {loading ? 'Ingesting...' : 'Ingest Documents'}
      </button>
    </div>
  );
}

// Embed Panel
function EmbedPanel({ loading, setLoading, result, setResult, error, setError }) {
  const [config, setConfig] = useState({
    input: './data/chunks.json',
    output: './data/embeddings.safetensors',
    model: 'BAAI/bge-base-en-v1.5',
    batch_size: 32,
  });

  const handleEmbed = async () => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await api.generateEmbeddings(config);
      setResult(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Input Chunks
          </label>
          <input
            type="text"
            value={config.input}
            onChange={(e) => setConfig({...config, input: e.target.value})}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Output Embeddings
          </label>
          <input
            type="text"
            value={config.output}
            onChange={(e) => setConfig({...config, output: e.target.value})}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Embedding Model
          </label>
          <select
            value={config.model}
            onChange={(e) => setConfig({...config, model: e.target.value})}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg"
          >
            <option value="BAAI/bge-base-en-v1.5">BGE Base (768d)</option>
            <option value="BAAI/bge-large-en-v1.5">BGE Large (1024d)</option>
            <option value="bert-base-uncased">BERT Base (768d)</option>
            <option value="roberta-base">RoBERTa Base (768d)</option>
          </select>
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
      </div>

      {error && <ErrorMessage message={error} />}
      {result && <SuccessMessage message={`Generated ${result.embeddings_count || 0} embeddings`} />}

      <button
        onClick={handleEmbed}
        disabled={loading}
        className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 font-medium"
      >
        {loading ? <Loader2 className="h-5 w-5 animate-spin" /> : <Layers className="h-5 w-5" />}
        {loading ? 'Generating...' : 'Generate Embeddings'}
      </button>
    </div>
  );
}

// Index Panel
function IndexPanel({ loading, setLoading, result, setResult, error, setError }) {
  const [config, setConfig] = useState({
    embeddings: './data/embeddings.safetensors',
    output: './index',
    index_type: 'hnsw',
  });

  const handleBuildIndex = async () => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await api.buildIndex(config);
      setResult(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Embeddings Path
          </label>
          <input
            type="text"
            value={config.embeddings}
            onChange={(e) => setConfig({...config, embeddings: e.target.value})}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Output Directory
          </label>
          <input
            type="text"
            value={config.output}
            onChange={(e) => setConfig({...config, output: e.target.value})}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Index Type
          </label>
          <select
            value={config.index_type}
            onChange={(e) => setConfig({...config, index_type: e.target.value})}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg"
          >
            <option value="hnsw">HNSW (Fast, Approximate)</option>
            <option value="flat">Flat (Exact, Slower)</option>
          </select>
        </div>
      </div>

      {error && <ErrorMessage message={error} />}
      {result && <SuccessMessage message="Index built successfully" />}

      <button
        onClick={handleBuildIndex}
        disabled={loading}
        className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 font-medium"
      >
        {loading ? <Loader2 className="h-5 w-5 animate-spin" /> : <Database className="h-5 w-5" />}
        {loading ? 'Building...' : 'Build Index'}
      </button>
    </div>
  );
}

// Query Panel
function QueryPanel({ loading, setLoading, result, setResult, error, setError }) {
  const [config, setConfig] = useState({
    index: './index',
    query: '',
    top_k: 5,
  });

  const handleQuery = async () => {
    if (!config.query.trim()) {
      setError('Please enter a query');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await api.queryRAG(config);
      setResult(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Index Path
          </label>
          <input
            type="text"
            value={config.index}
            onChange={(e) => setConfig({...config, index: e.target.value})}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Query
          </label>
          <textarea
            value={config.query}
            onChange={(e) => setConfig({...config, query: e.target.value})}
            placeholder="What is RAG?"
            rows={4}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 font-mono"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Top K Results
          </label>
          <input
            type="number"
            value={config.top_k}
            onChange={(e) => setConfig({...config, top_k: parseInt(e.target.value)})}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg"
          />
        </div>
      </div>

      {error && <ErrorMessage message={error} />}
      
      {result && result.results && (
        <div className="space-y-3">
          <h3 className="font-semibold text-gray-900">Search Results:</h3>
          {result.results.map((doc, idx) => (
            <div key={idx} className="border rounded-lg p-4 hover:shadow-md transition-shadow">
              <div className="flex items-start justify-between mb-2">
                <span className="text-sm font-medium text-blue-600">Result #{idx + 1}</span>
                <span className="text-xs text-gray-500">Score: {doc.score?.toFixed(4)}</span>
              </div>
              <p className="text-sm text-gray-700">{doc.text}</p>
            </div>
          ))}
        </div>
      )}

      <button
        onClick={handleQuery}
        disabled={loading}
        className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 font-medium"
      >
        {loading ? <Loader2 className="h-5 w-5 animate-spin" /> : <Search className="h-5 w-5" />}
        {loading ? 'Searching...' : 'Search'}
      </button>
    </div>
  );
}

function ErrorMessage({ message }) {
  return (
    <div className="bg-red-50 border border-red-200 rounded-lg p-3 flex items-start gap-2">
      <div className="flex-shrink-0">⚠️</div>
      <p className="text-red-800 text-sm">{message}</p>
    </div>
  );
}

function SuccessMessage({ message }) {
  return (
    <div className="bg-green-50 border border-green-200 rounded-lg p-3 flex items-center gap-2">
      <CheckCircle className="h-5 w-5 text-green-600" />
      <p className="text-green-800 text-sm">{message}</p>
    </div>
  );
}
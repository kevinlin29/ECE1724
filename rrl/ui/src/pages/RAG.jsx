// src/pages/RAG.jsx
import React, { useState } from 'react';
import { FileText, Database, Search, Layers, Play, CheckCircle, Loader2, MessageSquare, Settings, ChevronDown, ChevronUp } from 'lucide-react';
import api from '../api';

export default function RAG() {
  const [activeTab, setActiveTab] = useState('setup'); // setup, query, generate
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
          <nav className="flex -mb-px overflow-x-auto">
            <TabButton
              active={activeTab === 'setup'}
              onClick={() => setActiveTab('setup')}
              icon={Settings}
            >
              1. Setup
            </TabButton>
            <TabButton
              active={activeTab === 'query'}
              onClick={() => setActiveTab('query')}
              icon={Search}
            >
              2. Search
            </TabButton>
            <TabButton
              active={activeTab === 'generate'}
              onClick={() => setActiveTab('generate')}
              icon={MessageSquare}
            >
              3. Generate (LLM)
            </TabButton>
          </nav>
        </div>

        <div className="p-6">
          {activeTab === 'setup' && (
            <SetupPanel loading={loading} setLoading={setLoading} result={result} setResult={setResult} error={error} setError={setError} />
          )}
          {activeTab === 'query' && (
            <QueryPanel loading={loading} setLoading={setLoading} result={result} setResult={setResult} error={error} setError={setError} />
          )}
          {activeTab === 'generate' && (
            <GeneratePanel loading={loading} setLoading={setLoading} result={result} setResult={setResult} error={error} setError={setError} />
          )}
        </div>
      </div>

      {/* Workflow Guide */}
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-blue-900 mb-3">RAG Workflow</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <WorkflowStep number="1" title="Setup" description="Ingest, embed, and index documents" />
          <WorkflowStep number="2" title="Search" description="Retrieve relevant documents" />
          <WorkflowStep number="3" title="Generate" description="LLM-powered answers" />
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

// Setup Panel - Combined Ingest + Embed + Index
function SetupPanel({ loading, setLoading, result, setResult, error, setError }) {
  const [config, setConfig] = useState({
    input: './docs',
    output: './data',
    chunk_size: 512,
    chunk_overlap: 50,
    hardware: 'auto',
  });
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [currentStep, setCurrentStep] = useState('');

  const handleBuildIndex = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    setCurrentStep('Starting pipeline...');

    try {
      setCurrentStep('Running: Ingest → Embed → Index');
      const res = await api.ragPipeline(config);
      setResult(res);
      setCurrentStep('');
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
      setCurrentStep('');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-lg p-4">
        <p className="text-blue-800 text-sm">
          <strong>Setup Pipeline:</strong> This will automatically run the full RAG setup process:
          <span className="font-mono ml-2">Ingest → Embed → Index</span>
        </p>
        <p className="text-blue-600 text-xs mt-1">
          Uses <code className="bg-blue-100 px-1 rounded">bert-base-uncased</code> for embeddings
        </p>
      </div>

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
            Output Directory
          </label>
          <input
            type="text"
            value={config.output}
            onChange={(e) => setConfig({...config, output: e.target.value})}
            placeholder="./data"
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
          />
          <p className="mt-1 text-sm text-gray-500">
            Base output directory (creates chunks/, embeddings/, index/ subdirectories)
          </p>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Hardware
          </label>
          <select
            value={config.hardware}
            onChange={(e) => setConfig({...config, hardware: e.target.value})}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg"
          >
            <option value="auto">Auto (CUDA → Metal → CPU)</option>
            <option value="cuda">CUDA (NVIDIA GPU)</option>
            <option value="metal">Metal (Apple GPU)</option>
            <option value="cpu">CPU</option>
          </select>
        </div>

        {/* Advanced Settings */}
        <div className="border rounded-lg">
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="w-full flex items-center justify-between px-4 py-3 text-sm font-medium text-gray-700 hover:bg-gray-50"
          >
            <span>Advanced Settings</span>
            {showAdvanced ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
          </button>
          {showAdvanced && (
            <div className="px-4 pb-4 border-t">
              <div className="grid grid-cols-2 gap-4 mt-4">
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
          )}
        </div>
      </div>

      {currentStep && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 flex items-center gap-2">
          <Loader2 className="h-5 w-5 text-blue-600 animate-spin" />
          <p className="text-blue-800 text-sm">{currentStep}</p>
        </div>
      )}

      {error && <ErrorMessage message={error} />}
      {result && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-2">
            <CheckCircle className="h-5 w-5 text-green-600" />
            <p className="text-green-800 font-medium">Pipeline Complete!</p>
          </div>
          <div className="text-sm text-green-700 space-y-1">
            <p>Chunks: {result.chunks_count || 0}</p>
            <p>Index path: <code className="bg-green-100 px-1 rounded">{result.index_path || config.output + '/index'}</code></p>
          </div>
        </div>
      )}

      <button
        onClick={handleBuildIndex}
        disabled={loading}
        className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 font-medium"
      >
        {loading ? <Loader2 className="h-5 w-5 animate-spin" /> : <Play className="h-5 w-5" />}
        {loading ? 'Building...' : 'Build Index'}
      </button>
    </div>
  );
}

// Query Panel
function QueryPanel({ loading, setLoading, result, setResult, error, setError }) {
  const [config, setConfig] = useState({
    index: './data/index',
    query: '',
    top_k: 5,
    retriever: 'hybrid',
    model: 'bert-base-uncased',
    backend: 'token',
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
            placeholder="How can I make Risotto alla Milanese?"
            rows={3}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
          />
        </div>

        <div className="grid grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Top K
            </label>
            <input
              type="number"
              value={config.top_k}
              onChange={(e) => setConfig({...config, top_k: parseInt(e.target.value)})}
              min={1}
              max={20}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Retriever
            </label>
            <select
              value={config.retriever}
              onChange={(e) => setConfig({...config, retriever: e.target.value})}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg"
            >
              <option value="hybrid">Hybrid (HNSW + BM25)</option>
              <option value="hnsw">HNSW (Dense)</option>
              <option value="bm25">BM25 (Sparse)</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Backend
            </label>
            <select
              value={config.backend}
              onChange={(e) => setConfig({...config, backend: e.target.value})}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg"
            >
              <option value="token">Token</option>
              <option value="mock">Mock</option>
            </select>
          </div>
        </div>
      </div>

      <button
        onClick={handleQuery}
        disabled={loading}
        className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 font-medium"
      >
        {loading ? <Loader2 className="h-5 w-5 animate-spin" /> : <Search className="h-5 w-5" />}
        {loading ? 'Searching...' : 'Search'}
      </button>

      {error && <ErrorMessage message={error} />}

      {result && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="font-semibold text-gray-900">
              Results for: "{result.query}"
            </h3>
            <span className="text-sm text-gray-500">
              {result.results?.length || 0} results found
            </span>
          </div>

          {result.results && result.results.length > 0 ? (
            <div className="space-y-3">
              {result.results.map((doc, idx) => (
                <div key={idx} className="border rounded-lg p-4 hover:shadow-md transition-shadow bg-white">
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <span className="inline-flex items-center justify-center w-6 h-6 rounded-full bg-blue-100 text-blue-600 text-xs font-bold">
                        {doc.rank || idx + 1}
                      </span>
                      <span className="text-sm font-medium text-gray-600">{doc.doc_id}</span>
                    </div>
                    <span className="text-xs font-mono bg-gray-100 px-2 py-1 rounded">
                      Score: {doc.score?.toFixed(4)}
                    </span>
                  </div>
                  <p className="text-sm text-gray-700 whitespace-pre-wrap">{doc.text}</p>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8 text-gray-500">
              No results found. Try a different query.
            </div>
          )}

          {/* Raw output toggle */}
          <details className="mt-4">
            <summary className="cursor-pointer text-sm text-gray-500 hover:text-gray-700">
              Show raw output
            </summary>
            <pre className="mt-2 p-3 bg-gray-50 rounded-lg text-xs overflow-x-auto whitespace-pre-wrap">
              {result.raw_output}
            </pre>
          </details>
        </div>
      )}
    </div>
  );
}

// Generate Panel - Full RAG with LLM
function GeneratePanel({ loading, setLoading, result, setResult, error, setError }) {
  const [config, setConfig] = useState({
    index: './data/index',
    query: '',
    generator: '',
    embedder: 'bert-base-uncased',
    generator_checkpoint: '',
    embedder_checkpoint: '',
    top_k: 5,
    retriever: 'hybrid',
    temperature: 0.7,
    max_tokens: 512,
    template: 'default',
    device: 'auto',
  });
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [models, setModels] = useState([]);

  // Load models on mount
  React.useEffect(() => {
    loadModels();
  }, []);

  const loadModels = async () => {
    try {
      const res = await api.listGeneratorModels();
      setModels(res.models || []);
    } catch (err) {
      console.error('Failed to load models:', err);
    }
  };

  const handleSelectModel = (model) => {
    setConfig({
      ...config,
      generator: model.path,
      generator_checkpoint: model.checkpoint_path || '',
    });
  };

  const handleGenerate = async () => {
    if (!config.query.trim()) {
      setError('Please enter a question');
      return;
    }
    if (!config.generator.trim()) {
      setError('Please select a model or enter a path');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const requestConfig = {
        ...config,
        generator_checkpoint: config.generator_checkpoint || null,
        embedder_checkpoint: config.embedder_checkpoint || null,
      };
      const res = await api.ragGenerate(requestConfig);
      setResult(res);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-purple-50 to-pink-50 border border-purple-200 rounded-lg p-4">
        <p className="text-purple-800 text-sm">
          <strong>Full RAG Pipeline:</strong> This uses an LLM to generate natural language answers
          based on retrieved documents. Requires the <code className="bg-purple-100 px-1 rounded">cuda</code> feature.
        </p>
      </div>

      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Your Question
          </label>
          <textarea
            value={config.query}
            onChange={(e) => setConfig({...config, query: e.target.value})}
            placeholder="How do I make Risotto alla Milanese?"
            rows={3}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500"
          />
        </div>

        <div className="grid grid-cols-2 gap-4">
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
              Device
            </label>
            <select
              value={config.device}
              onChange={(e) => setConfig({...config, device: e.target.value})}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg"
            >
              <option value="auto">Auto (CUDA → CPU)</option>
              <option value="cuda">CUDA</option>
              <option value="cpu">CPU</option>
            </select>
          </div>
        </div>

        {/* Model Selection Section */}
        <div className="border rounded-lg p-4 space-y-4 bg-gray-50">
          <div className="flex items-center justify-between">
            <h4 className="font-medium text-gray-900">LLM Model</h4>
            <a
              href="/models"
              className="text-sm text-purple-600 hover:text-purple-800"
            >
              Manage Models →
            </a>
          </div>

          {/* Model List */}
          {models.length > 0 ? (
            <div className="space-y-2 max-h-48 overflow-y-auto">
              {models.map((model) => (
                <div
                  key={model.name}
                  className={`flex items-center p-3 rounded-lg border cursor-pointer transition-colors ${
                    config.generator === model.path
                      ? 'border-purple-500 bg-purple-50'
                      : 'border-gray-200 bg-white hover:border-purple-300'
                  }`}
                  onClick={() => handleSelectModel(model)}
                >
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-gray-900">{model.name}</span>
                      {model.is_finetuned && (
                        <span className="text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded">Fine-tuned</span>
                      )}
                      <span className="text-xs text-gray-500">{model.architecture}</span>
                    </div>
                    <p className="text-xs text-gray-500 truncate mt-1">{model.path}</p>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-4">
              <p className="text-sm text-gray-500">No models registered.</p>
              <a href="/models" className="text-sm text-purple-600 hover:text-purple-800">
                Add models in the Models page →
              </a>
            </div>
          )}

          {/* Manual Path Input */}
          <div className="border-t pt-4">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Or enter path manually
            </label>
            <input
              type="text"
              value={config.generator}
              onChange={(e) => setConfig({...config, generator: e.target.value})}
              placeholder="/path/to/model or Qwen/Qwen2.5-0.5B"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
            />
          </div>
        </div>

        {/* Checkpoint Paths */}
        <div className="border rounded-lg p-4 space-y-4 bg-gray-50">
          <h4 className="font-medium text-gray-900">Checkpoints (Optional)</h4>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              LLM Checkpoint Path
            </label>
            <input
              type="text"
              value={config.generator_checkpoint}
              onChange={(e) => setConfig({...config, generator_checkpoint: e.target.value})}
              placeholder="/path/to/checkpoint.safetensors"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Embedder Checkpoint Path
            </label>
            <input
              type="text"
              value={config.embedder_checkpoint}
              onChange={(e) => setConfig({...config, embedder_checkpoint: e.target.value})}
              placeholder="/path/to/embedder_checkpoint.safetensors"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg"
            />
          </div>
        </div>

        {/* Advanced Settings */}
        <div className="border rounded-lg">
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="w-full flex items-center justify-between px-4 py-3 text-sm font-medium text-gray-700 hover:bg-gray-50"
          >
            <span>Advanced Settings</span>
            {showAdvanced ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
          </button>
          {showAdvanced && (
            <div className="px-4 pb-4 border-t">
              <div className="grid grid-cols-3 gap-4 mt-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Top K Documents
                  </label>
                  <input
                    type="number"
                    value={config.top_k}
                    onChange={(e) => setConfig({...config, top_k: parseInt(e.target.value)})}
                    min={1}
                    max={20}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Temperature
                  </label>
                  <input
                    type="number"
                    value={config.temperature}
                    onChange={(e) => setConfig({...config, temperature: parseFloat(e.target.value)})}
                    min={0}
                    max={2}
                    step={0.1}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Max Tokens
                  </label>
                  <input
                    type="number"
                    value={config.max_tokens}
                    onChange={(e) => setConfig({...config, max_tokens: parseInt(e.target.value)})}
                    min={50}
                    max={2048}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4 mt-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Retriever
                  </label>
                  <select
                    value={config.retriever}
                    onChange={(e) => setConfig({...config, retriever: e.target.value})}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg"
                  >
                    <option value="hybrid">Hybrid</option>
                    <option value="dense">Dense (HNSW)</option>
                    <option value="sparse">Sparse (BM25)</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Template
                  </label>
                  <select
                    value={config.template}
                    onChange={(e) => setConfig({...config, template: e.target.value})}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg"
                  >
                    <option value="default">Default</option>
                    <option value="concise">Concise</option>
                    <option value="detailed">Detailed</option>
                    <option value="recipe">Recipe</option>
                    <option value="chat">Chat</option>
                  </select>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      <button
        onClick={handleGenerate}
        disabled={loading}
        className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:bg-gray-400 font-medium"
      >
        {loading ? <Loader2 className="h-5 w-5 animate-spin" /> : <MessageSquare className="h-5 w-5" />}
        {loading ? 'Generating Answer...' : 'Generate Answer'}
      </button>

      {error && <ErrorMessage message={error} />}

      {result && (
        <div className="space-y-4">
          <div className="bg-white border rounded-lg p-6 shadow-sm">
            <div className="flex items-center gap-2 mb-3">
              <MessageSquare className="h-5 w-5 text-purple-600" />
              <h3 className="font-semibold text-gray-900">Answer</h3>
            </div>
            <div className="prose prose-sm max-w-none">
              <p className="text-gray-700 whitespace-pre-wrap leading-relaxed">
                {result.answer}
              </p>
            </div>
          </div>

          {result.sources && result.sources.length > 0 && (
            <div className="bg-gray-50 border rounded-lg p-4">
              <h4 className="text-sm font-medium text-gray-700 mb-2">Sources:</h4>
              <ul className="text-sm text-gray-600 space-y-1">
                {result.sources.map((source, idx) => (
                  <li key={idx} className="flex items-center gap-2">
                    <span className="text-purple-500">•</span>
                    {source}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Raw output toggle */}
          <details className="mt-4">
            <summary className="cursor-pointer text-sm text-gray-500 hover:text-gray-700">
              Show raw output
            </summary>
            <pre className="mt-2 p-3 bg-gray-50 rounded-lg text-xs overflow-x-auto whitespace-pre-wrap">
              {result.raw_output}
            </pre>
          </details>
        </div>
      )}
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

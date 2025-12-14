# RRL Web Server - FastAPI Backend
#
# Run: uvicorn server:app --reload --port 8000
#
# IMPORTANT: Build the binary first before starting the server:
#   cargo build --release --features cuda    (for CUDA support)
#   cargo build --release --features training  (for CPU-only training)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import json
import subprocess
import os
from pathlib import Path
import uuid
from datetime import datetime

from pydantic import BaseModel
from typing import Optional


def expand_path(path: str) -> str:
    """Expand ~ and environment variables in paths"""
    if path:
        return os.path.expanduser(os.path.expandvars(path))
    return path


# Path to pre-built RRL binary (build once with: cargo build --release --features cuda)
RRL_BINARY = os.path.join(os.path.dirname(__file__), "target", "release", "rrl")

def get_rrl_command(line_buffered=False):
    """Get the RRL command - uses pre-built binary if available, falls back to cargo run

    Args:
        line_buffered: If True, wrap command with stdbuf for real-time output streaming
    """
    if os.path.exists(RRL_BINARY):
        cmd = [RRL_BINARY]
    else:
        print(f"Warning: Pre-built binary not found at {RRL_BINARY}")
        print("Using 'cargo run' (slower). Build with: cargo build --release --features cuda")
        cmd = ["cargo", "run", "--release", "--features", "cuda", "--"]

    # Use stdbuf to force line-buffered output for real-time log streaming
    if line_buffered:
        # stdbuf -oL forces stdout to be line-buffered
        return ["stdbuf", "-oL"] + cmd
    return cmd


# Model Registry - persisted to JSON file
MODEL_REGISTRY_FILE = "model_registry.json"

class GeneratorModelInfo(BaseModel):
    name: str
    path: str
    architecture: str = "unknown"
    size: str = "unknown"
    is_finetuned: bool = False
    checkpoint_path: Optional[str] = None
    added_at: str = ""

class ModelRegistry:
    def __init__(self):
        self.models: Dict[str, GeneratorModelInfo] = {}
        self.load()

    def load(self):
        """Load registry from file"""
        if os.path.exists(MODEL_REGISTRY_FILE):
            try:
                with open(MODEL_REGISTRY_FILE, 'r') as f:
                    data = json.load(f)
                    for name, info in data.items():
                        self.models[name] = GeneratorModelInfo(**info)
            except Exception as e:
                print(f"Warning: Could not load model registry: {e}")

    def save(self):
        """Save registry to file"""
        try:
            data = {name: model.dict() for name, model in self.models.items()}
            with open(MODEL_REGISTRY_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save model registry: {e}")

    def add_model(self, name: str, path: str, architecture: str = "unknown",
                  size: str = "unknown", is_finetuned: bool = False,
                  checkpoint_path: Optional[str] = None) -> GeneratorModelInfo:
        """Add a model to the registry"""
        expanded_path = expand_path(path)
        model = GeneratorModelInfo(
            name=name,
            path=expanded_path,
            architecture=architecture,
            size=size,
            is_finetuned=is_finetuned,
            checkpoint_path=checkpoint_path,
            added_at=datetime.now().isoformat()
        )
        self.models[name] = model
        self.save()
        return model

    def remove_model(self, name: str) -> bool:
        """Remove a model from the registry"""
        if name in self.models:
            del self.models[name]
            self.save()
            return True
        return False

    def get_model(self, name: str) -> Optional[GeneratorModelInfo]:
        """Get a model by name"""
        return self.models.get(name)

    def list_models(self) -> List[GeneratorModelInfo]:
        """List all models"""
        return list(self.models.values())

    def scan_finetuned_models(self, output_dir: str = "./output"):
        """Scan for fine-tuned model checkpoints"""
        output_path = Path(expand_path(output_dir))
        if not output_path.exists():
            return

        for subdir in output_path.iterdir():
            if subdir.is_dir():
                # Look for LoRA checkpoints
                checkpoint = subdir / "lora_checkpoint.safetensors"
                if checkpoint.exists():
                    name = f"finetuned-{subdir.name}"
                    if name not in self.models:
                        self.add_model(
                            name=name,
                            path=str(checkpoint),
                            architecture="LoRA",
                            is_finetuned=True,
                            checkpoint_path=str(checkpoint)
                        )

# Global model registry
model_registry = ModelRegistry()


app = FastAPI(title="RRL Training Server", version="1.0.0")

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React/Vite
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
active_trainings = {}
training_logs = {}
websocket_connections = []

# Data models
class TrainingConfig(BaseModel):
    model_name: str
    model_path: str = ""  # For generator models with local path
    dataset_path: str
    output_dir: str
    output_name: str = ""  # Name for the fine-tuned model
    epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 5e-5
    lora_rank: int = 8
    lora_alpha: float = 16.0
    device: str = "auto"  # auto-detects CUDA > Metal > CPU
    max_seq_length: int = 512
    gradient_accumulation: int = 1
    warmup_ratio: float = 0.1
    save_steps: int = 500
    logging_steps: int = 10

class IngestConfig(BaseModel):
    input: str
    output: str
    chunk_size: int = 512
    chunk_overlap: int = 50

class EmbedConfig(BaseModel):
    input: str
    output: str
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    backend: str = "token"
    hardware: str = "auto"  # auto-detects CUDA > Metal > CPU

class IndexConfig(BaseModel):
    chunks: str
    embeddings: str
    output: str
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    index_type: str = "both"

class QueryConfig(BaseModel):
    index: str
    query: str
    top_k: int = 5
    retriever: str = "hybrid"  # hybrid, hnsw, bm25
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    backend: str = "token"

class RagGenerateConfig(BaseModel):
    index: str
    query: str
    generator: str = ""  # Local path to LLM model (required)
    embedder: str = "bert-base-uncased"
    generator_checkpoint: Optional[str] = None  # Path to fine-tuned generator checkpoint
    embedder_checkpoint: Optional[str] = None   # Path to fine-tuned embedder checkpoint
    top_k: int = 5
    retriever: str = "hybrid"  # dense, sparse, hybrid
    temperature: float = 0.7
    max_tokens: int = 512
    template: str = "default"  # default, concise, detailed, recipe, chat
    device: str = "auto"  # auto, cpu, cuda, metal

class PipelineConfig(BaseModel):
    input: str              # Document directory
    output: str             # Base output dir (creates chunks/, embeddings/, index/)
    chunk_size: int = 512
    chunk_overlap: int = 50
    hardware: str = "auto"  # auto, cuda, cpu

class EvalConfig(BaseModel):
    model_name: str
    checkpoint_path: Optional[str] = None
    data_path: str
    lora_rank: int = 8
    lora_alpha: float = 16.0

class InferenceRequest(BaseModel):
    model_name: str
    checkpoint_path: Optional[str] = None
    queries: List[str]
    options: Optional[List[List[str]]] = None

class ModelInfo(BaseModel):
    name: str
    size: str
    architecture: str
    hidden_size: int
    recommended_rank: int

# Model catalog
AVAILABLE_MODELS = {
    "bert-base-uncased": ModelInfo(
        name="bert-base-uncased",
        size="110M",
        architecture="BERT",
        hidden_size=768,
        recommended_rank=8
    ),
    "roberta-base": ModelInfo(
        name="roberta-base",
        size="125M",
        architecture="RoBERTa",
        hidden_size=768,
        recommended_rank=8
    ),
    "BAAI/bge-base-en-v1.5": ModelInfo(
        name="BAAI/bge-base-en-v1.5",
        size="110M",
        architecture="BERT",
        hidden_size=768,
        recommended_rank=16
    ),
    "BAAI/bge-large-en-v1.5": ModelInfo(
        name="BAAI/bge-large-en-v1.5",
        size="335M",
        architecture="BERT",
        hidden_size=1024,
        recommended_rank=16
    ),
    "xlm-roberta-base": ModelInfo(
        name="xlm-roberta-base",
        size="270M",
        architecture="RoBERTa",
        hidden_size=768,
        recommended_rank=16
    ),
}

# WebSocket manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

# API Endpoints

@app.get("/")
async def root():
    return {"message": "RRL Training Server", "version": "1.0.0"}

@app.get("/health")
async def health():
    return {"status": "healthy", "active_trainings": len(active_trainings)}

@app.get("/models")
async def list_models():
    """Get list of available models"""
    return {"models": list(AVAILABLE_MODELS.values())}

@app.get("/models/{model_name}")
async def get_model_info(model_name: str):
    """Get detailed info about a specific model"""
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(status_code=404, detail="Model not found")
    return AVAILABLE_MODELS[model_name]


# Generator Model Management Endpoints
class AddModelRequest(BaseModel):
    name: str
    path: str
    architecture: str = "auto"

@app.get("/generator-models")
async def list_generator_models():
    """List all registered generator models"""
    # Scan for fine-tuned models first
    model_registry.scan_finetuned_models("./output")
    return {"models": model_registry.list_models()}

@app.post("/generator-models")
async def add_generator_model(request: AddModelRequest):
    """Add a new generator model to the registry"""
    expanded_path = expand_path(request.path)

    # Validate path exists
    if not os.path.exists(expanded_path):
        raise HTTPException(status_code=400, detail=f"Path does not exist: {expanded_path}")

    # Try to detect architecture from config.json
    architecture = request.architecture
    size = "unknown"
    if architecture == "auto":
        config_path = os.path.join(expanded_path, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path) as f:
                    config = json.load(f)
                    arch_list = config.get("architectures", [])
                    if arch_list:
                        architecture = arch_list[0]
                    elif config.get("model_type"):
                        architecture = config["model_type"]
                    # Try to estimate size
                    hidden = config.get("hidden_size", 0)
                    layers = config.get("num_hidden_layers", 0)
                    if hidden and layers:
                        params = hidden * hidden * layers * 4  # rough estimate
                        if params > 5e9:
                            size = f"~{params/1e9:.0f}B"
                        elif params > 1e6:
                            size = f"~{params/1e6:.0f}M"
            except:
                pass

    model = model_registry.add_model(
        name=request.name,
        path=expanded_path,
        architecture=architecture,
        size=size
    )
    return {"status": "success", "model": model}

@app.delete("/generator-models/{model_name}")
async def remove_generator_model(model_name: str):
    """Remove a generator model from the registry"""
    if model_registry.remove_model(model_name):
        return {"status": "success", "message": f"Model '{model_name}' removed"}
    raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

@app.get("/generator-models/{model_name}")
async def get_generator_model(model_name: str):
    """Get details of a specific generator model"""
    model = model_registry.get_model(model_name)
    if model:
        return model
    raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

@app.post("/generator-models/scan")
async def scan_finetuned_models(output_dir: str = "./output"):
    """Scan output directory for fine-tuned models"""
    model_registry.scan_finetuned_models(output_dir)
    return {"status": "success", "models": model_registry.list_models()}

@app.post("/upload/dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload training dataset"""
    upload_dir = Path("uploads/datasets")
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = upload_dir / file.filename
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    return {
        "filename": file.filename,
        "path": str(file_path),
        "size": len(content)
    }

@app.post("/train")
async def start_training(config: TrainingConfig):
    """Start a new training job"""
    job_id = str(uuid.uuid4())

    # Determine output directory (incorporate output_name if provided)
    output_dir = expand_path(config.output_dir)
    if config.output_name:
        output_dir = os.path.join(output_dir, config.output_name)

    # Expand dataset path
    dataset_path = expand_path(config.dataset_path)

    # Determine model: use model_path if provided (for generator models with local paths),
    # otherwise use model_name (for HuggingFace model IDs like "bert-base-uncased")
    model = config.model_name
    if config.model_path:
        model = expand_path(config.model_path)

    # Build command using pre-built binary with line-buffered output for real-time logs
    cmd = get_rrl_command(line_buffered=True) + [
        "train",
        "--data", dataset_path,
        "--output", output_dir,
        "--model", model,
        "--epochs", str(config.epochs),
        "--batch-size", str(config.batch_size),
        "--learning-rate", str(config.learning_rate),
        "--lora-rank", str(config.lora_rank),
        "--lora-alpha", str(config.lora_alpha),
        "--device", config.device,
        "--max-seq-length", str(config.max_seq_length),
        "--gradient-accumulation", str(config.gradient_accumulation),
        "--warmup-ratio", str(config.warmup_ratio),
        "--save-steps", str(config.save_steps),
        "--logging-steps", str(config.logging_steps),
    ]

    # Start training process
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    active_trainings[job_id] = {
        "id": job_id,
        "config": config.dict(),
        "process": process,
        "status": "running",
        "started_at": datetime.now().isoformat(),
        "current_step": 0,
        "current_loss": None,
    }

    training_logs[job_id] = []

    # Start log streaming task
    asyncio.create_task(stream_training_logs(job_id, process))

    return {
        "job_id": job_id,
        "status": "started",
        "message": "Training job started successfully"
    }

@app.post("/rag/ingest")
async def rag_ingest(config: IngestConfig):
    """Ingest documents and create chunks"""
    input_path = expand_path(config.input)
    output_path = expand_path(config.output)

    cmd = get_rrl_command() + [
        "ingest",
        "--input", input_path,
        "--output", output_path,
        "--chunk-size", str(config.chunk_size),
        "--chunk-overlap", str(config.chunk_overlap),
    ]

    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        error_msg = result.stderr or result.stdout or "Unknown error"
        print(f"Ingest error: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

    print(f"Ingest stdout: {result.stdout}")

    # Parse output to get chunk count
    chunks_count = 0
    for line in result.stdout.split('\n'):
        if "chunks" in line.lower() or "chunk" in line.lower():
            try:
                nums = [int(s) for s in line.split() if s.isdigit()]
                if nums:
                    chunks_count = nums[0]
            except:
                pass

    return {
        "status": "success",
        "chunks_count": chunks_count,
        "output": config.output
    }

@app.post("/rag/embed")
async def rag_embed(config: EmbedConfig):
    """Generate embeddings for chunks"""
    input_path = expand_path(config.input)
    output_path = expand_path(config.output)
    model_path = expand_path(config.model)

    cmd = get_rrl_command() + [
        "embed",
        "--input", input_path,
        "--output", output_path,
        "--model", model_path,
        "--backend", config.backend,
        "--hardware", config.hardware,
    ]

    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        error_msg = result.stderr or result.stdout or "Unknown error"
        print(f"Embed error: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

    print(f"Embed stdout: {result.stdout}")

    # Parse embeddings count from output
    embeddings_count = 0
    for line in result.stdout.split('\n'):
        if "embedding" in line.lower():
            try:
                embeddings_count = int(''.join(filter(str.isdigit, line)))
            except:
                pass

    return {
        "status": "success",
        "embeddings_count": embeddings_count,
        "output": config.output
    }

@app.post("/rag/index")
async def rag_index(config: IndexConfig):
    """Build vector index"""
    chunks_path = expand_path(config.chunks)
    embeddings_path = expand_path(config.embeddings)
    output_path = expand_path(config.output)
    model_path = expand_path(config.model)

    cmd = get_rrl_command() + [
        "index",
        "--chunks", chunks_path,
        "--embeddings", embeddings_path,
        "--output", output_path,
        "--model", model_path,
        "--index-type", config.index_type,
    ]

    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        error_msg = result.stderr or result.stdout or "Unknown error"
        print(f"Index error: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

    print(f"Index stdout: {result.stdout}")

    return {
        "status": "success",
        "index_path": config.output
    }

@app.post("/rag/pipeline")
async def rag_pipeline(config: PipelineConfig):
    """Run full RAG pipeline: ingest → embed → index"""
    # Expand paths
    input_path = expand_path(config.input)
    output_path = expand_path(config.output)

    # Create output subdirectories
    chunks_dir = os.path.join(output_path, "chunks")
    embeddings_dir = os.path.join(output_path, "embeddings")
    index_dir = os.path.join(output_path, "index")

    results = {
        "status": "running",
        "steps": [],
        "chunks_count": 0,
        "index_path": index_dir
    }

    # Step 1: Ingest
    print(f"[Pipeline] Step 1/3: Ingesting documents from {input_path}")
    ingest_cmd = get_rrl_command() + [
        "ingest",
        "--input", input_path,
        "--output", chunks_dir,
        "--chunk-size", str(config.chunk_size),
        "--chunk-overlap", str(config.chunk_overlap),
    ]
    print(f"Running: {' '.join(ingest_cmd)}")
    ingest_result = subprocess.run(ingest_cmd, capture_output=True, text=True)

    if ingest_result.returncode != 0:
        error_msg = ingest_result.stderr or ingest_result.stdout or "Ingest failed"
        print(f"Ingest error: {error_msg}")
        raise HTTPException(status_code=500, detail=f"Ingest failed: {error_msg}")

    results["steps"].append({"step": "ingest", "status": "completed"})
    print(f"[Pipeline] Ingest completed")

    # Parse chunks count
    for line in ingest_result.stdout.split('\n'):
        if "chunk" in line.lower():
            try:
                nums = [int(s) for s in line.split() if s.isdigit()]
                if nums:
                    results["chunks_count"] = nums[0]
            except:
                pass

    # Step 2: Embed
    print(f"[Pipeline] Step 2/3: Generating embeddings")
    embed_cmd = get_rrl_command() + [
        "embed",
        "--input", chunks_dir,
        "--output", embeddings_dir,
        "--model", "bert-base-uncased",
        "--backend", "token",
        "--hardware", config.hardware,
    ]
    print(f"Running: {' '.join(embed_cmd)}")
    embed_result = subprocess.run(embed_cmd, capture_output=True, text=True)

    if embed_result.returncode != 0:
        error_msg = embed_result.stderr or embed_result.stdout or "Embed failed"
        print(f"Embed error: {error_msg}")
        raise HTTPException(status_code=500, detail=f"Embed failed: {error_msg}")

    results["steps"].append({"step": "embed", "status": "completed"})
    print(f"[Pipeline] Embed completed")

    # Step 3: Index
    print(f"[Pipeline] Step 3/3: Building index")
    index_cmd = get_rrl_command() + [
        "index",
        "--chunks", chunks_dir,
        "--embeddings", embeddings_dir,
        "--output", index_dir,
        "--model", "bert-base-uncased",
        "--index-type", "both",
    ]
    print(f"Running: {' '.join(index_cmd)}")
    index_result = subprocess.run(index_cmd, capture_output=True, text=True)

    if index_result.returncode != 0:
        error_msg = index_result.stderr or index_result.stdout or "Index failed"
        print(f"Index error: {error_msg}")
        raise HTTPException(status_code=500, detail=f"Index failed: {error_msg}")

    results["steps"].append({"step": "index", "status": "completed"})
    results["status"] = "completed"
    print(f"[Pipeline] Index completed. Full pipeline done!")

    return results

@app.post("/rag/query")
async def rag_query(config: QueryConfig):
    """Query the RAG system"""
    index_path = expand_path(config.index)
    model_path = expand_path(config.model)

    cmd = get_rrl_command() + [
        "query",
        "--index", index_path,
        "--query", config.query,
        "--top-k", str(config.top_k),
        "--retriever", config.retriever,
        "--model", model_path,
        "--backend", config.backend,
    ]

    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        error_msg = result.stderr or result.stdout or "Unknown error"
        print(f"Query error: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

    print(f"Query stdout: {result.stdout}")

    # Parse results from output
    results = []
    current_result = {}
    content_lines = []
    in_content = False

    for line in result.stdout.split('\n'):
        line = line.strip()

        # Parse rank line: "Rank 1: doc_xxx (score: 0.0323)"
        if line.startswith("Rank "):
            # Save previous result if exists
            if current_result and content_lines:
                current_result["text"] = '\n'.join(content_lines).strip()
                results.append(current_result)
                content_lines = []

            current_result = {}
            try:
                # Extract rank, doc_id, and score
                parts = line.split(":")
                rank = int(parts[0].replace("Rank ", "").strip())
                rest = ":".join(parts[1:]).strip()

                # Parse "doc_xxx (score: 0.0323)"
                if "(score:" in rest:
                    doc_id = rest.split("(score:")[0].strip()
                    score_str = rest.split("(score:")[1].replace(")", "").strip()
                    score = float(score_str)
                    current_result = {"rank": rank, "doc_id": doc_id, "score": score}
            except:
                pass
            in_content = False

        elif line.startswith("Content:"):
            in_content = True
            # Get content after "Content: "
            content = line.replace("Content:", "").strip()
            if content:
                content_lines.append(content)
        elif in_content and line and not line.startswith("Document:") and line != "...":
            content_lines.append(line)
        elif line.startswith("Document:"):
            in_content = False

    # Don't forget the last result
    if current_result and content_lines:
        current_result["text"] = '\n'.join(content_lines).strip()
        results.append(current_result)

    return {
        "query": config.query,
        "results": results,
        "raw_output": result.stdout
    }

@app.post("/rag/generate")
async def rag_generate(config: RagGenerateConfig):
    """Full RAG: Retrieve + Generate with LLM"""
    if not config.generator:
        raise HTTPException(status_code=400, detail="Generator model path is required")

    # Expand ~ and environment variables in paths
    generator_path = expand_path(config.generator)
    index_path = expand_path(config.index)
    embedder_path = expand_path(config.embedder)

    cmd = get_rrl_command() + [
        "rag",
        "--index", index_path,
        "--query", config.query,
        "--generator", generator_path,
        "--embedder", embedder_path,
        "-k", str(config.top_k),
        "--retriever", config.retriever,
        "--temperature", str(config.temperature),
        "--max-tokens", str(config.max_tokens),
        "--template", config.template,
        "--format", "text",
        "--device", config.device,
    ]

    # Add optional checkpoint paths (also expand)
    if config.generator_checkpoint:
        cmd.extend(["--generator-checkpoint", expand_path(config.generator_checkpoint)])
    if config.embedder_checkpoint:
        cmd.extend(["--embedder-checkpoint", expand_path(config.embedder_checkpoint)])

    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 min timeout

    if result.returncode != 0:
        error_msg = result.stderr or result.stdout or "Unknown error"
        print(f"RAG generate error: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

    print(f"RAG generate stdout: {result.stdout}")

    # Parse the response - look for the generated answer
    answer = ""
    sources = []
    lines = result.stdout.split('\n')

    in_answer = False
    answer_lines = []

    for line in lines:
        # Look for answer section
        if "Answer:" in line or "Response:" in line:
            in_answer = True
            # Get text after "Answer:" on same line
            after_label = line.split(":", 1)[-1].strip()
            if after_label:
                answer_lines.append(after_label)
        elif in_answer:
            # Stop at sources or empty sections
            if line.startswith("Sources:") or line.startswith("---") or line.startswith("Retrieved"):
                in_answer = False
            elif line.strip():
                answer_lines.append(line)

        # Parse sources
        if line.startswith("- ") and "doc_" in line:
            sources.append(line[2:].strip())

    answer = '\n'.join(answer_lines).strip()

    # If no structured answer found, use full output
    if not answer:
        answer = result.stdout

    return {
        "query": config.query,
        "answer": answer,
        "sources": sources,
        "raw_output": result.stdout
    }

async def stream_training_logs(job_id: str, process: subprocess.Popen):
    """Stream training logs via WebSocket"""
    try:
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue

            training_logs[job_id].append(line)

            # Check for error patterns (OOM, CUDA errors, panics)
            is_error = any(err in line.lower() for err in [
                'out of memory', 'oom', 'cuda error', 'memory allocation',
                'panic', 'error:', 'failed', 'exception', 'alloc::'
            ])

            # Parse metrics from log line
            metrics = parse_training_metrics(line)

            if metrics:
                active_trainings[job_id].update(metrics)

            # Broadcast to WebSocket clients
            await manager.broadcast({
                "type": "training_log",
                "job_id": job_id,
                "log": line,
                "metrics": metrics,
                "is_error": is_error
            })

            # If we detect an error, mark it
            if is_error:
                active_trainings[job_id]["last_error"] = line

        # Training finished - wait for process and get return code
        return_code = process.wait()

        # Determine status based on return code
        if return_code == 0:
            status = "completed"
            error_msg = None
        else:
            status = "failed"
            # Try to get last error from logs
            error_msg = active_trainings[job_id].get("last_error", f"Process exited with code {return_code}")

            # Common error code meanings
            if return_code == -9:
                error_msg = "Process killed (likely OOM - Out of Memory)"
            elif return_code == -6:
                error_msg = "Process aborted (SIGABRT - possibly assertion failure or OOM)"
            elif return_code == 137:
                error_msg = "Process killed by OOM killer (exit code 137)"
            elif return_code == 139:
                error_msg = "Segmentation fault (exit code 139)"

            # Log the error
            error_log = f"ERROR: {error_msg}"
            training_logs[job_id].append(error_log)
            await manager.broadcast({
                "type": "training_log",
                "job_id": job_id,
                "log": error_log,
                "metrics": None,
                "is_error": True
            })

        active_trainings[job_id]["status"] = status
        active_trainings[job_id]["finished_at"] = datetime.now().isoformat()
        if error_msg:
            active_trainings[job_id]["error"] = error_msg

        await manager.broadcast({
            "type": "training_complete",
            "job_id": job_id,
            "status": status,
            "error": error_msg
        })

    except Exception as e:
        error_msg = f"Training error: {str(e)}"
        active_trainings[job_id]["status"] = "error"
        active_trainings[job_id]["error"] = error_msg
        training_logs[job_id].append(f"ERROR: {error_msg}")

        await manager.broadcast({
            "type": "training_complete",
            "job_id": job_id,
            "status": "error",
            "error": error_msg
        })

def parse_training_metrics(log_line: str) -> Optional[Dict[str, Any]]:
    """Parse metrics from training log line"""
    # Example: "Step 100 | Epoch 1 | Loss: 0.4523 | LR: 5.00e-05"
    metrics = {}
    
    if "Step" in log_line and "Loss:" in log_line:
        parts = log_line.split("|")
        for part in parts:
            part = part.strip()
            if part.startswith("Step"):
                metrics["current_step"] = int(part.split()[1])
            elif part.startswith("Epoch"):
                metrics["current_epoch"] = int(part.split()[1])
            elif part.startswith("Loss:"):
                metrics["current_loss"] = float(part.split()[1])
            elif part.startswith("LR:"):
                metrics["learning_rate"] = float(part.split()[1])
    
    return metrics if metrics else None

@app.get("/train/{job_id}")
async def get_training_status(job_id: str):
    """Get status of a training job"""
    if job_id not in active_trainings:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    job = active_trainings[job_id].copy()
    job.pop("process", None)  # Don't serialize process object
    
    return job

@app.get("/train/{job_id}/logs")
async def get_training_logs(job_id: str, lines: int = 100):
    """Get training logs"""
    if job_id not in training_logs:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    logs = training_logs[job_id]
    return {"logs": logs[-lines:]}

@app.post("/train/{job_id}/stop")
async def stop_training(job_id: str):
    """Stop a running training job"""
    if job_id not in active_trainings:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    job = active_trainings[job_id]
    if job["status"] == "running":
        job["process"].terminate()
        job["status"] = "stopped"
    
    return {"status": "stopped"}

@app.get("/trainings")
async def list_trainings():
    """List all training jobs"""
    jobs = []
    for job_id, job in active_trainings.items():
        job_copy = job.copy()
        job_copy.pop("process", None)
        jobs.append(job_copy)
    
    return {"trainings": jobs}

@app.post("/eval")
async def run_evaluation(config: EvalConfig):
    """Run evaluation on trained model"""
    cmd = get_rrl_command() + [
        "eval-mc",
        "--data", config.data_path,
        "--model", config.model_name,
        "--lora-rank", str(config.lora_rank),
        "--lora-alpha", str(config.lora_alpha),
    ]
    
    if config.checkpoint_path:
        cmd.extend(["--checkpoint", config.checkpoint_path])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse evaluation results
    output = result.stdout
    accuracy = None
    mrr = None
    
    for line in output.split('\n'):
        if "Accuracy:" in line:
            accuracy = float(line.split(":")[-1].strip().rstrip('%'))
        elif "MRR:" in line:
            mrr = float(line.split(":")[-1].strip())
    
    return {
        "accuracy": accuracy,
        "mrr": mrr,
        "output": output
    }

@app.post("/inference")
async def run_inference(request: InferenceRequest):
    """Run inference with trained model"""
    # This would integrate with your Rust inference code
    # Placeholder for now
    return {
        "results": [
            {"query": q, "embedding": [0.0] * 768}
            for q in request.queries
        ]
    }

@app.get("/checkpoints")
async def list_checkpoints(output_dir: str = "./outputs"):
    """List available model checkpoints"""
    checkpoints = []
    output_path = Path(output_dir)
    
    if output_path.exists():
        for ckpt_dir in output_path.glob("checkpoint-*"):
            if ckpt_dir.is_dir():
                checkpoints.append({
                    "name": ckpt_dir.name,
                    "path": str(ckpt_dir),
                    "created_at": datetime.fromtimestamp(ckpt_dir.stat().st_mtime).isoformat()
                })
    
    return {"checkpoints": checkpoints}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for live training updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            # Echo back for heartbeat
            await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
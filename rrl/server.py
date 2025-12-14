# RRL Web Server - FastAPI Backend
# 
# Run: uvicorn server:app --reload --port 8000

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
    dataset_path: str
    output_dir: str
    epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 5e-5
    lora_rank: int = 8
    lora_alpha: float = 16.0
    device: str = "auto"
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
    model: str = "BAAI/bge-base-en-v1.5"
    batch_size: int = 32

class IndexConfig(BaseModel):
    embeddings: str
    output: str
    index_type: str = "hnsw"

class QueryConfig(BaseModel):
    index: str
    query: str
    top_k: int = 5

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
    
    # Build command
    cmd = [
        "cargo", "run", "--release", "--features", "training", "--",
        "train",
        "--data", config.dataset_path,
        "--output", config.output_dir,
        "--model", config.model_name,
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

@app.post("/rag/ingest")
async def rag_ingest(config: IngestConfig):
    """Ingest documents and create chunks"""
    cmd = [
        "cargo", "run", "--release", "--",
        "ingest",
        "--input", config.input,
        "--output", config.output,
        "--chunk-size", str(config.chunk_size),
        "--chunk-overlap", str(config.chunk_overlap),
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=result.stderr)
    
    # Parse output to get chunk count
    chunks_count = 0
    for line in result.stdout.split('\n'):
        if "chunks" in line.lower():
            try:
                chunks_count = int(''.join(filter(str.isdigit, line)))
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
    cmd = [
        "cargo", "run", "--release", "--",
        "embed",
        "--input", config.input,
        "--output", config.output,
        "--model", config.model,
        "--batch-size", str(config.batch_size),
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=result.stderr)
    
    return {
        "status": "success",
        "embeddings_count": 0,  # Parse from output
        "output": config.output
    }

@app.post("/rag/index")
async def rag_index(config: IndexConfig):
    """Build vector index"""
    cmd = [
        "cargo", "run", "--release", "--",
        "index", "build",
        "--embeddings", config.embeddings,
        "--output", config.output,
        "--type", config.index_type,
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=result.stderr)
    
    return {
        "status": "success",
        "index_path": config.output
    }

@app.post("/rag/query")
async def rag_query(config: QueryConfig):
    """Query the RAG system"""
    cmd = [
        "cargo", "run", "--release", "--",
        "query",
        "--index", config.index,
        "--query", config.query,
        "--top-k", str(config.top_k),
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=result.stderr)
    
    # Parse results from output
    results = []
    # This would parse the actual output format from your Rust code
    # For now, return a placeholder
    
    return {
        "query": config.query,
        "results": results,
        "raw_output": result.stdout
    }
    
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

async def stream_training_logs(job_id: str, process: subprocess.Popen):
    """Stream training logs via WebSocket"""
    try:
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            
            training_logs[job_id].append(line)
            
            # Parse metrics from log line
            metrics = parse_training_metrics(line)
            
            if metrics:
                active_trainings[job_id].update(metrics)
            
            # Broadcast to WebSocket clients
            await manager.broadcast({
                "type": "training_log",
                "job_id": job_id,
                "log": line,
                "metrics": metrics
            })
        
        # Training finished
        process.wait()
        active_trainings[job_id]["status"] = "completed" if process.returncode == 0 else "failed"
        active_trainings[job_id]["finished_at"] = datetime.now().isoformat()
        
        await manager.broadcast({
            "type": "training_complete",
            "job_id": job_id,
            "status": active_trainings[job_id]["status"]
        })
        
    except Exception as e:
        active_trainings[job_id]["status"] = "error"
        active_trainings[job_id]["error"] = str(e)

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
    cmd = [
        "cargo", "run", "--release", "--features", "training", "--",
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
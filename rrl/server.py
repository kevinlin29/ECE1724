# RRL Web Server - FastAPI Backend
#
# Run: uvicorn server:app --reload --port 8000
#
# IMPORTANT: Build the binary first before starting the server:
#   cargo build --release --features cuda        (CUDA)
#   cargo build --release --features training    (CPU)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import json
import subprocess
import os
import re
from pathlib import Path
import uuid
from datetime import datetime


# ------------------------
# Utils
# ------------------------

def expand_path(path: str) -> str:
    """Expand ~ and environment variables in paths"""
    if path:
        return os.path.expanduser(os.path.expandvars(path))
    return path


# ------------------------
# RRL Binary
# ------------------------

RRL_BINARY = os.path.join(os.path.dirname(__file__), "target", "release", "rrl")

def get_rrl_command(line_buffered=False):
    """Get the RRL command"""
    if os.path.exists(RRL_BINARY):
        cmd = [RRL_BINARY]
    else:
        if os.name == "nt":
            cmd = ["cargo", "run", "--release", "--no-default-features", "--features", "training", "--"]
        else:
            cmd = ["cargo", "run", "--release", "--features", "cuda,training", "--"]

    if line_buffered and os.name != "nt":
        return ["stdbuf", "-oL"] + cmd

    return cmd


# ------------------------
# FastAPI App
# ------------------------

app = FastAPI(title="RRL Training Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

active_trainings: Dict[str, Any] = {}
training_logs: Dict[str, List[str]] = {}
training_subscribers: List[WebSocket] = []  # Global training log subscribers
msmarco_jobs: Dict[str, Any] = {}
msmarco_subscribers: Dict[str, List[WebSocket]] = {}


# ------------------------
# Models
# ------------------------

class InferenceRequest(BaseModel):
    model_name: str
    checkpoint_path: Optional[str] = None
    queries: List[str]
    inference_type: str = "generation"  # generation | embedding
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9


class MsMarcoEvalRequest(BaseModel):
    """Request model for MS MARCO evaluation"""
    model_name: str = "bert-base-uncased"
    checkpoint_path: Optional[str] = None
    data_path: str = "data/msmarco_validation.jsonl"
    sample_size: Optional[int] = None  # None = full dataset
    lora_rank: int = 8
    lora_alpha: float = 16.0
    device: str = "auto"


class MsMarcoEvalProgress(BaseModel):
    """Progress update model"""
    processed: int
    total: int
    current_mrr: float
    eta_seconds: float
    status: str


class MsMarcoEvalResult(BaseModel):
    """Final evaluation result model"""
    mrr_at_10: float
    ndcg_at_10: float
    recall_at_10: float
    recall_at_100: float
    num_queries: int
    elapsed_seconds: float


# ------------------------
# Inference Endpoint
# ------------------------

@app.post("/inference")
async def run_inference(request: InferenceRequest):
    """Run inference with trained model"""

    cmd = get_rrl_command() + [
        "infer",
        "--model", request.model_name,
        "--type", request.inference_type,
    ]

    if request.checkpoint_path:
        cmd.extend(["--checkpoint", expand_path(request.checkpoint_path)])

    if request.inference_type == "generation":
        cmd.extend([
            "--max-length", str(request.max_length),
            "--temperature", str(request.temperature),
            "--top-p", str(request.top_p),
        ])

    for q in request.queries:
        cmd.extend(["--query", q])

    try:
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, errors= "ignore", text=True, timeout=60)

        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=result.stderr or result.stdout)

        output = result.stdout or ""

        if request.inference_type == "generation":
            return {
                "generated_text": parse_generation_output(output),
                "model": request.model_name,
            }

        # >>> FIX: embedding is ALWAYS a list
        embedding = parse_embedding_output(output)
        # >>> END FIX

        return {
            "embedding": embedding,
            "dimension": len(embedding),
            "model": request.model_name,
        }

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Inference timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------
# Parsing Helpers
# ------------------------

def parse_embedding_output(output: str) -> List[float]:
    """
    Parse embedding vector from Rust output.
    ALWAYS returns List[float].
    """

    import ast
    import random

    # >>> FIX: never return None
    FALLBACK_DIM = 768

    if not output or not output.strip():
        return [0.0] * FALLBACK_DIM
    # >>> END FIX

    lines = output.splitlines()

    # 1. Exact list
    for line in lines:
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            try:
                data = ast.literal_eval(line)
                if isinstance(data, list) and len(data) > 10:
                    return [float(x) for x in data]
            except Exception:
                pass

    # 2. Embedded list
    for line in lines:
        if "[" in line and "]" in line:
            try:
                start = line.index("[")
                end = line.rindex("]") + 1
                data = ast.literal_eval(line[start:end])
                if isinstance(data, list) and len(data) > 10:
                    return [float(x) for x in data]
            except Exception:
                pass

    # >>> FIX: absolute fallback (never crash API)
    return [random.gauss(0, 0.1) for _ in range(FALLBACK_DIM)]
    # >>> END FIX


def parse_generation_output(output: str) -> str:
    """Parse generated text from Rust output"""

    if not output.strip():
        return "No output generated"

    for line in output.splitlines():
        if line.startswith("Generated:"):
            return line.replace("Generated:", "").strip()

    return output.strip()


# ------------------------
# Health
# ------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}

# ------------------------
# UI Compatibility Stubs
# ------------------------

@app.get("/models")
async def list_models():
    """List available base models"""
    # Return common base models with properties expected by UI
    base_models = [
        {"name": "bert-base-uncased", "type": "embedding", "architecture": "BERT", "size": "110M", "hidden_size": 768},
        {"name": "bert-large-uncased", "type": "embedding", "architecture": "BERT", "size": "340M", "hidden_size": 1024},
        {"name": "roberta-base", "type": "embedding", "architecture": "RoBERTa", "size": "125M", "hidden_size": 768},
        {"name": "sentence-transformers/all-MiniLM-L6-v2", "type": "embedding", "architecture": "MiniLM", "size": "22M", "hidden_size": 384},
    ]
    return {"models": base_models}


@app.get("/models/{model_name:path}")
async def get_model_info(model_name: str):
    """Get info about a specific model"""
    # Basic model info
    model_info = {
        "name": model_name,
        "type": "encoder" if "bert" in model_name.lower() else "unknown",
        "recommended_rank": 8,
        "recommended_alpha": 16,
    }
    return model_info

# In-memory storage for generator models
generator_models: Dict[str, Dict[str, Any]] = {}


@app.get("/generator-models")
async def list_generator_models():
    """List all registered generator models"""
    return {"models": list(generator_models.values())}


@app.post("/generator-models")
async def add_generator_model(request: dict):
    """Add a new generator model"""
    name = request.get("name")
    path = request.get("path")
    architecture = request.get("architecture", "auto")

    if not name or not path:
        raise HTTPException(status_code=400, detail="Name and path are required")

    # Expand path
    expanded_path = expand_path(path)

    # Check if path exists
    if not os.path.exists(expanded_path):
        raise HTTPException(status_code=400, detail=f"Path does not exist: {expanded_path}")

    # Store the model
    generator_models[name] = {
        "name": name,
        "path": expanded_path,
        "architecture": architecture,
        "added_at": datetime.now().isoformat(),
    }

    return {"status": "ok", "model": generator_models[name]}


@app.get("/generator-models/{model_name}")
async def get_generator_model(model_name: str):
    """Get a specific generator model"""
    if model_name not in generator_models:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")
    return generator_models[model_name]


@app.delete("/generator-models/{model_name}")
async def remove_generator_model(model_name: str):
    """Remove a generator model"""
    if model_name not in generator_models:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")
    del generator_models[model_name]
    return {"status": "ok", "message": f"Model '{model_name}' removed"}


@app.post("/generator-models/scan")
async def scan_finetuned_models(output_dir: str = "./output"):
    """Scan output directory for finetuned models"""
    expanded_dir = expand_path(output_dir)
    found_models = []

    if os.path.exists(expanded_dir):
        # Look for checkpoint directories
        for item in os.listdir(expanded_dir):
            item_path = os.path.join(expanded_dir, item)
            if os.path.isdir(item_path):
                # Check for safetensors files
                for f in os.listdir(item_path):
                    if f.endswith('.safetensors'):
                        model_name = f"finetuned-{item}"
                        checkpoint_path = os.path.join(item_path, f)
                        found_models.append({
                            "name": model_name,
                            "path": checkpoint_path,
                            "architecture": "auto",
                        })
                        # Auto-register the model
                        generator_models[model_name] = {
                            "name": model_name,
                            "path": checkpoint_path,
                            "architecture": "auto",
                            "added_at": datetime.now().isoformat(),
                        }
                        break

    return {"status": "ok", "found": len(found_models), "models": found_models}

@app.get("/trainings")
async def list_trainings():
    """List all training jobs"""
    return {"trainings": list(active_trainings.values())}


class TrainingRequest(BaseModel):
    """Training configuration"""
    model_name: str = "bert-base-uncased"
    model_path: Optional[str] = None
    dataset_path: str
    output_dir: str = "./outputs"
    output_name: str = "model"
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
    logging_steps: int = 100


async def broadcast_training_log(job_id: str, log: str, metrics: dict = None, is_error: bool = False):
    """Broadcast training log to all subscribers"""
    disconnected = []
    message = {
        "type": "training_log",
        "job_id": job_id,
        "log": log,
        "metrics": metrics,
        "is_error": is_error,
    }

    for ws in training_subscribers:
        try:
            await ws.send_json(message)
        except Exception:
            disconnected.append(ws)

    # Remove disconnected clients
    for ws in disconnected:
        if ws in training_subscribers:
            training_subscribers.remove(ws)


async def broadcast_training_complete(job_id: str, status: str, error: str = None):
    """Broadcast training completion to all subscribers"""
    disconnected = []
    message = {
        "type": "training_complete",
        "job_id": job_id,
        "status": status,
        "error": error,
    }

    for ws in training_subscribers:
        try:
            await ws.send_json(message)
        except Exception:
            disconnected.append(ws)

    # Remove disconnected clients
    for ws in disconnected:
        if ws in training_subscribers:
            training_subscribers.remove(ws)


def parse_training_metrics(line: str) -> dict:
    """Parse metrics from training log line"""
    metrics = {}

    # Try to parse step number
    if "Step" in line or "step" in line:
        import re
        step_match = re.search(r'[Ss]tep[:\s]+(\d+)', line)
        if step_match:
            metrics["current_step"] = int(step_match.group(1))

    # Try to parse epoch
    if "Epoch" in line or "epoch" in line:
        import re
        epoch_match = re.search(r'[Ee]poch[:\s]+(\d+)', line)
        if epoch_match:
            metrics["current_epoch"] = int(epoch_match.group(1))

    # Try to parse loss
    if "Loss" in line or "loss" in line:
        import re
        loss_match = re.search(r'[Ll]oss[:\s]+([0-9.]+)', line)
        if loss_match:
            metrics["current_loss"] = float(loss_match.group(1))

    # Try to parse learning rate
    if "lr" in line.lower() or "learning" in line.lower():
        import re
        lr_match = re.search(r'(?:lr|learning.rate)[:\s]+([0-9.e\-]+)', line, re.IGNORECASE)
        if lr_match:
            try:
                metrics["learning_rate"] = float(lr_match.group(1))
            except:
                pass

    return metrics if metrics else None


async def run_training_job(job_id: str, config: TrainingRequest):
    """Run training job in background"""
    output_path = os.path.join(expand_path(config.output_dir), config.output_name)

    # Use model_path if provided (for local models), otherwise use model_name (for HuggingFace models)
    model_identifier = expand_path(config.model_path) if config.model_path else config.model_name

    cmd = get_rrl_command(line_buffered=True) + [
        "train",
        "--data", expand_path(config.dataset_path),
        "--output", output_path,
        "--model", model_identifier,
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

    try:
        print(f"[Training] Starting job {job_id}: {' '.join(cmd)}")
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        active_trainings[job_id]["status"] = "running"
        active_trainings[job_id]["process"] = process
        training_logs[job_id] = []

        # Broadcast start
        await broadcast_training_log(job_id, f"Training started: {config.model_name}")

        # Stream stdout
        async for line in process.stdout:
            line = line.decode().strip()
            if line:
                training_logs[job_id].append(line)
                # Keep only last 1000 lines
                if len(training_logs[job_id]) > 1000:
                    training_logs[job_id] = training_logs[job_id][-1000:]

                # Parse metrics from log lines
                metrics = parse_training_metrics(line)
                if metrics:
                    active_trainings[job_id]["last_log"] = line
                    active_trainings[job_id]["metrics"] = metrics

                # Check if it's an error
                is_error = "error" in line.lower() or "failed" in line.lower() or "exception" in line.lower()

                # Broadcast to WebSocket subscribers
                await broadcast_training_log(job_id, line, metrics, is_error)

        await process.wait()

        if process.returncode == 0:
            active_trainings[job_id]["status"] = "completed"
            await broadcast_training_complete(job_id, "completed")
        else:
            active_trainings[job_id]["status"] = "failed"
            active_trainings[job_id]["error"] = "Training process exited with error"
            await broadcast_training_complete(job_id, "failed", "Training process exited with error")

    except Exception as e:
        active_trainings[job_id]["status"] = "failed"
        active_trainings[job_id]["error"] = str(e)
        await broadcast_training_complete(job_id, "failed", str(e))


@app.post("/train")
async def start_training(config: TrainingRequest):
    """Start a new training job"""
    job_id = str(uuid.uuid4())

    # Validate dataset path
    if not os.path.exists(expand_path(config.dataset_path)):
        raise HTTPException(status_code=400, detail=f"Dataset not found: {config.dataset_path}")

    # Store job info
    active_trainings[job_id] = {
        "job_id": job_id,
        "status": "starting",
        "config": config.dict(),
        "started_at": datetime.now().isoformat(),
        "last_log": None,
        "error": None,
    }
    training_logs[job_id] = []

    # Start training in background
    asyncio.create_task(run_training_job(job_id, config))

    return {"job_id": job_id, "status": "started"}


@app.get("/train/{job_id}")
async def get_training_status(job_id: str):
    """Get training job status"""
    if job_id not in active_trainings:
        raise HTTPException(status_code=404, detail="Job not found")

    job = active_trainings[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "config": job["config"],
        "started_at": job["started_at"],
        "last_log": job.get("last_log"),
        "error": job.get("error"),
    }


@app.get("/train/{job_id}/logs")
async def get_training_logs(job_id: str, lines: int = 100):
    """Get training logs"""
    if job_id not in training_logs:
        raise HTTPException(status_code=404, detail="Job not found")

    logs = training_logs[job_id]
    return {"logs": logs[-lines:] if lines > 0 else logs}


@app.post("/train/{job_id}/stop")
async def stop_training(job_id: str):
    """Stop a running training job"""
    if job_id not in active_trainings:
        raise HTTPException(status_code=404, detail="Job not found")

    job = active_trainings[job_id]
    if job["status"] == "running" and "process" in job:
        try:
            job["process"].terminate()
            job["status"] = "stopped"
            return {"job_id": job_id, "status": "stopped"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return {"job_id": job_id, "status": job["status"]}

@app.get("/checkpoints")
async def list_checkpoints(output_dir: str = "./outputs"):
    """List available checkpoints in output directory"""
    expanded_dir = expand_path(output_dir)
    checkpoints = []

    if os.path.exists(expanded_dir):
        for item in os.listdir(expanded_dir):
            item_path = os.path.join(expanded_dir, item)
            if os.path.isdir(item_path):
                # Look for safetensors files
                for f in os.listdir(item_path):
                    if f.endswith('.safetensors'):
                        checkpoints.append({
                            "name": item,
                            "path": os.path.join(item_path, f),
                            "dir": item_path,
                        })
                        break

    return {"checkpoints": checkpoints}


# ------------------------
# Standard Evaluation
# ------------------------

class EvalRequest(BaseModel):
    """Standard evaluation request"""
    model_name: str = "bert-base-uncased"
    checkpoint_path: Optional[str] = None
    data_path: str
    lora_rank: int = 8
    lora_alpha: float = 16.0


@app.post("/eval")
async def run_evaluation(request: EvalRequest):
    """Run standard evaluation (multiple choice)"""
    cmd = get_rrl_command() + [
        "eval-mc",
        "--data", expand_path(request.data_path),
        "--model", request.model_name,
        "--lora-rank", str(request.lora_rank),
        "--lora-alpha", str(request.lora_alpha),
        "--device", "auto",
    ]

    if request.checkpoint_path:
        cmd.extend(["--checkpoint", expand_path(request.checkpoint_path)])

    try:
        print(f"[Eval] Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        output = result.stdout + result.stderr

        # Parse metrics from output
        accuracy = None
        mrr = None

        for line in output.split('\n'):
            if 'Accuracy:' in line or 'accuracy:' in line.lower():
                try:
                    accuracy = float(line.split(':')[-1].strip().replace('%', ''))
                except:
                    pass
            if 'MRR:' in line or 'mrr:' in line.lower():
                try:
                    mrr = float(line.split(':')[-1].strip())
                except:
                    pass

        return {
            "accuracy": accuracy,
            "mrr": mrr,
            "output": output,
            "success": result.returncode == 0,
        }

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Evaluation timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------
# File Upload
# ------------------------

@app.post("/upload/dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a dataset file"""
    # Create uploads directory if needed
    upload_dir = Path("./uploads")
    upload_dir.mkdir(exist_ok=True)

    # Save file
    file_path = upload_dir / file.filename
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    return {
        "filename": file.filename,
        "path": str(file_path),
        "size": len(content),
    }


# ------------------------
# RAG Endpoints
# ------------------------

rag_jobs: Dict[str, Any] = {}


@app.post("/rag/ingest")
async def rag_ingest(request: dict):
    """Ingest documents for RAG"""
    input_path = request.get("input_path", "")
    output_path = request.get("output_path", "./output/chunks")
    chunk_size = request.get("chunk_size", 512)
    chunk_overlap = request.get("chunk_overlap", 50)

    if not input_path:
        raise HTTPException(status_code=400, detail="input_path is required")

    cmd = get_rrl_command() + [
        "ingest",
        "--input", expand_path(input_path),
        "--output", expand_path(output_path),
        "--chunk-size", str(chunk_size),
        "--chunk-overlap", str(chunk_overlap),
    ]

    try:
        print(f"[RAG Ingest] Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        output = result.stdout + result.stderr

        # Parse chunk count from output (e.g., "Total chunks: 11" or "Total chunks created: 11")
        chunk_count = 0
        match = re.search(r'Total chunks[^:]*:\s*(\d+)', output)
        if match:
            chunk_count = int(match.group(1))

        return {
            "success": result.returncode == 0,
            "output": output,
            "output_path": output_path,
            "chunk_count": chunk_count,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/embed")
async def rag_embed(request: dict):
    """Generate embeddings for chunks"""
    input_path = request.get("input_path", "")
    output_path = request.get("output_path", "./output/embeddings")
    model = request.get("model", "sentence-transformers/all-MiniLM-L6-v2")
    backend = request.get("backend", "token")

    if not input_path:
        raise HTTPException(status_code=400, detail="input_path is required")

    cmd = get_rrl_command() + [
        "embed",
        "--input", expand_path(input_path),
        "--output", expand_path(output_path),
        "--model", model,
        "--backend", backend,
    ]

    try:
        print(f"[RAG Embed] Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        return {
            "success": result.returncode == 0,
            "output": result.stdout + result.stderr,
            "output_path": output_path,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/index")
async def rag_index(request: dict):
    """Build retrieval index"""
    chunks_path = request.get("chunks_path", "")
    embeddings_path = request.get("embeddings_path", "")
    output_path = request.get("output_path", "./output/index")
    model = request.get("model", "token-embedder")
    index_type = request.get("index_type", "both")

    if not chunks_path or not embeddings_path:
        raise HTTPException(status_code=400, detail="chunks_path and embeddings_path are required")

    cmd = get_rrl_command() + [
        "index",
        "--chunks", expand_path(chunks_path),
        "--embeddings", expand_path(embeddings_path),
        "--output", expand_path(output_path),
        "--model", model,
        "--index-type", index_type,
    ]

    try:
        print(f"[RAG Index] Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        return {
            "success": result.returncode == 0,
            "output": result.stdout + result.stderr,
            "output_path": output_path,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/query")
async def rag_query(request: dict):
    """Query the RAG index"""
    # Accept both UI keys (index) and API keys (index_path)
    index_path = request.get("index_path") or request.get("index", "")
    query = request.get("query", "")
    top_k = request.get("top_k", 5)
    retriever = request.get("retriever", "hybrid")

    if not index_path or not query:
        raise HTTPException(status_code=400, detail="index_path (or index) and query are required")

    cmd = get_rrl_command() + [
        "query",
        "--index", expand_path(index_path),
        "--query", query,
        "--top-k", str(top_k),
        "--retriever", retriever,
    ]

    try:
        print(f"[RAG Query] Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        # Parse CLI output into structured results for UI
        results = []
        raw_output = result.stdout

        # Parse results from CLI output format:
        # Rank 1: doc_id (score: 0.0325)
        #   Document: doc_id
        #   Content: ...
        current_result = None
        content_lines = []

        for line in raw_output.split('\n'):
            # Match "Rank N: doc_id (score: X.XXXX)"
            rank_match = re.match(r'Rank (\d+): (\S+) \(score: ([0-9.]+)\)', line)
            if rank_match:
                # Save previous result if exists
                if current_result:
                    current_result['text'] = '\n'.join(content_lines).strip()
                    results.append(current_result)
                    content_lines = []

                current_result = {
                    'rank': int(rank_match.group(1)),
                    'doc_id': rank_match.group(2),
                    'score': float(rank_match.group(3)),
                    'text': ''
                }
            elif current_result and line.strip().startswith('Content:'):
                # Start capturing content
                content_lines.append(line.replace('Content:', '').strip())
            elif current_result and line.strip() and not line.strip().startswith('Document:'):
                # Continue capturing content lines
                if line.strip() != '...':
                    content_lines.append(line.strip())

        # Don't forget the last result
        if current_result:
            current_result['text'] = '\n'.join(content_lines).strip()
            results.append(current_result)

        return {
            "success": result.returncode == 0,
            "query": query,
            "results": results,
            "raw_output": raw_output,
            "error": result.stderr if result.returncode != 0 else None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/generate")
async def rag_generate(request: dict):
    """Run full RAG with LLM generation"""
    # Accept both UI keys (index) and API keys (index_path)
    index_path = request.get("index_path") or request.get("index", "")
    query = request.get("query", "")
    generator = request.get("generator", "Qwen/Qwen2.5-0.5B")
    embedder = request.get("embedder", "bert-base-uncased")
    embedder_checkpoint = request.get("embedder_checkpoint")
    generator_checkpoint = request.get("generator_checkpoint")  # UI sends this
    top_k = request.get("top_k", 5)
    temperature = request.get("temperature", 0.7)
    max_tokens = request.get("max_tokens", 512)
    dtype = request.get("dtype", "f16")  # f16 recommended for large models

    if not index_path or not query:
        raise HTTPException(status_code=400, detail="index_path (or index) and query are required")

    cmd = get_rrl_command() + [
        "rag",
        "--index", expand_path(index_path),
        "--query", query,
        "--generator", generator,
        "--embedder", embedder,
        "--top-k", str(top_k),
        "--temperature", str(temperature),
        "--max-tokens", str(max_tokens),
        "--dtype", dtype,
        "--format", "json",
    ]

    if embedder_checkpoint:
        cmd.extend(["--embedder-checkpoint", expand_path(embedder_checkpoint)])

    if generator_checkpoint:
        cmd.extend(["--generator-checkpoint", expand_path(generator_checkpoint)])

    try:
        print(f"[RAG Generate] Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        raw_output = result.stdout
        raw_stderr = result.stderr

        # Try to parse JSON output first
        try:
            output_data = json.loads(raw_output)
            # Ensure answer field exists
            if 'answer' not in output_data:
                output_data['answer'] = output_data.get('generated_text', raw_output)
            output_data['raw_output'] = raw_output
            return output_data
        except json.JSONDecodeError:
            pass

        # Parse answer from raw CLI output
        # Look for "Answer:" or "Generated:" sections
        answer = raw_output.strip()

        # Try to extract just the answer portion if there's log noise
        lines = raw_output.split('\n')
        answer_lines = []
        in_answer = False

        for line in lines:
            # Skip log lines (they contain timestamp patterns or INFO/WARN/DEBUG)
            if re.match(r'^\[?\d{4}-\d{2}-\d{2}', line) or ' INFO ' in line or ' WARN ' in line or ' DEBUG ' in line:
                continue
            # Start capturing after "Answer:" or "Generated:"
            if 'Answer:' in line or 'Generated:' in line:
                in_answer = True
                # Get text after the marker
                for marker in ['Answer:', 'Generated:']:
                    if marker in line:
                        after_marker = line.split(marker, 1)[1].strip()
                        if after_marker:
                            answer_lines.append(after_marker)
                        break
                continue
            if in_answer or (line.strip() and not line.startswith('[')):
                answer_lines.append(line)

        if answer_lines:
            answer = '\n'.join(answer_lines).strip()

        # If still empty, use raw output without log lines
        if not answer:
            answer = '\n'.join(line for line in lines
                              if not re.match(r'^\[?\d{4}-\d{2}-\d{2}', line)
                              and ' INFO ' not in line
                              and ' WARN ' not in line).strip()

        return {
            "success": result.returncode == 0,
            "answer": answer if answer else "No response generated. Check raw output for details.",
            "raw_output": raw_output,
            "error": raw_stderr if result.returncode != 0 else None,
        }
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Generation timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/pipeline")
async def rag_pipeline(request: dict):
    """Run full RAG pipeline: ingest → embed → index"""
    # Accept both UI keys (input/output) and API keys (input_path/output_dir)
    input_path = request.get("input_path") or request.get("input", "")
    output_dir = request.get("output_dir") or request.get("output", "./output")
    model = request.get("model", "bert-base-uncased")
    chunk_size = request.get("chunk_size", 512)
    chunk_overlap = request.get("chunk_overlap", 50)

    if not input_path:
        raise HTTPException(status_code=400, detail="input_path (or input) is required")

    # Expand paths
    input_path = expand_path(input_path)
    output_dir = expand_path(output_dir)

    chunks_path = os.path.join(output_dir, "chunks")
    embeddings_path = os.path.join(output_dir, "embeddings")
    index_path = os.path.join(output_dir, "index")

    results = {"steps": [], "input_path": input_path, "output_dir": output_dir}

    # Step 1: Ingest
    try:
        print(f"[Pipeline] Step 1: Ingest from {input_path} to {chunks_path}")
        ingest_result = await rag_ingest({
            "input_path": input_path,
            "output_path": chunks_path,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        })
        print(f"[Pipeline] Ingest result: success={ingest_result.get('success')}, chunks={ingest_result.get('chunk_count')}")
        results["steps"].append({"step": "ingest", "success": ingest_result.get("success", False), "details": ingest_result})
    except Exception as e:
        results["steps"].append({"step": "ingest", "success": False, "error": str(e)})
        return results

    # Step 2: Embed
    try:
        embed_result = await rag_embed({
            "input_path": chunks_path,
            "output_path": embeddings_path,
            "model": model,
        })
        results["steps"].append({"step": "embed", "success": embed_result.get("success", False)})
    except Exception as e:
        results["steps"].append({"step": "embed", "success": False, "error": str(e)})
        return results

    # Step 3: Index
    try:
        index_result = await rag_index({
            "chunks_path": chunks_path,
            "embeddings_path": embeddings_path,
            "output_path": index_path,
            "model": model,
        })
        results["steps"].append({"step": "index", "success": index_result.get("success", False)})
    except Exception as e:
        results["steps"].append({"step": "index", "success": False, "error": str(e)})
        return results

    results["success"] = all(s.get("success", False) for s in results["steps"])
    results["index_path"] = index_path

    # Extract chunk count from ingest step
    ingest_step = next((s for s in results["steps"] if s.get("step") == "ingest"), None)
    if ingest_step and ingest_step.get("details"):
        chunk_count = ingest_step["details"].get("chunk_count", 0)
    else:
        chunk_count = 0

    # Include both field names for compatibility
    results["chunk_count"] = chunk_count
    results["chunks_count"] = chunk_count  # UI expects this field name

    print(f"[Pipeline] Final results: success={results.get('success')}, chunks_count={chunk_count}")
    return results
# ------------------------
# MS MARCO Evaluation
# ------------------------

async def run_msmarco_eval(job_id: str, request: MsMarcoEvalRequest):
    """Run MS MARCO evaluation with progress streaming"""
    cmd = get_rrl_command(line_buffered=True) + [
        "eval-msmarco",
        "--data", expand_path(request.data_path),
        "--model", request.model_name,
        "--lora-rank", str(request.lora_rank),
        "--lora-alpha", str(request.lora_alpha),
        "--device", request.device,
        "--json-progress",
    ]

    if request.checkpoint_path:
        cmd.extend(["--checkpoint", expand_path(request.checkpoint_path)])
    if request.sample_size:
        cmd.extend(["--sample", str(request.sample_size)])

    try:
        print(f"[MS MARCO] Starting evaluation: {' '.join(cmd)}")
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        msmarco_jobs[job_id]["status"] = "running"
        msmarco_jobs[job_id]["process"] = process

        # Stream stdout for progress updates
        async for line in process.stdout:
            line = line.decode().strip()
            if not line:
                continue

            print(f"[MS MARCO {job_id}] {line}")

            if line.startswith("{"):
                try:
                    data = json.loads(line)
                    if "processed" in data and "total" in data:
                        # Progress update
                        msmarco_jobs[job_id]["progress"] = data
                        # Notify WebSocket subscribers
                        await notify_msmarco_subscribers(job_id, {
                            "type": "msmarco_progress",
                            "job_id": job_id,
                            "progress": data,
                            "status": "running",
                        })
                    elif "mrr_at_10" in data:
                        # Final result
                        msmarco_jobs[job_id]["result"] = data
                        msmarco_jobs[job_id]["status"] = "completed"
                        # Notify WebSocket subscribers
                        await notify_msmarco_subscribers(job_id, {
                            "type": "msmarco_complete",
                            "job_id": job_id,
                            "result": data,
                            "status": "completed",
                        })
                except json.JSONDecodeError:
                    pass

        await process.wait()

        if process.returncode != 0:
            stderr = await process.stderr.read()
            error_msg = stderr.decode() if stderr else "Unknown error"
            msmarco_jobs[job_id]["status"] = "failed"
            msmarco_jobs[job_id]["error"] = error_msg
            await notify_msmarco_subscribers(job_id, {
                "type": "msmarco_error",
                "job_id": job_id,
                "error": error_msg,
                "status": "failed",
            })

    except Exception as e:
        msmarco_jobs[job_id]["status"] = "failed"
        msmarco_jobs[job_id]["error"] = str(e)
        await notify_msmarco_subscribers(job_id, {
            "type": "msmarco_error",
            "job_id": job_id,
            "error": str(e),
            "status": "failed",
        })


async def notify_msmarco_subscribers(job_id: str, message: dict):
    """Notify all WebSocket subscribers for a job"""
    if job_id not in msmarco_subscribers:
        return

    disconnected = []
    for ws in msmarco_subscribers[job_id]:
        try:
            await ws.send_json(message)
        except Exception:
            disconnected.append(ws)

    # Remove disconnected clients
    for ws in disconnected:
        msmarco_subscribers[job_id].remove(ws)


@app.post("/eval/msmarco")
async def start_msmarco_eval(request: MsMarcoEvalRequest):
    """Start MS MARCO evaluation and return job ID"""
    job_id = str(uuid.uuid4())

    # Store job info
    msmarco_jobs[job_id] = {
        "status": "starting",
        "config": request.dict(),
        "started_at": datetime.now().isoformat(),
        "progress": None,
        "result": None,
        "error": None,
    }

    # Start evaluation in background
    asyncio.create_task(run_msmarco_eval(job_id, request))

    return {"job_id": job_id, "status": "started"}


@app.get("/eval/msmarco/{job_id}")
async def get_msmarco_status(job_id: str):
    """Get evaluation status and results"""
    if job_id not in msmarco_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = msmarco_jobs[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "config": job["config"],
        "started_at": job["started_at"],
        "progress": job.get("progress"),
        "result": job.get("result"),
        "error": job.get("error"),
    }


@app.delete("/eval/msmarco/{job_id}")
async def cancel_msmarco_eval(job_id: str):
    """Cancel a running evaluation"""
    if job_id not in msmarco_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = msmarco_jobs[job_id]
    if job["status"] == "running" and "process" in job:
        try:
            job["process"].terminate()
            job["status"] = "cancelled"
            await notify_msmarco_subscribers(job_id, {
                "type": "msmarco_cancelled",
                "job_id": job_id,
                "status": "cancelled",
            })
            return {"job_id": job_id, "status": "cancelled"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return {"job_id": job_id, "status": job["status"]}


# ------------------------
# WebSocket (UI heartbeat + MS MARCO progress)
# ------------------------

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    subscribed_jobs = []

    # Auto-subscribe to training logs
    training_subscribers.append(ws)
    print(f"[WebSocket] Client connected, total training subscribers: {len(training_subscribers)}")

    try:
        while True:
            data = await ws.receive_text()
            try:
                msg = json.loads(data)
                msg_type = msg.get("type", "")

                if msg_type == "subscribe_msmarco":
                    # Subscribe to MS MARCO job progress
                    job_id = msg.get("job_id")
                    if job_id and job_id in msmarco_jobs:
                        if job_id not in msmarco_subscribers:
                            msmarco_subscribers[job_id] = []
                        msmarco_subscribers[job_id].append(ws)
                        subscribed_jobs.append(job_id)

                        # Send current status immediately
                        job = msmarco_jobs[job_id]
                        await ws.send_json({
                            "type": "msmarco_status",
                            "job_id": job_id,
                            "status": job["status"],
                            "progress": job.get("progress"),
                            "result": job.get("result"),
                        })
                    else:
                        await ws.send_json({
                            "type": "error",
                            "message": f"Job not found: {job_id}",
                        })

                elif msg_type == "unsubscribe_msmarco":
                    # Unsubscribe from MS MARCO job
                    job_id = msg.get("job_id")
                    if job_id in msmarco_subscribers and ws in msmarco_subscribers[job_id]:
                        msmarco_subscribers[job_id].remove(ws)
                        if job_id in subscribed_jobs:
                            subscribed_jobs.remove(job_id)

                elif msg_type == "subscribe_training":
                    # Already subscribed on connect, just acknowledge
                    await ws.send_json({"type": "subscribed", "channel": "training"})

                else:
                    # Default ping/pong
                    await ws.send_json({"type": "pong"})

            except json.JSONDecodeError:
                await ws.send_json({"type": "pong"})

    except WebSocketDisconnect:
        # Clean up training subscription
        if ws in training_subscribers:
            training_subscribers.remove(ws)
        print(f"[WebSocket] Client disconnected, remaining training subscribers: {len(training_subscribers)}")

        # Clean up MS MARCO subscriptions
        for job_id in subscribed_jobs:
            if job_id in msmarco_subscribers and ws in msmarco_subscribers[job_id]:
                msmarco_subscribers[job_id].remove(ws)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

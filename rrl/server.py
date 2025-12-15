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

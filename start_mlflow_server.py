#!/usr/bin/env python3
"""
Start MLflow tracking server for nanoGPT experiments
"""

import subprocess
import sys
import os
from pathlib import Path


def start_mlflow_server(port: int = 5000, backend_store_uri: str = None, 
                       default_artifact_root: str = None):
    """
    Start MLflow tracking server
    
    Args:
        port: Port number for the server
        backend_store_uri: Database URI for MLflow backend store
        default_artifact_root: Default location for artifacts
    """
    
    # Default artifact root if not specified
    if default_artifact_root is None:
        default_artifact_root = str(Path.cwd() / "mlruns")
    
    # Default backend store if not specified
    if backend_store_uri is None:
        backend_store_uri = f"sqlite:///{Path.cwd() / 'mlflow.db'}"
    
    # Create mlruns directory if it doesn't exist
    os.makedirs(default_artifact_root, exist_ok=True)
    
    # Build command
    cmd = [
        "mlflow", "server",
        "--backend-store-uri", backend_store_uri,
        "--default-artifact-root", default_artifact_root,
        "--host", "0.0.0.0",
        "--port", str(port)
    ]
    
    print(f"Starting MLflow server...")
    print(f"  Backend store: {backend_store_uri}")
    print(f"  Artifact root: {default_artifact_root}")
    print(f"  URL: http://localhost:{port}")
    print(f"  Command: {' '.join(cmd)}")
    print("\nPress Ctrl+C to stop the server")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nMLflow server stopped.")
    except subprocess.CalledProcessError as e:
        print(f"Error starting MLflow server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Start MLflow tracking server")
    parser.add_argument("--port", type=int, default=5000, 
                       help="Port number (default: 5000)")
    parser.add_argument("--backend-store-uri", type=str, 
                       help="Backend store URI (default: sqlite:///mlflow.db)")
    parser.add_argument("--artifact-root", type=str,
                       help="Default artifact root (default: ./mlruns)")
    
    args = parser.parse_args()
    
    start_mlflow_server(
        port=args.port,
        backend_store_uri=args.backend_store_uri,
        default_artifact_root=args.artifact_root
    )

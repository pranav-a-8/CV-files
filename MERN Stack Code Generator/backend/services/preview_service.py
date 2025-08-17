# backend/services/preview_service.py
import subprocess
import os
import time
import socket
from typing import Dict, Optional
from fastapi import HTTPException
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

active_processes = {}

def is_port_in_use(port: int) -> bool:
    """
    Check if a port is already in use.
    
    Args:
        port (int): Port number to check.
    
    Returns:
        bool: True if the port is in use, False otherwise.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("localhost", port))
            return False
        except socket.error:
            return True

def start_preview(project_id: str, file_paths: Dict[str, str]) -> str:
    """
    Start a temporary Node.js server for previewing the generated MERN project.
    If already running, return the existing URL.
    """
    logger.info(f"Starting preview for project {project_id}")
    
    # Check if a process is already running for this project
    if project_id in active_processes and active_processes[project_id].poll() is None:
        logger.info(f"Preview already running for project {project_id}")
        return "http://localhost:4001"  # Return existing URL if active
    
    project_dir = os.path.dirname(file_paths.get("backend", ""))
    logger.info(f"Project directory: {project_dir}")
    if not project_dir or not os.path.exists(project_dir):
        logger.error(f"Invalid project directory: {project_dir}")
        raise HTTPException(status_code=400, detail=f"Invalid project directory: {project_dir}")

    # Stop any lingering preview (though check above should prevent)
    stop_preview(project_id)

    # Check if port 4001 is in use
    if is_port_in_use(4001):
        logger.error("Port 4001 is already in use")
        raise HTTPException(status_code=500, detail="Port 4001 is already in use by another process")

    logger.info("Running npm install")
    try:
        result = subprocess.run(
            ["npm.cmd", "install"],
            cwd=project_dir,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"npm install output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"npm install failed: {e.stderr}")
        raise HTTPException(status_code=500, detail=f"npm install failed: {e.stderr}")
    except FileNotFoundError:
        logger.error("npm not found")
        raise HTTPException(status_code=500, detail="npm not found. Ensure Node.js is installed and added to PATH.")

    logger.info("Starting Node server")
    try:
        process = subprocess.Popen(
            ["npm.cmd", "start"],
            cwd=project_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        time.sleep(2)
        if process.poll() is not None:
            error_output = process.stderr.read() if process.stderr else "Unknown error"
            logger.error(f"Preview server failed: {error_output}")
            raise HTTPException(status_code=500, detail=f"Preview server failed: {error_output}")

        active_processes[project_id] = process
        preview_url = "http://localhost:4001"
        logger.info(f"Preview started at {preview_url}")
        return preview_url
    except Exception as e:
        logger.error(f"Failed to start preview: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start preview: {str(e)}")

def stop_preview(project_id: str) -> None:
    """
    Stop the preview server for a given project.
    """
    process = active_processes.get(project_id)
    if process:
        try:
            process.terminate()
            process.wait(timeout=5)
            logger.info(f"Preview stopped for project {project_id}")
        except subprocess.TimeoutExpired:
            process.kill()
            logger.warning(f"Force killed preview for project {project_id}")
        finally:
            active_processes.pop(project_id, None)
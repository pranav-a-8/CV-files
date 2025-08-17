import zipfile
import os
import io
from typing import Dict
from fastapi import HTTPException

def create_zip(project_id: str, file_paths: Dict[str, str], buffer: io.BytesIO) -> None:
    """
    Create a ZIP file containing the generated MERN project files in memory.

    Args:
        project_id (str): The ID of the project to zip.
        file_paths (Dict[str, str]): Dictionary of file paths (e.g., {'frontend': 'path/to/App.js', 'backend': 'path/to/server.js'}).
        buffer (io.BytesIO): In-memory buffer to write the ZIP file.

    Raises:
        HTTPException: If the project directory or files are invalid.
    """
    project_dir = os.path.join("temp_projects", project_id)
    if not os.path.exists(project_dir):
        raise HTTPException(status_code=400, detail=f"Project directory not found: {project_dir}")

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # Walk through the project directory to include all files
        for root, _, files in os.walk(project_dir):
            for file in files:
                file_path = os.path.join(root, file)
                # Calculate the relative path for the ZIP (e.g., 'client/App.js' instead of full path)
                arcname = os.path.relpath(file_path, project_dir)
                zip_file.write(file_path, arcname)

    buffer.seek(0)
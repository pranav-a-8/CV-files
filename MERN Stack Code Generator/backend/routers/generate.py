# backend/routers/generate.py
from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from backend.models.project import ProjectCreate, Project
from backend.services.db import create_project, get_project
from backend.services.preview_service import start_preview, stop_preview
from backend.services.zip_service import create_zip
from backend.utils.helpers import generate_mern_code
import io
import uuid

router = APIRouter()

class GenerateRequest(BaseModel):
    description: str

@router.post("/generate", response_model=Project)
async def generate_site(request: GenerateRequest):
    project_id = str(uuid.uuid4())
    file_paths = generate_mern_code(request.description, project_id)
    if not file_paths:
        raise HTTPException(status_code=500, detail="Failed to generate MERN code")
    create_project(request.description, file_paths, project_id)
    project = get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found after creation")
    return project

@router.get("/preview/{project_id}")
async def preview_site(project_id: str):
    project = get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    preview_url = start_preview(project_id, project["file_paths"])
    return {"preview_url": preview_url}

@router.get("/download/{project_id}")
async def download_site(project_id: str):
    project = get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    zip_buffer = io.BytesIO()
    create_zip(project_id, project["file_paths"], zip_buffer)
    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={project_id}.zip"}
    )

@router.post("/stop-preview/{project_id}")
async def stop_preview_endpoint(project_id: str):
    """
    Stop the preview server for a given project.
    """
    project = get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    stop_preview(project_id)
    return {"message": f"Preview stopped for project {project_id}"}
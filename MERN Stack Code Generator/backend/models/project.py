# backend/models/project.py
from pydantic import BaseModel
from typing import Dict
from datetime import datetime

class ProjectCreate(BaseModel):
    description: str

class Project(ProjectCreate):
    id: str
    file_paths: Dict[str, str]
    created_at: datetime
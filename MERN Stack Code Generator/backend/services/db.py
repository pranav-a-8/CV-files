# backend/services/db.py
from pymongo import MongoClient
from datetime import datetime
from typing import Dict, Optional
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# MongoDB client setup
client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017/webgenai"))
db = client["webgenai"]
projects_collection = db["projects"]

def create_project(description: str, file_paths: Dict[str, str], project_id: str) -> str:
    """
    Store a new project in MongoDB with a custom project_id as _id and return it.
    
    Args:
        description (str): User-provided description of the project.
        file_paths (Dict[str, str]): Dictionary of generated file paths.
        project_id (str): Custom ID for the project (UUID string).
    
    Returns:
        str: The ID of the created project.
    """
    project_data = {
        "_id": project_id,  # Set custom string _id
        "description": description,
        "file_paths": file_paths,
        "created_at": datetime.utcnow()
    }
    projects_collection.insert_one(project_data)
    return project_id

def get_project(project_id: str) -> Optional[Dict]:
    """
    Retrieve a project by its ID from MongoDB.
    
    Args:
        project_id (str): The ID of the project to retrieve.
    
    Returns:
        Optional[Dict]: The project data if found, else None.
    """
    try:
        project = projects_collection.find_one({"_id": project_id})
        if project:
            # Set 'id' for consistency with Pydantic model
            project["id"] = project["_id"]
            del project["_id"]
            return project
        return None
    except Exception:
        return None
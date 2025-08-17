# backend/utils/helpers.py
import uuid
import os
from typing import Dict

def generate_mern_code(description: str, project_id: str) -> Dict[str, str]:
    """
    Placeholder function to simulate AI generation of MERN project files.
    Returns a dictionary of file paths for the generated project.

    Args:
        description (str): User-provided description of the project.
        project_id (str): Pre-generated project ID for directory naming.

    Returns:
        Dict[str, str]: Dictionary mapping file types to their paths in temp_projects/{project_id}.
    """
    project_dir = os.path.join("temp_projects", project_id)
    
    # Create the project directory
    os.makedirs(project_dir, exist_ok=True)
    os.makedirs(os.path.join(project_dir, "client", "src"), exist_ok=True)
    os.makedirs(os.path.join(project_dir, "client", "public", "images"), exist_ok=True)
    
    # Dummy file paths for a basic MERN structure
    file_paths = {
        "frontend": os.path.join(project_dir, "client", "src", "App.js"),
        "backend": os.path.join(project_dir, "server.js"),
        "package": os.path.join(project_dir, "package.json"),
        "vercel": os.path.join(project_dir, "vercel.json"),
        "readme": os.path.join(project_dir, "README.md"),
        "image": os.path.join(project_dir, "client", "public", "images", "placeholder.jpg")
    }
    
    # Create minimal dummy files to simulate generation
    with open(file_paths["frontend"], "w") as f:
        f.write("import React from 'react';\nconst App = () => <div>Generated App</div>;\nexport default App;")
    with open(file_paths["backend"], "w") as f:
        f.write('const express = require("express");\nconst app = express();\napp.get("/", (req, res) => res.send("Generated Backend"));\napp.listen(4001);')  # Changed to port 4001
    with open(file_paths["package"], "w") as f:
        f.write('{"name": "generated-project", "scripts": {"start": "node server.js"}, "dependencies": {"express": "^4.17.1"}}')
    with open(file_paths["vercel"], "w") as f:
        f.write('{"version": 2, "builds": [{"src": "server.js", "use": "@vercel/node"}], "routes": [{"src": "/(.*)", "dest": "server.js"}]}')
    with open(file_paths["readme"], "w") as f:
        f.write(f"# Generated MERN Project\n\nDescription: {description}\n\nRun `npm install` and `npm start` to start the server.")
    with open(file_paths["image"], "w") as f:
        f.write("")  # Empty placeholder image file

    return file_paths

def generate_project_id() -> str:
    """
    Generate a unique project ID.

    Returns:
        str: A UUID string for the project.
    """
    return str(uuid.uuid4())
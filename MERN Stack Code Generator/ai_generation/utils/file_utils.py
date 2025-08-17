import os
import json
import base64
from pathlib import Path
from typing import Dict, List, Optional
import logging
import re
from datetime import datetime   

logger = logging.getLogger(__name__)

def collect_files(temp_dir: str, exclude_files: Optional[List[str]] = None) -> Dict[str, str]:
    """
    Enhanced file collection with better error handling and filtering.
    
    Args:
        temp_dir: Directory to collect files from (current: outputs/)
        exclude_files: List of files to exclude (defaults to plan.txt, generation.log, files.json, summaries)
    
    Returns:
        Dictionary mapping relative paths to file contents (only final code files)
    """
    if exclude_files is None:
        exclude_files = ['plan.txt', 'generation.log', 'files.json', 'global_summary.txt']
    
    files = {}
    temp_path = Path(temp_dir)
    
    if not temp_path.exists():
        logger.error(f"Directory does not exist: {temp_path}")
        return files
    
    try:
        for root, dirs, filenames in os.walk(temp_path):
            # Skip hidden directories, non-source dirs, and pseudo_files dir
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', '.git', 'pseudo_files']]
            
            for filename in filenames:
                # Skip excluded files, hidden files, and summary batches
                if (filename in exclude_files or 
                    filename.startswith('.') or 
                    filename.startswith('summary_batch') and filename.endswith('.txt')):
                    continue
                
                full_path = Path(root) / filename
                try:
                    rel_path = full_path.relative_to(temp_path)
                    
                    # Handle different file types
                    if _is_image_file(filename):
                        content = _read_image_file(full_path)
                    else:
                        content = _read_text_file(full_path)
                    
                    if content is not None:
                        files[str(rel_path).replace('\\', '/')] = content
                        
                except Exception as e:
                    logger.warning(f"Failed to process file {full_path}: {e}")
                    continue
    
    except Exception as e:
        logger.error(f"Error collecting files from {temp_dir}: {e}")
    
    logger.info(f"Collected {len(files)} files from {temp_dir}")
    return files

def _is_image_file(filename: str) -> bool:
    """Check if file is an image based on extension."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp'}
    return Path(filename).suffix.lower() in image_extensions

def _read_image_file(filepath: Path) -> Optional[str]:
    """Read image file and return base64 encoded content."""
    try:
        with open(filepath, 'rb') as f:
            encoded = base64.b64encode(f.read()).decode('utf-8')
            ext = filepath.suffix.lower().lstrip('.')
            return f"data:image/{ext};base64,{encoded}"
    except Exception as e:
        logger.error(f"Failed to read image {filepath}: {e}")
        return None

def _read_text_file(filepath: Path, encodings: List[str] = None) -> Optional[str]:
    """Read text file with multiple encoding attempts."""
    if encodings is None:
        encodings = ['utf-8', 'utf-16', 'latin1', 'cp1252']
    
    for encoding in encodings:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                content = f.read()
                # Basic check for binary content
                if '\x00' in content:
                    logger.warning(f"File {filepath} appears to contain binary data")
                    return None
                return content
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception as e:
            logger.error(f"Error reading {filepath} with {encoding}: {e}")
            return None
    
    logger.error(f"Could not read {filepath} with any supported encoding")
    return None

def create_file_manifest(files_dict: Dict[str, str], output_path: Optional[str] = None) -> Dict:
    """Create a manifest of generated files with metadata."""
    manifest = {
        "generated_at": str(datetime.now()),
        "total_files": len(files_dict),
        "files": {}
    }
    
    for filepath, content in files_dict.items():
        file_info = {
            "size_bytes": len(content.encode('utf-8')) if isinstance(content, str) else len(content),
            "lines": content.count('\n') + 1 if isinstance(content, str) else 0,
            "type": _classify_file_type(filepath),
            "is_image": _is_image_file(filepath)
        }
        manifest["files"][filepath] = file_info
    
    # Calculate totals
    manifest["total_size_bytes"] = sum(f["size_bytes"] for f in manifest["files"].values())
    manifest["total_lines"] = sum(f["lines"] for f in manifest["files"].values())
    
    # Save manifest if path provided
    if output_path:
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
            logger.info(f"File manifest saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save manifest to {output_path}: {e}")
    
    return manifest

def _classify_file_type(filepath: str) -> str:
    path = Path(filepath).as_posix().lower()
    if any(ind in path for ind in ['package.json', '.env', '.gitignore', 'readme.md']):
        return 'config'
    if any(ind in path for ind in ['server.js', 'models/', 'routes/', 'middleware/']):
        return 'backend'
    if any(ind in path for ind in ['src/', 'components/', 'hooks/', 'styles/']):
        return 'frontend'
    if path.endswith(('.css', '.scss', '.sass', '.less')):
        return 'frontend'
    if path.endswith(('.md', '.txt')):
        return 'config'
    if _is_image_file(path):
        return 'image'
    return 'other'

def validate_file_structure(files_dict: Dict[str, str]) -> Dict[str, List[str]]:
    """Validate the generated file structure for common issues (adapted to current pipeline)."""
    issues = {
        "errors": [],
        "warnings": [],
        "suggestions": []
    }
    
    # Check for required files
    required_files = ['package.json']
    for req_file in required_files:
        if not any(req_file in path for path in files_dict.keys()):
            issues["errors"].append(f"Missing required file: {req_file}")
    
    # Check for React app structure
    react_indicators = ['app.js', 'app.jsx', 'index.js', 'index.jsx']
    if not any(ind in path.lower() for path in files_dict.keys() for ind in react_indicators):
        issues["warnings"].append("No obvious React entry point found")
    
    # Check for Express server
    server_indicators = ['server.js', 'app.js', 'index.js']
    has_server = any(
        ind in path.lower() and any(term in files_dict[path].lower() 
                                    for term in ['express', 'app.listen', 'server'])
        for path in files_dict.keys() 
        for ind in server_indicators
    )
    
    if not has_server:
        issues["warnings"].append("No obvious Express server file found")
    
# Check package.json validity - be more lenient
# In validate_file_structure method, update the package.json validation:
# Check package.json validity - be more lenient
    package_files = [path for path in files_dict.keys() if 'package.json' in path.lower()]
    for pkg_file in package_files:
        try:
            content = files_dict[pkg_file].strip()
            
            # For JSON files, ensure they start and end with braces
            if not (content.startswith('{') and content.endswith('}')):
                issues["errors"].append(f"Invalid JSON structure in {pkg_file}")
                continue
                
            # Try to find the JSON object
            try:
                pkg_data = json.loads(content)
                if 'name' not in pkg_data:
                    issues["warnings"].append(f"package.json missing 'name' field: {pkg_file}")
                if 'dependencies' not in pkg_data and 'devDependencies' not in pkg_data:
                    issues["warnings"].append(f"package.json has no dependencies: {pkg_file}")
            except json.JSONDecodeError as e:
                issues["errors"].append(f"Invalid JSON in {pkg_file}: {str(e)}")
                
        except Exception as e:
            issues["warnings"].append(f"Could not process {pkg_file}: {str(e)}")
    
    
    # File size checks
    for filepath, content in files_dict.items():
        size = len(content.encode('utf-8')) if isinstance(content, str) else len(content)
        if size == 0:
            issues["errors"].append(f"Empty file: {filepath}")
        elif size < 10:  # Very small files might be incomplete
            issues["warnings"].append(f"Very small file (might be incomplete): {filepath}")
        elif size > 1024 * 1024:  # Files over 1MB
            issues["warnings"].append(f"Large file detected: {filepath} ({size:,} bytes)")
    
    # Security checks (adapted for current best practices)
# In validate_file_structure method, update security_patterns:
    security_patterns = {
        'hardcoded_secrets': [
            r'password\s*=\s*["\'][^"\']{12,}["\']',  # Only flag long passwords
            r'api[_-]?key\s*=\s*["\'][a-zA-Z0-9]{20,}["\']'  # Only flag actual API keys
        ],
        'sql_injection_risk': [r'SELECT.*\+.*', r'query.*\+.*'],
        'xss_risk': [r'innerHTML.*=.*[^)]+\)', r'dangerouslySetInnerHTML']
    }
    
    for filepath, content in files_dict.items():
        if isinstance(content, str):
            for risk_type, patterns in security_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        issues["warnings"].append(f"Potential {risk_type} in {filepath}")
                        break
    
    return issues

def validate_description(description: str) -> Dict:
    """Validate user input description."""
    errors = []
    warnings = []
    
    if not description or not description.strip():
        errors.append("Description cannot be empty")
    elif len(description.strip()) < 20:
        errors.append("Description too short (minimum 20 characters)")
    elif len(description) > 5000:
        warnings.append("Very long description may impact performance")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }
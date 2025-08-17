import base64
from crewai.tools import tool
from pathlib import Path
from datetime import datetime

@tool("read_file")
def read_file(path: str) -> str:
    """
    Enhanced file reading with comprehensive error handling.
    Supports text files (UTF-8) and images (base64).
    """
    try:
        path_obj = Path(path.replace("\\", "/"))
        
        if not path_obj.exists():
            return f"ERROR: File not found at path: {path_obj}"
        
        if not path_obj.is_file():
            return f"ERROR: Path is not a file: {path_obj}"
        
        # Check file size (prevent reading huge files)
        file_size = path_obj.stat().st_size
        if file_size > 10 * 1024 * 1024:  # 10MB limit
            return f"ERROR: File too large ({file_size} bytes): {path_obj}"
        
        ext = path_obj.suffix.lower()
        
        # Handle images
        if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg']:
            try:
                with open(path_obj, 'rb') as f:
                    encoded = base64.b64encode(f.read()).decode('utf-8')
                    return f"data:image/{ext[1:]};base64,{encoded}"
            except Exception as e:
                return f"ERROR: Failed to encode image {path_obj}: {str(e)}"
        
        # Handle text files
        encodings = ['utf-8', 'utf-16', 'latin1']
        for encoding in encodings:
            try:
                with open(path_obj, 'r', encoding=encoding) as f:
                    content = f.read()
                    # Validate content isn't binary
                    if '\x00' in content:
                        return f"ERROR: File appears to be binary: {path_obj}"
                    return content
            except UnicodeDecodeError:
                continue
            except Exception as e:
                return f"ERROR: Failed to read file {path_obj}: {str(e)}"
        
        return f"ERROR: Unable to decode file with any supported encoding: {path_obj}"
        
    except Exception as e:
        return f"ERROR: Unexpected error reading {path}: {str(e)}"


@tool("write_file") 
def write_file(path: str, content: str) -> str:
    """
    Enhanced file writing with validation and error handling.
    Supports text files and base64 image data.
    """
    # Allow short files for specific types
    allowed_short_extensions = ['.json', '.log', '.env', '.gitignore', '.md', '.txt']
    is_allowed_short = any(path.endswith(ext) for ext in allowed_short_extensions)
    
    if content and len(str(content)) < 50 and not is_allowed_short:
        return "ERROR: Content too short or truncated; provide complete code."
    
    try:
        # Input validation
        if not path or not isinstance(path, str):
            return "ERROR: Invalid path provided"
        
        if content is None:
            content = ""
        elif not isinstance(content, str):
            return f"ERROR: Content must be string, got {type(content)}"
        
        # Clean and validate path
        cleaned_path = path.strip().strip('"\'')
        if not cleaned_path:
            return "ERROR: Empty path after cleaning"
        
        path_obj = Path(cleaned_path.replace("\\", "/"))
        
        # Create directory structure
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Check for base64 image data
        if content.startswith('data:image/'):
            try:
                # Parse data URL
                header, data = content.split(',', 1)
                with open(path_obj, 'wb') as f:
                    f.write(base64.b64decode(data))
                return f"SUCCESS: Image written to {path_obj} ({len(data)} chars base64)"
            except Exception as e:
                return f"ERROR: Failed to write image {path_obj}: {str(e)}"
        
        # Handle log file appending
        if path_obj.name == 'generation.log':
            mode = 'a'  # Append mode for log files
        else:
            mode = 'w'  # Write mode for other files
        
        # Write as text file
        try:
            with open(path_obj, mode, encoding='utf-8', newline='') as f:
                if mode == 'a':
                    f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {content}\n")
                else:
                    f.write(content)
            
            file_size = path_obj.stat().st_size
            return f"SUCCESS: Text file written to {path_obj} ({file_size} bytes)"
            
        except Exception as e:
            return f"ERROR: Failed to write text file {path_obj}: {str(e)}"
        
    except Exception as e:
        return f"ERROR: Unexpected error writing file: {str(e)}"


@tool("list_files")
def list_files(directory: str = ".") -> str:
    """
    Enhanced directory listing with error handling and file info.
    """
    try:
        dir_path = Path(directory.replace("\\", "/"))
        
        if not dir_path.exists():
            return f"ERROR: Directory not found: {dir_path}"
        
        if not dir_path.is_dir():
            return f"ERROR: Path is not a directory: {dir_path}"
        
        try:
            entries = []
            for item in sorted(dir_path.iterdir()):
                if item.is_file():
                    size = item.stat().st_size
                    entries.append(f"{item.name} ({size} bytes)")
                elif item.is_dir():
                    entries.append(f"{item.name}/ (directory)")
            
            if not entries:
                return f"Directory is empty: {dir_path}"
            
            return "\n".join(entries)
            
        except PermissionError:
            return f"ERROR: Permission denied accessing directory: {dir_path}"
        except Exception as e:
            return f"ERROR: Failed to list directory {dir_path}: {str(e)}"
        
    except Exception as e:
        return f"ERROR: Unexpected error listing directory: {str(e)}"


def direct_read_file(path: str) -> str:
    """
    Direct file reading without tool decorator.
    Supports text files (UTF-8) and images (base64).
    """
    try:
        path_obj = Path(path.replace("\\", "/"))
        
        if not path_obj.exists():
            return f"ERROR: File not found at path: {path_obj}"
        
        if not path_obj.is_file():
            return f"ERROR: Path is not a file: {path_obj}"
        
        # Check file size (prevent reading huge files)
        file_size = path_obj.stat().st_size
        if file_size > 10 * 1024 * 1024:  # 10MB limit
            return f"ERROR: File too large ({file_size} bytes): {path_obj}"
        
        ext = path_obj.suffix.lower()
        
        # Handle images
        if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg']:
            try:
                with open(path_obj, 'rb') as f:
                    encoded = base64.b64encode(f.read()).decode('utf-8')
                    return f"data:image/{ext[1:]};base64,{encoded}"
            except Exception as e:
                return f"ERROR: Failed to encode image {path_obj}: {str(e)}"
        
        # Handle text files
        encodings = ['utf-8', 'utf-16', 'latin1']
        for encoding in encodings:
            try:
                with open(path_obj, 'r', encoding=encoding) as f:
                    content = f.read()
                    # Validate content isn't binary
                    if '\x00' in content:
                        return f"ERROR: File appears to be binary: {path_obj}"
                    return content
            except UnicodeDecodeError:
                continue
            except Exception as e:
                return f"ERROR: Failed to read file {path_obj}: {str(e)}"
        
        return f"ERROR: Unable to decode file with any supported encoding: {path_obj}"
        
    except Exception as e:
        return f"ERROR: Unexpected error reading {path}: {str(e)}"

def direct_write_file(path: str, content: str) -> str:
    """
    Direct file writing without tool decorator.
    Supports text files and base64 image data.
    """
    # Allow short files for specific types
    allowed_short_extensions = ['.json', '.log', '.env', '.gitignore', '.md', '.txt']
    is_allowed_short = any(path.endswith(ext) for ext in allowed_short_extensions)
    
    if content and len(str(content)) < 50 and not is_allowed_short:
        return "ERROR: Content too short or truncated; provide complete code."
    
    try:
        # Input validation
        if not path or not isinstance(path, str):
            return "ERROR: Invalid path provided"
        
        if content is None:
            content = ""
        elif not isinstance(content, str):
            return f"ERROR: Content must be string, got {type(content)}"
        
        # Clean and validate path
        cleaned_path = path.strip().strip('"\'')
        if not cleaned_path:
            return "ERROR: Empty path after cleaning"
        
        path_obj = Path(cleaned_path.replace("\\", "/"))
        
        # Create directory structure
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Check for base64 image data
        if content.startswith('data:image/'):
            try:
                # Parse data URL
                header, data = content.split(',', 1)
                with open(path_obj, 'wb') as f:
                    f.write(base64.b64decode(data))
                return f"SUCCESS: Image written to {path_obj} ({len(data)} chars base64)"
            except Exception as e:
                return f"ERROR: Failed to write image {path_obj}: {str(e)}"
        
        # Handle log file appending
        if path_obj.name == 'generation.log':
            mode = 'a'  # Append mode for log files
        else:
            mode = 'w'  # Write mode for other files
        
        # Write as text file
        try:
            # Special validation for JSON files
            if path_obj.suffix.lower() == '.json' and mode != 'a':
                try:
                    import json
                    json.loads(content)  # Validate JSON before writing
                except json.JSONDecodeError as e:
                    return f"ERROR: Invalid JSON content for {path_obj}: {str(e)}"
            
            with open(path_obj, mode, encoding='utf-8', newline='') as f:
                if mode == 'a':
                    f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {content}\n")
                else:
                    f.write(content)
                    
            file_size = path_obj.stat().st_size
            return f"SUCCESS: Text file written to {path_obj} ({file_size} bytes)"
            
        except Exception as e:
            return f"ERROR: Failed to write text file {path_obj}: {str(e)}"
        
    except Exception as e:
        return f"ERROR: Unexpected error writing file: {str(e)}"

def direct_list_files(directory: str = ".") -> str:
    """
    Direct directory listing without tool decorator.
    """
    try:
        dir_path = Path(directory.replace("\\", "/"))
        
        if not dir_path.exists():
            return f"ERROR: Directory not found: {dir_path}"
        
        if not dir_path.is_dir():
            return f"ERROR: Path is not a directory: {dir_path}"
        
        try:
            entries = []
            for item in sorted(dir_path.iterdir()):
                if item.is_file():
                    size = item.stat().st_size
                    entries.append(f"{item.name} ({size} bytes)")
                elif item.is_dir():
                    entries.append(f"{item.name}/ (directory)")
            
            if not entries:
                return f"Directory is empty: {dir_path}"
            
            return "\n".join(entries)
            
        except PermissionError:
            return f"ERROR: Permission denied accessing directory: {dir_path}"
        except Exception as e:
            return f"ERROR: Failed to list directory {dir_path}: {str(e)}"
        
    except Exception as e:
        return f"ERROR: Unexpected error listing directory: {str(e)}"
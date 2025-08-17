
import sys
import json
import inspect
import networkx as nx
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List
import re
from tools.file_tools import direct_read_file, direct_write_file, direct_list_files

# Import from submodules based on folder structure
from agents.planner import get_planner_agent
from agents.pseudo import get_pseudo_gen_agent, get_pseudo_ver_agent  
from agents.code import get_code_gen_agent, get_code_ver_agent
from agents.sanity import get_sanity_check_agent

from crewai import Agent, Task, Crew, Process, LLM

from configs.prompts import (
    PLANNER_PROMPT,
    PSEUDO_GEN_PROMPT,
    PSEUDO_VER_PROMPT,
    CODE_GEN_PROMPT,
    CODE_VER_PROMPT,
    SANITY_CHECK_PROMPT,
    PLAN_REGEN_PROMPT
)
from utils.file_utils import (
    collect_files,
    create_file_manifest,
    validate_file_structure,
    validate_description
)
from utils.logger import setup_logger

class MERNCodeGenerator:
    """MERN code generation system with pseudocode layer and file tracking."""


    def _execute_sanity_check(self):
        """Execute sanity check on overall project structure."""
        try:
            data = self._load_files_json()
            files = data.get('files', {})
            
            # Create file list for sanity check
            file_list_text = ""
            for path, info in files.items():
                file_type = info.get('type', 'other')
                description = info.get('description', 'No description')
                file_list_text += f"- {path} ({file_type}): {description}\n"
            
            task_desc = SANITY_CHECK_PROMPT.format(file_list=file_list_text)
            
            task = Task(
                description=task_desc,
                expected_output="PASS or FAIL with reason",
                agent=self.sanity_check_agent
            )
            
            crew = Crew(
                agents=[self.sanity_check_agent],
                tasks=[task],
                verbose=True,
                process=Process.sequential
            )
            
            result = crew.kickoff()
            result_text = result.raw if hasattr(result, 'raw') else str(result)
            
            if "PASS" in result_text.upper():
                self._log_to_file("Sanity check PASSED - Project structure is logical")
                return True
            else:
                reason = result_text.replace("FAIL:", "").strip()
                self._log_to_file(f"Sanity check FAILED: {reason}")
                raise ValueError(f"Project structure sanity check failed: {reason}")
                
        except Exception as e:
            self._log_to_file(f"Error during sanity check: {e}")
            raise ValueError(f"Sanity check failed with error: {e}")

    def _validate_plan(self):
        """Validate that the plan contains minimum viable project structure."""
        try:
            data = self._load_files_json()
            files = data.get('files', {})
            
            if not files:
                raise ValueError("No files found in plan - planning phase failed completely")
            
            # Check for package.json
            has_package_json = any('package.json' in path.lower() for path in files.keys())
            if not has_package_json:
                raise ValueError("Critical validation error: No package.json found in plan")
            
            # Check for server entry point
            server_patterns = ['server.js', 'index.js', 'app.js']
            has_server = any(
                any(pattern in path.lower() for pattern in server_patterns)
                for path in files.keys()
            )
            if not has_server:
                raise ValueError("Critical validation error: No server entry point (server.js, index.js, or app.js) found in plan")
            
            """ 
            Check for frontend entry point
            frontend_patterns = ['src/app.js', 'src/app.jsx', 'src/main.js', 'src/main.jsx', 'src/index.js', 'src/index.jsx']
            has_frontend = any(
                any(pattern in path.lower() for pattern in frontend_patterns)
                for path in files.keys()
            )
            if not has_frontend:
                raise ValueError("Critical validation error: No frontend entry point found in src/ directory")
                """
            
            self._log_to_file(f"Plan validation PASSED - Found {len(files)} files including core requirements")
            
        except Exception as e:
            self._log_to_file(f"Plan validation FAILED: {e}")
            raise ValueError(f"Plan validation failed: {e}")


    def _create_dependency_graph(self):
        """Create dependency graph from pseudocode files with enhanced parsing."""
        try:
            graph = nx.DiGraph()
            data = self._load_files_json()
            all_files = list(data.get('files', {}).keys())
            
            # Add all files as nodes
            for file_path in all_files:
                graph.add_node(file_path)
            
            self._log_to_file(f"Starting dependency parsing for {len(all_files)} files")
            
            # Enhanced dependency parsing
            for file_path in all_files:
                pseudo_path = f"{self.pseudo_dir_str}/{file_path}.pseudo"
                content = direct_read_file(pseudo_path)
                
                if "ERROR" in content:
                    self._log_to_file(f"Could not read pseudocode for {file_path}")
                    continue
                
                self._log_to_file(f"Parsing dependencies for: {file_path}")
                
                # Extract imports/dependencies section
                deps_match = re.search(
                    r'# Imports/Dependencies:(.*?)(# Main Logic:|# Functions/Classes:|# Exports/Outputs:|END FILE|$)', 
                    content, 
                    re.DOTALL | re.IGNORECASE
                )
                
                if not deps_match:
                    self._log_to_file(f"No dependencies section found in {file_path}")
                    continue
                    
                deps_section = deps_match.group(1)
                self._log_to_file(f"Dependencies section for {file_path}: {deps_section[:100]}...")
                
                # Enhanced regex patterns for different import styles
                import_patterns = [
                    # import Component from './path' or import Component from 'path'
                    r"import\s+\w+\s+from\s+['\"]([^'\"]*\.js)['\"]",
                    # import { thing } from './path'
                    r"import\s+\{[^}]+\}\s+from\s+['\"]([^'\"]*\.js)['\"]",
                    # const Thing = require('./path')
                    r"require\s*\(\s*['\"]([^'\"]*\.js)['\"]\s*\)",
                    # from './path' (pseudocode style)
                    r"from\s+['\"]([^'\"]*\.js)['\"]",
                    # Import from ./path (pseudocode style)
                    r"Import\s+.*from\s+['\"]([^'\"]*\.js)['\"]",
                    # - Import variable from path (bullet point style)
                    r"-\s*Import\s+.*from\s+([^\s]+\.js)",
                ]
                
                found_deps = set()
                
                for pattern in import_patterns:
                    matches = re.findall(pattern, deps_section, re.IGNORECASE)
                    for match in matches:
                        # Clean up the path
                        clean_path = match.strip().lstrip('./')
                        if clean_path and clean_path != file_path:  # Don't self-reference
                            found_deps.add(clean_path)
                
                if found_deps:
                    self._log_to_file(f"Found dependencies for {file_path}: {list(found_deps)}")
                    
                    for imported_file in found_deps:
                        if imported_file in all_files:
                            # Add edge: imported_file -> file_path (file_path depends on imported_file)
                            graph.add_edge(imported_file, file_path)
                            self._log_to_file(f"Added dependency: {imported_file} -> {file_path}")
                        else:
                            self._log_to_file(f"Dependency {imported_file} not found in file list (referenced by {file_path})")
                else:
                    self._log_to_file(f"No valid dependencies found for {file_path}")
            
            total_edges = graph.number_of_edges()
            self._log_to_file(f"Dependency graph created: {graph.number_of_nodes()} nodes, {total_edges} edges")
            
            if total_edges == 0:
                self._log_to_file("WARNING: Dependency graph has no edges - this may indicate parsing issues")
                
            return graph
            
        except Exception as e:
            self._log_to_file(f"Failed to create dependency graph: {e}")
            # Return simple graph with all nodes, no edges
            graph = nx.DiGraph()
            data = self._load_files_json()
            for file_path in data.get('files', {}).keys():
                graph.add_node(file_path)
            return graph
    

    def _extract_dependencies_from_plan(self, plan_content: str):
        """Extract npm dependencies from plan and create dependencies.json."""
        try:
            dependency_task = Task(
                description=f"""Based on the following plan, list all necessary npm packages in JSON format.
                
                Plan: {plan_content}
                
                Output a JSON object with this structure:
                {{
                    "dependencies": ["express", "mongoose", "cors", "etc"],
                    "devDependencies": ["nodemon", "@types/node", "etc"]
                }}
                
                Include only actual package names, no versions.""",
                expected_output="JSON object with dependencies arrays",
                agent=self.planner_agent
            )
            
            crew = Crew(
                agents=[self.planner_agent],
                tasks=[dependency_task],
                verbose=True,
                process=Process.sequential
            )
            
            result = crew.kickoff()
            deps_content = result.raw if hasattr(result, 'raw') else str(result)
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', deps_content, re.DOTALL)
            if json_match:
                deps_data = json.loads(json_match.group(0))
                deps_file_path = str(self.working_dir / "dependencies.json")
                direct_write_file(deps_file_path, json.dumps(deps_data, indent=2))
                self._log_to_file(f"Dependencies extracted: {len(deps_data.get('dependencies', []))} deps, {len(deps_data.get('devDependencies', []))} devDeps")
                return True
            
        except Exception as e:
            self._log_to_file(f"Failed to extract dependencies: {e}")
            return False
    
    def _load_files_json(self) -> Dict:
        """Load files.json data safely."""
        try:
            with open(self.files_json, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {"files": {}}

    def _save_files_json(self, data: Dict):
        """Save files.json data safely."""
        try:
            with open(self.files_json, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self._log_to_file(f"Error saving files.json: {e}")

    def _get_unfinished_files(self, phase: str) -> List[Dict]:
        """Get list of unfinished files for a given phase."""
        data = self._load_files_json()
        flag = f"is_{phase}"
        unfinished = []
        
        for path, file_info in data.get('files', {}).items():
            if not file_info.get(flag, False):
                # Create consistent structure
                file_dict = {
                    'path': path,
                    'description': file_info.get('description', 'No description'),
                    'type': file_info.get('type', 'other')
                }
                unfinished.append(file_dict)
        
        return unfinished

    def _all_phase_complete(self, phase: str) -> bool:
        """Check if all files have completed the given phase."""
        return len(self._get_unfinished_files(phase)) == 0

    def _create_global_summary(self):
        """Create global summary from all pseudocode files."""
        summaries = []
        data = self._load_files_json()
        
        for file_path in data.get('files', {}).keys():
            pseudo_path = f"{self.pseudo_dir_str}/{file_path}.pseudo"
            content = direct_read_file(pseudo_path)
            
            if "ERROR" in content:
                continue
                
            # Extract dependencies section
            deps_match = re.search(
                r'# Imports/Dependencies:(.*?)(# Main Logic:|# Functions/Classes:|# Exports/Outputs:|END FILE|$)', 
                content, 
                re.DOTALL | re.IGNORECASE
            )
            deps = deps_match.group(1).strip() if deps_match else ""
            summaries.append(f"File {file_path}: {deps}")
        
        summary_content = "\n".join(summaries)
        direct_write_file(str(self.global_summary), summary_content)
        self._log_to_file("Global summary created")

    def _batch_files_by_type(self, files: List[Dict], batch_size: int = 5) -> List[List[Dict]]:
        """Original type-based batching as fallback."""
        # Group by type first
        groups = {'frontend': [], 'backend': [], 'config': [], 'other': []}
        
        for f in files:
            path_lower = f['path'].lower()
            file_type = f.get('type', 'other')
            
            # Map unexpected types to expected ones
            if file_type == 'documentation':
                file_type = 'config'
            elif file_type == 'styles':
                file_type = 'frontend'
            
            if file_type in groups:
                groups[file_type].append(f)
            elif 'src/' in path_lower or 'components/' in path_lower or path_lower.endswith(('.js', '.jsx', '.css')):
                groups['frontend'].append(f)
            elif 'models/' in path_lower or 'routes/' in path_lower or 'server.js' in path_lower:
                groups['backend'].append(f)
            elif path_lower in ['package.json', '.env', '.gitignore', 'readme.md']:
                groups['config'].append(f)
            else:
                groups['other'].append(f)
        
        # Create batches from groups
        batches = []
        for group in groups.values():
            for i in range(0, len(group), batch_size):
                batch = group[i:i+batch_size]
                if batch:
                    batches.append(batch)
        
        return batches


    def _batch_files(self, files: List[Dict], batch_size: int = 5, dependency_graph=None) -> List[List[Dict]]:
        """Group files into batches respecting dependency order."""
        
        self._log_to_file(f"DEBUG: _batch_files called with {len(files)} files")
        
        if dependency_graph is None:
            # Fallback to original type-based batching
            return self._batch_files_by_type(files, batch_size)
        
        try:
            # Get topologically sorted order
            file_paths = [f['path'] for f in files]
            subgraph = dependency_graph.subgraph(file_paths)
            
            if nx.is_directed_acyclic_graph(subgraph):
                sorted_paths = list(nx.topological_sort(subgraph))
            else:
                self._log_to_file("Dependency cycle detected, falling back to simple ordering")
                sorted_paths = file_paths
            
            # Create file lookup
            file_lookup = {f['path']: f for f in files}
            
            # Create batches in dependency order
            batches = []
            for i in range(0, len(sorted_paths), batch_size):
                batch_paths = sorted_paths[i:i+batch_size]
                batch = [file_lookup[path] for path in batch_paths if path in file_lookup]
                if batch:
                    batches.append(batch)
            
            self._log_to_file(f"Created {len(batches)} dependency-ordered batches")
            return batches
            
        except Exception as e:
            self._log_to_file(f"Error in dependency-based batching: {e}")
            return self._batch_files_by_type(files, batch_size)

    def __init__(self, reasoning_model: str = "ollama/granite3.3:2b-largectx", 
                 coding_model: str = "ollama/qwen2.5-coder:7b-largectx",
                 outputs_dir: Optional[str] = None, working_dir: Optional[str] = None):
        self.logger = setup_logger(__name__)
        self.reasoning_model = reasoning_model
        self.coding_model = coding_model
        self.context_threshold = 3200
        
        # Setup directories
        self.outputs_dir = Path(outputs_dir or "outputs").resolve()
        self.working_dir = Path(working_dir or "working").resolve()
        self.pseudo_dir = self.working_dir / "pseudo_files"
        
        # Create directories
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.pseudo_dir.mkdir(parents=True, exist_ok=True)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.plan_file = self.working_dir / "plan.txt"
        self.files_json = self.working_dir / "files.json"
        self.log_file = self.working_dir / "generation.log"
        self.global_summary = self.pseudo_dir / "global_summary.txt"
        
        # String paths for prompts (normalized)
        self.outputs_dir_str = str(self.outputs_dir).replace("\\", "/")
        self.pseudo_dir_str = str(self.pseudo_dir).replace("\\", "/")
        self.plan_file_str = str(self.plan_file).replace("\\", "/")
        self.files_json_str = str(self.files_json).replace("\\", "/")
        self.log_file_str = str(self.log_file).replace("\\", "/")
        
        # Configuration
        self.max_retries = 3
        self.max_regens_per_file = 5
        
        # Initialize LLMs
        self.reasoning_llm = LLM(model=reasoning_model, temperature=0.0, max_tokens=4000)
        self.coding_llm = LLM(model=coding_model, temperature=0.05, max_tokens=8000)
        
        # Initialize log and agents
        self._initialize_log_file()
        self._initialize_agents()
    
    def _sanitize_path(self, path: str) -> Optional[str]:
        """Sanitize file path by removing invalid characters."""
        if not path:
            return None
            
        clean = re.sub(r'\*+', '', path).strip()  # Remove **
        clean = re.sub(r'[\<\>\:\|\"\?\*]', '', clean)  # Remove Windows-invalid chars
        
        if '&' in clean:
            self._log_to_file(f"Skipping invalid concatenated path: {path}")
            return None
            
        return clean.replace('\\', '/')

    def _initialize_log_file(self):
        """Initialize the generation log file."""
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write(f"MERN Code Generation Log - Started at {datetime.now()}\n")
                f.write("="*60 + "\n\n")
        except Exception as e:
            print(f"Warning: Could not initialize log file: {e}")

    def _initialize_agents(self):
        """Initialize all agents."""
        try:
            self.planner_agent = get_planner_agent(self.reasoning_llm)
            self.pseudo_gen_agent = get_pseudo_gen_agent(self.reasoning_llm)
            self.pseudo_ver_agent = get_pseudo_ver_agent(self.reasoning_llm)
            self.code_gen_agent = get_code_gen_agent(self.coding_llm)
            self.code_ver_agent = get_code_ver_agent(self.coding_llm)
            self.sanity_check_agent = get_sanity_check_agent(self.reasoning_llm)
        except Exception as e:
            self.logger.error(f"Failed to initialize agents: {e}")
            raise
    
    def _log_to_file(self, message: str):
        """Append message to log file with caller info."""
        try:
            # Get the name of the function that called this one
            caller_name = inspect.stack()[1].function
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - [{caller_name}] - {message}\n")
        except Exception:
            pass  # Silent fail for logging

    def _clean_code_content(self, content: str, file_path: str = "") -> str:
        """
        More aggressive cleaning to remove markdown wrappers, explanations, 
        and incorrect JSON blocks.
        """
        # For JSON files, be extremely strict: find the first valid JSON object and return only that.
        if file_path.lower().endswith(".json"):
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                try:
                    # Test if the extracted part is valid JSON
                    json.loads(json_match.group(0))
                    return json_match.group(0)
                except json.JSONDecodeError:
                    # Fallback if the regex-matched part is still not valid
                    raise ValueError("Extracted content for JSON file is not valid JSON.")
            else:
                raise ValueError("No JSON object found in the output for package.json")

        # For other files, remove markdown and common LLM chatter
        # Remove ```...``` blocks
        content = re.sub(r'```[a-zA-Z]*\n', '', content)
        content = re.sub(r'```\n', '', content)
        content = re.sub(r'```', '', content)
        
        # Aggressively remove any JSON block at the beginning of non-JSON files
        if not file_path.lower().endswith(".json"):
             # This regex finds a JSON object that starts the file, possibly with whitespace before it.
            content = re.sub(r'^\s*\{.*\}\s*\n', '', content, flags=re.DOTALL)

        return content.strip()

    
    def generate_mern_code(self, description: str) -> Dict[str, str]:
        """Generate MERN code from description."""
        try:
            # Validate input
            validation = validate_description(description)
            if not validation["valid"]:
                raise ValueError(f"Invalid description: {validation['errors']}")
            
            self.logger.info("Starting MERN Code Generation...")
            self._log_to_file("Generation started")
            
            # Phase 1: Planning with Retry Logic
            plan_attempts = 0
            max_plan_attempts = 3  # 1 initial try + 2 retries
            
            while plan_attempts < max_plan_attempts:
                try:
                    if plan_attempts == 0:
                        self._execute_planning_phase(description)
                    
                    self._validate_plan()
                    self._log_to_file("Plan validation successful.")
                    break  # Exit loop if plan is valid
                    
                except ValueError as e:
                    plan_attempts += 1
                    error_message = str(e)
                    self._log_to_file(f"Plan validation failed (Attempt {plan_attempts}/{max_plan_attempts}): {error_message}")
                    
                    if plan_attempts >= max_plan_attempts:
                        self._log_to_file("Max planning attempts reached. Failing generation.")
                        raise ValueError(f"Failed to create a valid plan after {max_plan_attempts} attempts. Last error: {error_message}")
                    else:
                        self._log_to_file("Attempting to regenerate the plan.")
                        self._regenerate_plan(error_message)
            
            # Phase 2: Pseudocode Generation and Verification Loop
            self._execute_pseudo_loop()
            
            # Phase 3: Code Generation and Verification Loop
            self._execute_code_loop()
            
            # Collect and validate final files
            files_dict = collect_files(str(self.outputs_dir))
            validation_result = validate_file_structure(files_dict)
            
            if validation_result["errors"]:
                self.logger.warning(f"Validation errors: {validation_result['errors']}")
            if validation_result["warnings"]:
                self.logger.warning(f"Validation warnings: {validation_result['warnings']}")
            
            # Create manifest
            create_file_manifest(files_dict, str(self.outputs_dir / "manifest.json"))
            
            self.logger.info("Generation completed successfully!")
            return files_dict
            
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            self._log_to_file(f"FATAL ERROR: {e}")
            raise

    def _regenerate_plan(self, error_message: str):
        """Regenerate the plan based on validation feedback."""
        self._log_to_file("Starting plan regeneration")
        
        try:
            previous_plan = direct_read_file(self.plan_file_str)
            if "ERROR" in previous_plan:
                raise ValueError("Failed to read the previous plan for regeneration.")

            task_desc = PLAN_REGEN_PROMPT.format(
                previous_plan=previous_plan,
                error_message=error_message
            )
            
            task = Task(
                description=task_desc,
                expected_output="A new, corrected, and comprehensive plan text with JSON files at the end",
                agent=self.planner_agent
            )
            
            crew = Crew(
                agents=[self.planner_agent], 
                tasks=[task], 
                verbose=True, 
                process=Process.sequential
            )
            
            result = crew.kickoff()
            new_plan_content = result.raw if hasattr(result, 'raw') else str(result)
            
            direct_write_file(self.plan_file_str, new_plan_content)
            self._log_to_file("New plan generated and saved.")

            self._extract_and_save_files_json(new_plan_content)
            self._extract_dependencies_from_plan(new_plan_content)
            
            self._log_to_file("Plan regeneration completed. Re-validation will occur.")

        except Exception as e:
            self.logger.error(f"Plan regeneration phase failed: {e}")
            raise

    def _execute_planning_phase(self, description: str):
        """Execute the planning phase."""
        self._log_to_file("Starting planning phase")
        
        try:
            task_desc = PLANNER_PROMPT.format(description=description)
            
            task = Task(
                description=task_desc,
                expected_output="Comprehensive plan text with JSON files at end",
                agent=self.planner_agent
            )
            
            crew = Crew(
                agents=[self.planner_agent], 
                tasks=[task], 
                verbose=True, 
                process=Process.sequential
            )
            
            result = crew.kickoff()
            plan_content = result.raw if hasattr(result, 'raw') else str(result)
            
            # Save plan
            direct_write_file(self.plan_file_str, plan_content)
            
            # Extract and save files JSON
            self._extract_and_save_files_json(plan_content)
            
            # NEW: Extract dependencies for better package.json generation
            self._extract_dependencies_from_plan(plan_content)
            
            self._log_to_file("Planning completed")
            
        except Exception as e:
            self.logger.error(f"Planning phase failed: {e}")
            raise

    def _extract_and_save_files_json(self, plan_content: str):
        """Extract files JSON from plan content, save to files.json, and programmatically ensure package.json exists."""
        try:
            transformed_files = {}
            # Try to find JSON in the plan content
            json_match = re.search(r'\{[^{}]*"files"[^{}]*\[.*?\][^{}]*\}', plan_content, re.DOTALL)
            
            if json_match:
                try:
                    files_data = json.loads(json_match.group(0))
                    # Transform to internal format
                    for file_entry in files_data.get('files', []):
                        if isinstance(file_entry, dict) and 'path' in file_entry:
                            path = file_entry['path']
                            transformed_files[path] = {
                                'type': file_entry.get('type', 'other'),
                                'description': file_entry.get('description', ''),
                                'is_pseudo_gen': False,
                                'is_pseudo_ver': False,
                                'is_code_gen': False,
                                'is_code_ver': False
                            }
                except json.JSONDecodeError as e:
                    self._log_to_file(f"Failed to parse files JSON: {e}")
            
            # Fallback to parsing from structure if no files were extracted from JSON
            if not transformed_files:
                self._parse_files_from_plan_structure(plan_content)
                # Load the result of the parse
                data = self._load_files_json()
                transformed_files = data.get('files', {})

            # Programmatically ensure package.json exists, regardless of the parsing method
            if 'package.json' not in transformed_files:
                self._log_to_file("Planner failed to include 'package.json'. Injecting a default entry.")
                transformed_files['package.json'] = {
                    'type': 'config',
                    'description': 'Project dependencies and scripts. Auto-generated due to planning omission.',
                    'is_pseudo_gen': False,
                    'is_pseudo_ver': False,
                    'is_code_gen': False,
                    'is_code_ver': False
                }
            
            if transformed_files:
                self._save_files_json({'files': transformed_files})
                self._log_to_file(f"Extracted and finalized {len(transformed_files)} files from plan")
                return
            
        except Exception as e:
            self._log_to_file(f"Error extracting files JSON: {e}")
            # Create minimal fallback
            self._create_minimal_files_structure()

    def _parse_files_from_plan_structure(self, plan_content: str):
        """Parse files from plan structure section."""
        tracking_data = {"files": {}}
        
        # Find file structure section
        structure_patterns = [
            "## COMPLETE FILE STRUCTURE",
            "## FILE STRUCTURE", 
            "## Project Structure",
            "## Files Structure"
        ]
        
        structure_start = -1
        for pattern in structure_patterns:
            structure_start = plan_content.find(pattern)
            if structure_start != -1:
                break
        
        if structure_start == -1:
            self._log_to_file("No file structure section found in plan")
            self._create_minimal_files_structure()
            return
        
        # Extract structure content
        next_section = plan_content.find("##", structure_start + 1)
        if next_section == -1:
            structure_content = plan_content[structure_start:]
        else:
            structure_content = plan_content[structure_start:next_section]
        
        # Parse file lines
        lines = structure_content.split('\n')
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#') or ':' not in line:
                continue
            
            # Handle different formats
            if line.startswith('- '):
                line = line[2:].strip()
            
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    filename = parts[0].strip()
                    description = parts[1].strip()
                    
                    # Skip invalid filenames
                    if not filename or filename.endswith(':') or len(filename) < 2:
                        continue
                    
                    # Determine file type
                    file_type = self._determine_file_type(filename)
                    
                    tracking_data["files"][filename] = {
                        "type": file_type,
                        "description": description,
                        "is_pseudo_gen": False,
                        "is_pseudo_ver": False,
                        "is_code_gen": False,
                        "is_code_ver": False
                    }
        
        if tracking_data["files"]:
            self._save_files_json(tracking_data)
            self._log_to_file(f"Parsed {len(tracking_data['files'])} files from plan structure")
        else:
            self._create_minimal_files_structure()

    def _determine_file_type(self, filename: str) -> str:
        """Determine file type based on path."""
        path_lower = filename.lower()
        
        if filename in ['package.json', '.env', '.gitignore', 'README.md']:
            return 'config'
        elif 'src/' in path_lower or 'components/' in path_lower or path_lower.endswith(('.js', '.jsx', '.css')):
            return 'frontend'
        elif 'models/' in path_lower or 'routes/' in path_lower or 'server.js' in path_lower:
            return 'backend'
        else:
            return 'other'

    def _create_minimal_files_structure(self):
        """Create minimal fallback files structure."""
        minimal_files = {
            "files": {
                "package.json": {
                    "type": "config",
                    "description": "Project dependencies and scripts",
                    "is_pseudo_gen": False,
                    "is_pseudo_ver": False,
                    "is_code_gen": False,
                    "is_code_ver": False
                },
                "server.js": {
                    "type": "backend", 
                    "description": "Main Express server file",
                    "is_pseudo_gen": False,
                    "is_pseudo_ver": False,
                    "is_code_gen": False,
                    "is_code_ver": False
                },
                "src/App.js": {
                    "type": "frontend",
                    "description": "Main React application component",
                    "is_pseudo_gen": False,
                    "is_pseudo_ver": False,
                    "is_code_gen": False,
                    "is_code_ver": False
                },
                ".env": {
                    "type": "config",
                    "description": "Environment variables",
                    "is_pseudo_gen": False,
                    "is_pseudo_ver": False,
                    "is_code_gen": False,
                    "is_code_ver": False
                },
                "README.md": {
                    "type": "config",
                    "description": "Project documentation",
                    "is_pseudo_gen": False,
                    "is_pseudo_ver": False,
                    "is_code_gen": False,
                    "is_code_ver": False
                }
            }
        }
        
        self._save_files_json(minimal_files)
        self._log_to_file("Created minimal files structure")
    
    def _execute_pseudo_loop(self):
        """Execute pseudocode generation and verification loop."""
        self._log_to_file("Starting pseudocode generation and verification loop")
        
        # NEW: Check for files needing pseudocode review
# NEW: Check for files needing pseudocode review
        try:
            data = self._load_files_json()
            review_files = []
            for path, file_info in data.get('files', {}).items():
                if file_info.get('needs_pseudo_review', False):
                    review_files.append({
                        'path': path,
                        'description': file_info.get('description', 'No description'),
                        'verification_issues': file_info.get('verification_issues', '')
                    })
            
            if review_files:
                self._log_to_file(f"Found {len(review_files)} files needing pseudocode review")
                for file_info in review_files:
                    path = file_info['path']
                    desc = file_info['description']
                    issues = file_info['verification_issues']
                    
                    success = self._generate_pseudocode_for_file(path, desc, verification_feedback=issues)
                    if success:
                        # Clear review flag
                        data = self._load_files_json()
                        if path in data.get('files', {}):
                            data['files'][path]['needs_pseudo_review'] = False
                            data['files'][path].pop('verification_issues', None)
                            self._save_files_json(data)
                    else:
                        self._log_to_file(f"Failed to regenerate pseudocode for {path} during review")

        except Exception as e:
            self._log_to_file(f"Error during pseudocode review phase: {e}")

        # Generation phase
        unfinished_gen = self._get_unfinished_files('pseudo_gen')
        self._log_to_file(f"Found {len(unfinished_gen)} files needing pseudocode generation")
        
        for file_info in unfinished_gen:
            path = file_info['path']
            desc = file_info.get('description', 'No description')
            
            success = self._generate_pseudocode_for_file(path, desc)
            if not success:
                self._log_to_file(f"Failed to generate pseudocode for {path} after max retries")
            
        
        if self._all_phase_complete('pseudo_gen'):
            self._log_to_file("All pseudocode generation complete - running sanity check")
            self._execute_sanity_check()
            
            self._create_global_summary()
            dependency_graph = self._create_dependency_graph()
        else:
            dependency_graph = None

        # Create global summary and dependency graph if all generation complete
        if self._all_phase_complete('pseudo_gen'):
            self._create_global_summary()
            dependency_graph = self._create_dependency_graph()  # NEW
        else:
            dependency_graph = None
        
        # Verification phase with dependency-aware batching
        unfinished_ver = self._get_unfinished_files('pseudo_ver')
        if unfinished_ver:
            self._log_to_file(f"Found {len(unfinished_ver)} files needing pseudocode verification")
            
            # Use dependency graph for batching
            batches = self._batch_files(unfinished_ver, batch_size=5, dependency_graph=dependency_graph)
            
            for batch in batches:
                self._verify_pseudocode_batch(batch)
        
        self._log_to_file("Pseudocode loop complete")

    def _generate_pseudocode_for_file(self, path: str, description: str, verification_feedback: str = None) -> bool:
        """Generate pseudocode for a single file."""
        retries = 0
        per_file_regens = 0
        
        while retries < self.max_retries and per_file_regens < self.max_regens_per_file:
            try:
                # Read plan content
                plan_content = direct_read_file(self.plan_file_str)
                if "ERROR" in plan_content:
                    self._log_to_file(f"Failed to read plan for {path}: {plan_content}")
                    return False
                
                # Create task with optional feedback
                task_desc = PSEUDO_GEN_PROMPT.format(
                    plan_content=plan_content,
                    file_path=path,
                    file_desc=description,
                    verification_feedback=verification_feedback or ""  # NEW parameter
                )
                
                
                task = Task(
                    description=task_desc,
                    expected_output="Pseudocode content",
                    agent=self.pseudo_gen_agent
                )
                
                crew = Crew(
                    agents=[self.pseudo_gen_agent], 
                    tasks=[task], 
                    verbose=True, 
                    process=Process.sequential
                )
                
                result = crew.kickoff()
                pseudo_content = result.raw if hasattr(result, 'raw') else str(result)
                
                # Validate content
                if len(pseudo_content.strip()) < 50 or "BEGIN FILE" not in pseudo_content:
                    raise ValueError("Incomplete pseudocode generated")
                
                # Sanitize path and write file
                clean_path = self._sanitize_path(path)
                if clean_path is None:
                    self._log_to_file(f"Invalid path, skipping: {path}")
                    return False
                
                pseudo_file_path = f"{self.pseudo_dir_str}/{clean_path}.pseudo"
                write_result = direct_write_file(pseudo_file_path, pseudo_content)
                
                if "ERROR" in write_result:
                    raise ValueError(f"Failed to write pseudocode: {write_result}")
                
                # Update tracking
                data = self._load_files_json()
                if path in data.get('files', {}):
                    data['files'][path]['is_pseudo_gen'] = True
                    self._save_files_json(data)
                
                self._log_to_file(f"Pseudocode generated for {path}")
                return True
                
            except Exception as e:
                self._log_to_file(f"Error generating pseudocode for {path}: {e}")
                retries += 1
                per_file_regens += 1
        
        return False

    def _create_project_context(self, data: Dict) -> str:
        """Create a summary of all project files for context."""
        files = data.get('files', {})
        context_lines = []
        
        # Group files by type for better organization
        file_types = {'backend': [], 'frontend': [], 'config': [], 'other': []}
        
        for path, info in files.items():
            file_type = info.get('type', 'other')
            if file_type not in file_types:
                file_type = 'other'
            file_types[file_type].append(f"  - {path}: {info.get('description', 'No description')}")
        
        for file_type, file_list in file_types.items():
            if file_list:
                context_lines.append(f"{file_type.upper()} FILES:")
                context_lines.extend(file_list)
                context_lines.append("")
        
        return "\n".join(context_lines)


    def _verify_pseudocode_batch(self, batch: List[Dict]) -> bool:
        """Verify a batch of pseudocode files with improved error handling."""
        batch_files = [f['path'] for f in batch]
        retries = 0
        
        while retries < self.max_retries:
            try:
                plan_content = direct_read_file(self.plan_file_str)
                global_summary_content = direct_read_file(str(self.global_summary))
                
                # Use more specific error checking
                if plan_content.startswith("ERROR:"):
                    raise ValueError(f"Failed to read plan: {plan_content}")
                
                batch_pseudo_contents = ""
                for file_path in batch_files:
                    pseudo_path = self.pseudo_dir / f"{file_path}.pseudo"
                    
                    if not pseudo_path.exists() or pseudo_path.stat().st_size == 0:
                        self._log_to_file(f"Verification failed: Pseudocode file is missing or empty for {file_path}")
                        data = self._load_files_json()
                        if file_path in data.get('files', {}):
                            data['files'][file_path]['is_pseudo_gen'] = False
                        self._save_files_json(data)
                        continue

                    pseudo_content = direct_read_file(str(pseudo_path))
                    
                    # Use more specific error checking
                    if pseudo_content.startswith("ERROR:"):
                        raise ValueError(f"Failed to read pseudocode for {file_path}")
                    
                    batch_pseudo_contents += f"--- Pseudocode for {file_path} ---\n{pseudo_content}\n\n"
                
                if not batch_pseudo_contents.strip():
                    self._log_to_file("No valid pseudocode files to verify in this batch. Skipping.")
                    return True

                task_desc = PSEUDO_VER_PROMPT.format(
                    plan_content=plan_content,
                    global_summary_content=global_summary_content,
                    batch_pseudo_contents=batch_pseudo_contents
                )
                
                task = Task(description=task_desc, expected_output="JSON verification results", agent=self.pseudo_ver_agent)
                crew = Crew(agents=[self.pseudo_ver_agent], tasks=[task], verbose=True)
                
                result = crew.kickoff()
                result_text = result.raw if hasattr(result, 'raw') else str(result)
                
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if not json_match:
                    raise ValueError("Verification response did not contain valid JSON.")
                
                ver_results = json.loads(json_match.group(0))
                
                data = self._load_files_json()
                all_passed = True
                for file_path, verification in ver_results.items():
                    if file_path in data.get('files', {}):
                        if verification.get('pass', False):
                            data['files'][file_path]['is_pseudo_ver'] = True
                        else:
                            all_passed = False
                            issues = verification.get('issues', 'Unknown issues')
                            self._log_to_file(f"Verification failed for {file_path}: {issues}")
                            data['files'][file_path]['is_pseudo_gen'] = False
                
                self._save_files_json(data)
                if not all_passed:
                    self._log_to_file(f"Batch verification had failures. Some files marked for regeneration.")

                return True
                
            except Exception as e:
                self._log_to_file(f"Error verifying batch {batch_files}: {e}")
                retries += 1
        
        self._log_to_file(f"Max retries reached for batch {batch_files}")
        return False
    
    def _execute_code_loop(self):
        """Execute code generation and verification loop."""
        self._log_to_file("Starting code generation and verification loop")
        
        # Generation phase
        unfinished_gen = self._get_unfinished_files('code_gen')
        self._log_to_file(f"Found {len(unfinished_gen)} files needing code generation")
        
        for file_info in unfinished_gen:
            path = file_info['path']
            success = self._generate_code_for_file(path)
            if not success:
                self._log_to_file(f"Failed to generate code for {path} after max retries")
        
        # Individual verification phase
        unfinished_ver = self._get_unfinished_files('code_ver')
        self._log_to_file(f"Found {len(unfinished_ver)} files needing code verification")
        
        for file_info in unfinished_ver:
            path = file_info['path']
            success = self._verify_code_file(path)
            if not success:
                self._log_to_file(f"Failed to verify code for {path} after max retries")
        
        self._log_to_file("Code loop complete")

    def _generate_code_for_file(self, path: str) -> bool:
        """Generate code for a single file with enhanced context and retry logic."""
        retries = 0
        per_file_regens = 0
        retry_error = ""
        
        while retries < self.max_retries and per_file_regens < self.max_regens_per_file:
            try:
                # Read pseudocode
                pseudo_path = f"{self.pseudo_dir_str}/{path}.pseudo"
                pseudo_content = direct_read_file(pseudo_path)
                
                if pseudo_content.startswith("ERROR:"):
                    raise ValueError(f"Failed to read pseudocode for {path}: {pseudo_content}")
                
                # Get file info and project context
                data = self._load_files_json()
                file_info = data.get('files', {}).get(path, {})
                file_desc = file_info.get('description', 'No description available')
                
                # Create project context
                project_context = self._create_project_context(data)
                
                # NEW: For package.json, inject dependency list
                dependency_list = ""
                if path == "package.json":
                    deps_file = str(self.working_dir / "dependencies.json")
                    deps_content = direct_read_file(deps_file)
                    if "ERROR" not in deps_content:
                        dependency_list = deps_content
                
                # Create task with enhanced context
                task = Task(
                    description=CODE_GEN_PROMPT.format(
                        pseudo_content=pseudo_content,
                        file_desc=file_desc,
                        dependency_list=dependency_list,
                        project_context=project_context,
                        retry_error=retry_error
                    ),
                    agent=self.code_gen_agent,
                    expected_output="The complete code for the file."
                )
                
                crew = Crew(
                    agents=[self.code_gen_agent], 
                    tasks=[task], 
                    verbose=True, 
                    process=Process.sequential
                )
                
                result = crew.kickoff()
                code_content = result.raw if hasattr(result, 'raw') else str(result)
                code_content = self._clean_code_content(code_content, path)
                
                # Validate content
                if len(code_content.strip()) < 50:
                    error_msg = f"Incomplete code generated for {path} - content too short"
                    self._log_to_file(error_msg)
                    if retries < self.max_retries - 1:  # Will retry
                        retry_error = error_msg
                        retries += 1
                        continue
                    else:
                        raise ValueError(error_msg)
                
                # Write code file
                code_path = f"{self.outputs_dir_str}/{path}"
                # Special validation for JSON files
                if path.endswith('.json'):
                    try:
                        import json
                        json.loads(code_content)  # Validate JSON
                    except json.JSONDecodeError as e:
                        error_msg = f"Generated content is not valid JSON for {path}: {e}"
                        self._log_to_file(error_msg)
                        if retries < self.max_retries - 1:  # Will retry
                            retry_error = error_msg
                            retries += 1
                            continue
                        else:
                            raise ValueError(error_msg)

                write_result = direct_write_file(code_path, code_content)
                
                if "ERROR" in write_result:
                    error_msg = f"Failed to write code for {path}: {write_result}"
                    self._log_to_file(error_msg)
                    if retries < self.max_retries - 1:  # Will retry
                        retry_error = error_msg
                        retries += 1
                        continue
                    else:
                        raise ValueError(error_msg)
                
                # Update tracking
                if 'files' in data and path in data['files']:
                    data['files'][path]['is_code_gen'] = True
                    self._save_files_json(data)
                
                self._log_to_file(f"Code generated for {path} (attempt {retries + 1})")
                return True
                    
            except Exception as e:
                error_msg = f"Error generating code for {path}: {e}"
                self._log_to_file(error_msg)
                retries += 1
                per_file_regens += 1
                retry_error = error_msg
        
        self._log_to_file(f"Failed to generate code for {path} after {retries} attempts")
        return False
    

    def _regenerate_code_with_feedback(self, file_path: str, verification_issues: str) -> bool:
        """Regenerate code for a file with verification feedback."""
        try:
            # Read pseudocode
            pseudo_path = f"{self.pseudo_dir_str}/{file_path}.pseudo"
            pseudo_content = direct_read_file(pseudo_path)
            
            if pseudo_content.startswith("ERROR:"):
                return False
            
            # Create enhanced task with verification feedback
            task_desc = f"""
            Generate improved code for this file based on pseudocode and verification feedback:

            PSEUDOCODE:
            {pseudo_content}

            PREVIOUS VERIFICATION ISSUES:
            {verification_issues}

            CRITICAL INSTRUCTIONS:
            - If this is a JSON file (package.json, etc.): Output ONLY valid JSON, NO JavaScript code, NO comments
            - If this is a JavaScript file: Follow all best practices with proper syntax
            - Address the specific issues mentioned in verification feedback
            - Use modern MERN stack conventions
            - Include proper error handling and validation
            - DO NOT include any explanatory text or comments at the end

            Output ONLY the complete code content ready to save directly to the file.
            """
            
            task = Task(
                description=task_desc,
                expected_output="Clean code content",
                agent=self.code_gen_agent
            )
            
            crew = Crew(
                agents=[self.code_gen_agent], 
                tasks=[task], 
                verbose=True, 
                process=Process.sequential
            )
            
            result = crew.kickoff()
            code_content = result.raw if hasattr(result, 'raw') else str(result)
            code_content = self._clean_code_content(code_content, file_path)
            
            # Validate content
            if len(code_content.strip()) < 50:
                return False
            
            # Write improved code file
            code_path = f"{self.outputs_dir_str}/{file_path}"
            write_result = direct_write_file(code_path, code_content)
            
            if "ERROR" in write_result:
                return False
            
            self._log_to_file(f"Code regenerated for {file_path} with verification feedback")
            return True
            
        except Exception as e:
            self._log_to_file(f"Error regenerating code for {file_path}: {e}")
            return False
    

    def _verify_code_file(self, file_path: str) -> bool:
        """Verify a single code file against its pseudocode."""
        retries = 0
        regen_count = 0
        max_regens = 5
        
        while retries < self.max_retries:
            try:
                # Read both code and pseudocode files
                code_path = f"{self.outputs_dir_str}/{file_path}"
                pseudo_path = f"{self.pseudo_dir_str}/{file_path}.pseudo"
                
                code_content = direct_read_file(code_path)
                pseudo_content = direct_read_file(pseudo_path)
                
                if "ERROR" in code_content:
                    self._log_to_file(f"Code file not found for verification: {file_path}")
                    return False
                    
                if "ERROR" in pseudo_content:
                    self._log_to_file(f"Pseudocode file not found for verification: {file_path}")
                    return False
                
                # Create verification task with content injected in prompt
                task_desc = CODE_VER_PROMPT.format(
                    file_path=file_path,
                    pseudo_content=pseudo_content,
                    code_content=code_content
                )
                task = Task(
                    description=task_desc,
                    expected_output="PASS or FAIL with reason",
                    agent=self.code_ver_agent
                )
                
                crew = Crew(
                    agents=[self.code_ver_agent], 
                    tasks=[task], 
                    verbose=True, 
                    process=Process.sequential
                )
                
                result = crew.kickoff()
                result_text = result.raw if hasattr(result, 'raw') else str(result)
                
                # Parse result
                if "PASS" in result_text.upper():
                    # Update tracking
                    data = self._load_files_json()
                    if file_path in data.get('files', {}):
                        data['files'][file_path]['is_code_ver'] = True
                        # Clear any previous review flags
                        data['files'][file_path].pop('needs_pseudo_review', None)
                        data['files'][file_path].pop('verification_issues', None)
                        self._save_files_json(data)
                    
                    self._log_to_file(f"Code verification passed for {file_path}")
                    return True
                else:
                    issues = result_text.replace("FAIL:", "").strip()
                    self._log_to_file(f"Code verification failed for {file_path}: {issues}")
                    
                    # NEW: Mark for pseudocode review if verification fails consistently
                    if regen_count >= max_regens:
                        data = self._load_files_json()
                        if file_path in data.get('files', {}):
                            data['files'][file_path]['needs_pseudo_review'] = True
                            data['files'][file_path]['verification_issues'] = issues
                            # Reset flags to trigger regeneration
                            data['files'][file_path]['is_pseudo_gen'] = False
                            data['files'][file_path]['is_pseudo_ver'] = False
                            data['files'][file_path]['is_code_gen'] = False
                            self._save_files_json(data)
                        
                        self._log_to_file(f"Marking {file_path} for pseudocode review due to persistent verification failures")
                        return False
                    
                    # Try to regenerate code if within limits
                    if regen_count < max_regens:
                        self._log_to_file(f"Attempting to regenerate {file_path} (attempt {regen_count + 1}/{max_regens})")
                        
                        success = self._regenerate_code_with_feedback(file_path, issues)
                        if success:
                            regen_count += 1
                            continue
                        else:
                            self._log_to_file(f"Failed to regenerate {file_path}")
                            retries += 1
                    else:
                        self._log_to_file(f"Max regeneration attempts reached for {file_path}")
                        return False
                            
            except Exception as e:
                self._log_to_file(f"Error verifying code for {file_path}: {e}")
                retries += 1
        
        self._log_to_file(f"Max retries reached for code verification of {file_path}")
        return False

                    

    def _debug_files_json(self):
        """Debug method to check files.json content."""
        try:
            data = self._load_files_json()
            
            print(f"\n🔍 DEBUG: files.json content:")
            print(f"  Total files tracked: {len(data.get('files', {}))}")
            
            for filename, info in data.get('files', {}).items():
                print(f"  {filename}:")
                print(f"    - is_pseudo_gen: {info.get('is_pseudo_gen', 'MISSING')}")
                print(f"    - is_pseudo_ver: {info.get('is_pseudo_ver', 'MISSING')}")
                print(f"    - is_code_gen: {info.get('is_code_gen', 'MISSING')}")
                print(f"    - is_code_ver: {info.get('is_code_ver', 'MISSING')}")
                print(f"    - description: {info.get('description', 'MISSING')}")
            
            return data
        except Exception as e:
            print(f"ERROR reading files.json: {e}")
            return None

    def _debug_plan_content(self):
        """Debug method to check what's actually in the plan."""
        try:
            plan_content = direct_read_file(self.plan_file_str)
            
            print(f"\n🔍 DEBUG: Plan file analysis:")
            print(f"  Plan file size: {len(plan_content)} characters")
            
            # Check for file structure sections
            structure_headers = [
                "## COMPLETE FILE STRUCTURE",
                "## FILE STRUCTURE", 
                "## Project Structure",
                "## Files Structure"
            ]
            
            found_header = None
            for header in structure_headers:
                if header in plan_content:
                    found_header = header
                    break
            
            if found_header:
                print(f"  ✅ Found structure section: {found_header}")
                start_idx = plan_content.find(found_header)
                # Show next 500 chars after the header
                preview = plan_content[start_idx:start_idx + 500]
                print(f"  Preview:\n{preview}")
            else:
                print(f"  ❌ No file structure section found!")
                print(f"  Available sections:")
                lines = plan_content.split('\n')
                for line in lines:
                    if line.strip().startswith('##'):
                        print(f"    - {line.strip()}")
            
        except Exception as e:
            print(f"ERROR debugging plan: {e}")

    def get_generation_status(self) -> Dict:
        """Get current generation status."""
        try:
            data = self._load_files_json()
            files = data.get('files', {})
            
            status = {
                'total_files': len(files),
                'pseudo_gen_complete': sum(1 for f in files.values() if f.get('is_pseudo_gen', False)),
                'pseudo_ver_complete': sum(1 for f in files.values() if f.get('is_pseudo_ver', False)),
                'code_gen_complete': sum(1 for f in files.values() if f.get('is_code_gen', False)),
                'code_ver_complete': sum(1 for f in files.values() if f.get('is_code_ver', False)),
                'files': files
            }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Failed to get generation status: {e}")
            return {'error': str(e)}


def main():
    """Main entry point for interactive generation."""
    print("🚀 Enhanced MERN Stack Code Generator with File Tracking")
    print("=" * 60)
    
    sample_descriptions = {
        "calculator": """Build a calculator web app using the MERN stack (MongoDB, Express.js, React, Node.js). 
The app should perform basic arithmetic operations (+, −, ×, ÷), support decimals, negation, clear,
backspace, and equals functions. It should accept both button and keyboard inputs.

The frontend will be built using React with a clean, responsive UI that works on mobile and desktop.
The backend will use Node.js and Express to handle API requests with proper validation and error handling.
Calculation history should be stored in MongoDB and displayed in the UI with options to clear history.

Include API endpoints to fetch, add, and delete calculation history. Implement proper error boundaries,
loading states, and user feedback. Support light/dark mode toggle and ensure accessibility compliance.
Include comprehensive testing setup and deployment configuration.""",
        
        "todo": """Create a Todo List application using the MERN stack with user authentication.
Users should be able to register, login, create todos, mark as complete, edit, delete, and organize
by categories. Include priority levels, due dates, and search functionality.
        
Frontend: React with modern hooks, responsive design, drag-and-drop reordering, real-time updates.
Backend: Express.js with JWT authentication, input validation, rate limiting, and RESTful APIs.
Database: MongoDB with user and todo models, proper indexing for performance.
        
Include features like task filtering, bulk operations, data export, and offline capability.
Implement comprehensive error handling, security measures, and deployment configuration.""",
        
        "blog": """Build a full-featured blog platform using the MERN stack with rich text editing.
Support multiple authors, categories, tags, comments, and social sharing.
        
Features: User authentication, role-based permissions, rich text editor, image uploads,
comment system with moderation, search and filtering, SEO optimization, analytics dashboard.
        
Frontend: React with modern UI components, responsive design, infinite scrolling, syntax highlighting.
Backend: Express.js with file upload handling, email notifications, caching, and API rate limiting.
Database: MongoDB with optimized schemas for posts, users, comments, and analytics.
        
Include admin panel, content management, automated backups, and comprehensive deployment setup."""
    }
    
    print("\nAvailable sample projects:")
    for key, desc in sample_descriptions.items():
        preview = desc.split('.')[0] + "..."
        print(f"  {key}: {preview}")
    
    print("\nOptions:")
    print("1. Use a sample project (enter: calculator, todo, or blog)")
    print("2. Enter custom description")
    print("3. Exit")
    
    choice = input("\nYour choice: ").strip().lower()
    
    if choice in sample_descriptions:
        description = sample_descriptions[choice]
        print(f"\n📝 Using {choice} sample project")
    elif choice == "2" or choice == "custom":
        print("\n📝 Enter your project description (minimum 20 characters):")
        description = input().strip()
        if len(description) < 20:
            print("❌ Description too short. Please provide more details.")
            return
    elif choice == "exit" or choice == "3":
        print("👋 Goodbye!")
        return
    else:
        print("❌ Invalid choice. Using calculator sample.")
        description = sample_descriptions["calculator"]
    
    try:
        generator = MERNCodeGenerator()
        files_dict = generator.generate_mern_code(description)
        
        # Display results
        print("\n" + "="*60)
        print("✅ CODE GENERATION COMPLETED!")
        print("="*60)
        
        print(f"\n📊 Generation Summary:")
        manifest = create_file_manifest(files_dict, str(generator.outputs_dir / "manifest.json"))
        print(f"  Total files: {manifest['total_files']}")
        print(f"  Total lines: {manifest['total_lines']}")
        print(f"  Total size: {manifest['total_size_bytes']} bytes")
        
        print(f"\n📁 Generated files:")
        for filename in sorted(files_dict.keys()):
            print(f"  {filename}")
        
        print(f"\n📂 Files saved to: {generator.outputs_dir}")
        
        # Show generation status
        status = generator.get_generation_status()
        print(f"\n📈 Generation Status:")
        print(f"  Pseudocode generated: {status['pseudo_gen_complete']}/{status['total_files']}")
        print(f"  Pseudocode verified: {status['pseudo_ver_complete']}/{status['total_files']}")
        print(f"  Code generated: {status['code_gen_complete']}/{status['total_files']}")
        print(f"  Code verified: {status['code_ver_complete']}/{status['total_files']}")
        
    except Exception as e:
        print(f"\n❌ Generation failed: {e}")
        print("Check the log file for detailed error information.")
        sys.exit(1)

if __name__ == "__main__":
    main()

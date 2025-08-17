from crewai import Agent
from tools.file_tools import read_file, write_file, list_files

def get_code_gen_agent(llm):
    """Code generation agent using qwen coder for translating pseudocode to production code."""
    return Agent(
        role="Code Generator",
        goal="Translate verified pseudocode into robust, production-ready MERN code",
        backstory="""You are a senior developer expert in MERN stack, focusing on faithful implementation from pseudocode.
        You incorporate best practices for security, error handling, and performance.""",
        llm=llm,
        tools=[],  # No tools needed; content injected in prompt
        verbose=True,
        allow_delegation=False,
        max_iter=50,
        max_execution_time=1800,
        memory=False
    )

def get_code_ver_agent(llm):
    """Code verification agent using qwen coder for checking code against pseudocode."""
    return Agent(
        role="Code Verifier",
        goal="Verify generated code matches its pseudocode in logic, structure, and details",
        backstory="""You are a QA specialist in code auditing, ensuring no deviations from pseudocode blueprints.
        You check for syntax, best practices, and implementation fidelity per file.""",
        llm=llm,
        tools=[],  # Remove tools since we're injecting content
        verbose=True,
        allow_delegation=False,
        max_iter=40,
        max_execution_time=1800,
        memory=True
    )
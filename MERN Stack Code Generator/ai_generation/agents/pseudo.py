from crewai import Agent
from tools.file_tools import read_file, write_file, list_files

def get_pseudo_gen_agent(llm):
    """Pseudocode generation agent using granite for structured pseudocode creation."""
    return Agent(
        role="Pseudocode Generator",
        goal="Generate concise, consistent pseudocode for all files based on the plan",
        backstory="""You are an expert in algorithmic design and pseudocode, specializing in MERN applications.
        You create minimal, structured pseudocode that highlights logic, dependencies, and shared elements accurately.""",
        llm=llm,
        tools=[],  # No tools needed; content injected in prompt
        verbose=True,
        allow_delegation=False,
        max_iter=50,
        max_execution_time=1800,
        memory=True
    )
def get_pseudo_ver_agent(llm):
    """Pseudocode verification agent using granite for holistic checks with batching."""
    return Agent(
        role="Pseudocode Verifier & Auditor",
        goal="Verify all pseudocode for consistency, completeness, and plan adherence using batches and summaries",
        backstory="""You are a senior QA engineer specializing in logical verification of pseudocode in full-stack apps.
        You ensure cross-file dependencies resolve and logic is sound before code generation.""",
        llm=llm,
        tools=[read_file, write_file, list_files],
        verbose=True,
        allow_delegation=False,
        max_iter=40,
        max_execution_time=1800,
        memory=True
    )
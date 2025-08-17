from crewai import Agent
from tools.file_tools import read_file, write_file, list_files

def get_planner_agent(llm):
    """Planner agent for generating the architecture plan and files.json."""
    return Agent(
        role="Senior MERN Stack Architect",
        goal="Create comprehensive, production-ready architectural plans with detailed file tracking structure",
        backstory="""You are a senior full-stack architect with 10+ years of experience in MERN stack development.
        You excel at breaking down complex requirements into detailed, implementable plans with precise file tracking.
        You consider security, scalability, performance, and maintainability in every decision.""",
        llm=llm,
        tools=[],  # No tools needed for pure planning
        verbose=True,
        allow_delegation=False,
        max_iter=35,
        max_execution_time=1200,
        memory=False
    )




from crewai import Agent

def get_sanity_check_agent(llm):
    """Sanity check agent for high-level plan validation."""
    return Agent(
        role="Senior Software Architect",
        goal="Perform high-level sanity checks on project file structure",
        backstory="""You are a senior software architect with 15+ years of experience.
        Your specialty is quickly identifying fundamental flaws in project structures
        that would lead to non-functional applications.""",
        llm=llm,
        tools=[],
        verbose=True,
        allow_delegation=False,
        max_iter=10,
        max_execution_time=300,
        memory=False
    )
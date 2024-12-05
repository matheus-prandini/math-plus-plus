from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from pydantic import BaseModel
from crewai_tools import FileReadTool

# Uncomment the following line to use an example of a custom tool
# from evaluate_knowledge.tools.custom_tool import MyCustomTool

# Check our tools documentations for more information on how to use them
# from crewai_tools import SerperDevTool

class AnalysisReponse(BaseModel):
	common_errors: str
	weaknesses: str
	recommendations: str


@CrewBase
class ReportGeneratorCrew():
	"""ReportGenerator crew"""

	@agent
	def profile_analyzer(self) -> Agent:
		return Agent(
			config=self.agents_config['profile_analyzer'],
			verbose=True,
			llm="gpt-4o"
		)

	@task
	def profile_analysis_task(self) -> Task:
		return Task(
			config=self.tasks_config['profile_analysis_task'],
			output_json=AnalysisReponse,
			output_file="report_result.json"
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the ReportGenerator crew"""
		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)

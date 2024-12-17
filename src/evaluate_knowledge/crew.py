from crewai import LLM, Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from pydantic import BaseModel
from crewai_tools import FileReadTool

# Uncomment the following line to use an example of a custom tool
# from evaluate_knowledge.tools.custom_tool import MyCustomTool

# Check our tools documentations for more information on how to use them
# from crewai_tools import SerperDevTool

class ReviewerResponse(BaseModel):
	question: str
	predicted_math_reasoning: str
	predicted_math_solution: str
	predicted_scratch_reasoning: str
	predicted_scratch_solution: str
	math_feedback: str
	math_score: float
	scratch_feedback: str
	scratch_score: float


@CrewBase
class EvaluateKnowledgeCrew():
	"""EvaluateKnowledge crew"""

	@agent
	def solution_reviewer(self) -> Agent:
		return Agent(
			config=self.agents_config['solution_reviewer'],
			verbose=True,
			llm=LLM(
       			model="gpt-4o",
				temperature=0.0,
				seed=42
			)
		)

	@task
	def reviewer_task(self) -> Task:
		return Task(
			config=self.tasks_config['reviewer_task'],
			output_json=ReviewerResponse
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the EvaluateKnowledge crew"""
		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import FileReadTool
from pydantic import BaseModel


class CreatorItem(BaseModel):
	question: str

class CreatorResponse(BaseModel):
	items: list[CreatorItem]

class MathSolverItem(BaseModel):
	question: str
	math_reasoning: str
	math_solution: str

class MathSolverResponse(BaseModel):
	items: list[MathSolverItem]

class ScratchSolverItem(BaseModel):
	question: str
	math_reasoning: str
	math_solution: str
	scratch_reasoning: str
	scratch_solution: dict

class ScratchSolverResponse(BaseModel):
	items: list[ScratchSolverItem]

@CrewBase
class GenerateInitialDataCrew():
	"""GenerateInitialData crew"""

	@agent
	def creator(self) -> Agent:
		return Agent(
			config=self.agents_config['creator'],
			# tools=[FileReadTool(file_path='/Users/matƒhe/Doutorado/github/math-plus-plus/DESCRITORES DE MATEMÁTICA 9 ANO.docx.txt')],
			verbose=True,
			llm="gpt-4o"
		)

	@agent
	def math_solver(self) -> Agent:
		return Agent(
			config=self.agents_config['math_solver'],
			verbose=True,
			llm="gpt-4o"
		)

	@agent
	def scratch_solver(self) -> Agent:
		return Agent(
			config=self.agents_config['scratch_solver'],
			verbose=True,
			llm="gpt-4o"
		)

	@task
	def creator_task(self) -> Task:
		return Task(
			config=self.tasks_config['creator_task'],
			output_json=CreatorResponse,
		)

	@task
	def math_solver_task(self) -> Task:
		return Task(
			config=self.tasks_config['math_solver_task'],
			context=[self.creator_task()],
			output_json=MathSolverResponse,
		)

	@task
	def scratch_solver_task(self) -> Task:
		return Task(
			config=self.tasks_config['scratch_solver_task'],
			context=[self.math_solver_task()],
			output_json=ScratchSolverResponse,
			output_file='final_result.json'
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the GenerateInitialData crew"""
		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)
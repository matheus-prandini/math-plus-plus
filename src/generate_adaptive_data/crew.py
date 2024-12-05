from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import FileReadTool
from pydantic import BaseModel


class CreatorItem(BaseModel):
	question: str

class CreatorResponse(BaseModel):
	items: list[CreatorItem]

class Item(BaseModel):
	question: str
	reasoning: str
	final_answer: str

class ValidatorResponse(BaseModel):
	items: list[Item]

class ImplementerItem(Item):
	question: str
	reasoning: str
	final_answer: str
	solution: dict

class ImplementerResponse(BaseModel):
	items: list[ImplementerItem]

class ValidationItem(BaseModel):
	question: str
	reasoning: str
	final_answer: str
	solution: dict
	is_valid: bool

class SolverValidationResponse(BaseModel):
	items: list[ValidationItem]

@CrewBase
class GenerateAdaptiveDataCrew():
	"""GenerateAdaptiveData crew"""

	@agent
	def adaptive_creator(self) -> Agent:
		return Agent(
			config=self.agents_config['adaptive_creator'],
			verbose=True,
			llm="gpt-4o"
		)

	@agent
	def validator(self) -> Agent:
		return Agent(
			config=self.agents_config['validator'],
			verbose=True,
			llm="gpt-4o"
		)

	@agent
	def implementer(self) -> Agent:
		return Agent(
			config=self.agents_config['implementer'],
			verbose=True,
			llm="gpt-4o"
		)

	@agent
	def scratch_validator(self) -> Agent:
		return Agent(
			config=self.agents_config['scratch_validator'],
			verbose=True,
			llm="gpt-4o"
		)

	@task
	def adaptive_creator_task(self) -> Task:
		return Task(
			config=self.tasks_config['adaptive_creator_task'],
			output_json=CreatorResponse,
			# output_file='creator.md'
		)

	@task
	def validator_task(self) -> Task:
		return Task(
			config=self.tasks_config['validator_task'],
			context=[self.adaptive_creator_task()],
			output_json=ValidatorResponse,
			# output_file='questions.md'
		)

	@task
	def implementer_task(self) -> Task:
		return Task(
			config=self.tasks_config['implementer_task'],
			context=[self.validator_task()],
			output_json=ImplementerResponse,
			# output_file='scratch_solutions.md'
		)

	@task
	def scratch_validator_task(self) -> Task:
		return Task(
			config=self.tasks_config['scratch_validator_task'],
			context=[self.implementer_task()],
			output_json=SolverValidationResponse,
			output_file='adaptive_result.json'
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the GenerateAdaptiveData crew"""
		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)

import json
from openai import OpenAI
from dotenv import load_dotenv
from generate_initial_data.crew import GenerateInitialDataCrew
from generate_adaptive_data.crew import GenerateAdaptiveDataCrew
from evaluate_knowledge.crew import EvaluateKnowledgeCrew
from report_generator.crew import ReportGeneratorCrew

load_dotenv()


def generate_initial_training_data():
    N_ITERATIONS = 10
    all_data = []
    for _ in range(N_ITERATIONS):
        GenerateInitialDataCrew().crew().kickoff()
        with open("/Users/mathe/Doutorado/github/math-plus-plus/final_result.json", 'r') as file:
            current_data = json.load(file)
            all_data.extend([item for item in current_data["items"]])
    
    training_data = []
    for item in all_data:
        system_message = {"role": "system", "content": "You are a math and Scratch programming assistant. You are a developer and game designer specializing in MIT Scratch. You receive a gamified math problem and your task is to solve the problem and to implement the correct solution as an interactive game using a high level and simplified Scratch's block syntax in JSON format, ensuring it solves the problem, reaches the final answer, and provides a fun and educational experience."}
        user_message = {"role": "user", "content": item['question']}
        math_reasoning = item['math_reasoning']
        math_solution = item['math_solution']
        scratch_reasoning = item['scratch_reasoning']
        scratch_solution = item['scratch_solution']
        assistant_item = {
            "math_reasoning": math_reasoning,
            "math_solution": math_solution,
            "scratch_reasoning": scratch_reasoning,
            "scratch_solution": scratch_solution
            
        }
        assistant_message = {
            "role": "assistant",
            "content": json.dumps(assistant_item, ensure_ascii=False)
        }
        training_data.append({"messages": [system_message, user_message, assistant_message]})

    output_file = "/Users/mathe/Doutorado/github/math-plus-plus/training_data.jsonl"
    with open(output_file, 'w', encoding='utf-8') as file:
        for entry in training_data:
            json.dump(entry, file, ensure_ascii=False)
            file.write('\n')

    print(f"Training data saved to {output_file}")

def generate_initial_test_data():
    N_ITERATIONS = 2
    all_data = []
    for _ in range(N_ITERATIONS):
        GenerateInitialDataCrew().crew().kickoff()
        with open("/Users/mathe/Doutorado/github/math-plus-plus/final_result.json", 'r') as file:
            current_data = json.load(file)
            all_data.extend([item for item in current_data["items"]])

    test_data = []
    for item in all_data:
        system_message = {"role": "system", "content": "You are a math and Scratch programming assistant. You are a developer and game designer specializing in MIT Scratch. You receive a gamified math problem and your task is to solve the problem and to implement the correct solution as an interactive game using a high level and simplified Scratch's block syntax in JSON format, ensuring it solves the problem, reaches the final answer, and provides a fun and educational experience."}
        user_message = {"role": "user", "content": item['question']}
        math_reasoning = item['math_reasoning']
        math_solution = item['math_solution']
        scratch_reasoning = item['scratch_reasoning']
        scratch_solution = item['scratch_solution']
        assistant_item = {
            "math_reasoning": math_reasoning,
            "math_solution": math_solution,
            "scratch_reasoning": scratch_reasoning,
            "scratch_solution": scratch_solution
            
        }
        assistant_message = {
            "role": "assistant",
            "content": json.dumps(assistant_item, ensure_ascii=False)
        }
        test_data.append({"messages": [system_message, user_message, assistant_message]})

    output_file = "/Users/mathe/Doutorado/github/math-plus-plus/test_data.jsonl"
    with open(output_file, 'w', encoding='utf-8') as file:
        for entry in test_data:
            json.dump(entry, file, ensure_ascii=False)
            file.write('\n')

    print(f"Validation data saved to {output_file}")

def generate_adaptive_training_data():
    N_ITERATIONS = 3

    all_data = []
    for _ in range(N_ITERATIONS):
        with open("/Users/mathe/Doutorado/github/math-plus-plus/report_result.json", 'r') as file:
            report_data = json.load(file) 

        for elem in report_data['insights']:
            insight_inputs = {
                "insight": elem
            }
            print(insight_inputs)
            crew = GenerateAdaptiveDataCrew().crew()
            crew.kickoff(inputs=insight_inputs)
            print(f"crew: {crew.usage_metrics}")
            with open("/Users/mathe/Doutorado/github/math-plus-plus/adaptive_result.json", 'r') as file:
                current_data = json.load(file)
                all_data.extend([item for item in current_data["items"]])
    
    training_data = []
    for item in all_data:
        system_message = {"role": "system", "content": "You are a math and Scratch programming assistant. You are a developer and game designer specializing in MIT Scratch. You receive a gamified math problem and your task is to solve the problem and to implement the correct solution as an interactive game using a high level and simplified Scratch's block syntax in JSON format, ensuring it solves the problem, reaches the final answer, and provides a fun and educational experience."}
        user_message = {"role": "user", "content": item['question']}
        math_reasoning = item['math_reasoning']
        math_solution = item['math_solution']
        scratch_reasoning = item['scratch_reasoning']
        scratch_solution = item['scratch_solution']
        assistant_item = {
            "math_reasoning": math_reasoning,
            "math_solution": math_solution,
            "scratch_reasoning": scratch_reasoning,
            "scratch_solution": scratch_solution
            
        }
        assistant_message = {
            "role": "assistant",
            "content": json.dumps(assistant_item, ensure_ascii=False)
        }
        training_data.append({"messages": [system_message, user_message, assistant_message]})

    output_file = "/Users/mathe/Doutorado/github/math-plus-plus/adaptive_training_data.jsonl"
    with open(output_file, 'w', encoding='utf-8') as file:
        for entry in training_data:
            json.dump(entry, file, ensure_ascii=False)
            file.write('\n')

    print(f"Adaptive training data saved to {output_file}")

def get_validation_inference(model_name = "gpt-3.5-turbo-0125"):
    with open("/Users/mathe/Doutorado/github/math-plus-plus/test_data.jsonl", "r") as f:
        current_data = [json.loads(line) for line in f]

    inference_data = []
    for data in current_data:
        system_msg = data["messages"][0]["content"]
        user_msg = data["messages"][1]["content"]

        client = OpenAI()
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system", 
                    "content": system_msg
                },
                {
                    "role": "user",
                    "content": user_msg
                }
            ],
            temperature=0.0
        )
        assistant_msg = response.choices[0].message.content

        inference_data.append(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": system_msg
                    },
                    {
                        "role": "user",
                        "content": user_msg
                    },
                    {
                        "role": "assistant",
                        "content": assistant_msg
                    }
                ]
            }
        )

    output_file = "/Users/mathe/Doutorado/github/math-plus-plus/inference_data.jsonl"
    with open(output_file, 'w', encoding='utf-8') as file:
        for entry in inference_data:
            json.dump(entry, file, ensure_ascii=False)
            file.write('\n')

    print(f"Inference data saved to {output_file}")

def evaluate_inferences():
    with open("/Users/mathe/Doutorado/github/math-plus-plus/test_data.jsonl", "r") as f:
        target_data = [json.loads(line) for line in f]
    with open("/Users/mathe/Doutorado/github/math-plus-plus/inference_data.jsonl", "r") as f:
        predicted_data = [json.loads(line) for line in f]

    assert len(target_data) == len(predicted_data), f"inference_data must contains the same number of elements ({len(predicted_data)}) as test_data ({len(target_data)})"

    evaluation_data = []
    for i in range(len(target_data)):
        predicted_item = predicted_data[i]
        target_item = target_data[i]

        assert predicted_item["messages"][0:2] == target_item["messages"][0:2], f"predicted_item must contains the same system and user message as the target_item"

        evaluation_data.append(
            {
                "original_question": target_item["messages"][1]["content"],
                "target": target_item["messages"][2]["content"],
                "predicted": predicted_item["messages"][2]["content"]
            }
        )

    evaluation_data_converted = []
    for data in evaluation_data:
        evaluation_data_converted.append({
            "evaluation_data": data
        })

    crew = EvaluateKnowledgeCrew().crew()
    results = crew.kickoff_for_each(inputs=evaluation_data_converted)
    print(f"crew: {crew.usage_metrics}")

    score_data = []
    for result in results:
        score_data.append(
            result.model_dump()["json_dict"]
        )
    output_file = "/Users/mathe/Doutorado/github/math-plus-plus/feedback_data.jsonl"
    with open(output_file, 'w', encoding='utf-8') as file:
        for entry in score_data:
            json.dump([entry], file, ensure_ascii=False)
            file.write('\n')

    report_generator_inputs = {
        "feedbacks": score_data
    }
    report_crew = ReportGeneratorCrew().crew()
    report_result = report_crew.kickoff(inputs=report_generator_inputs)
    print(f"report_crew: {report_crew.usage_metrics}")

    print(f"len score data: {score_data}")
    math_scores = [item["math_score"] for item in score_data]
    scratch_scores = [item["scratch_score"] for item in score_data]
    all_scores = [(math_scores[i] + scratch_scores[i]) / 2 for i in range(len(math_scores))]
    mean_score = sum(all_scores) / len(all_scores)

    print(f"Results: \n\n All Math Scores: {math_scores} \n All Scratch Scores: {scratch_scores} \n All Combined Scores: {all_scores} \n Mean Score: {mean_score}")

    print(f"report result: {report_result}")

def create_finetuning_file(filepath):
    client = OpenAI()

    client.files.create(
        file=open(filepath, "rb"),
        purpose="fine-tune"
    )

def run_training_epoch(file_id, model_name = "gpt-3.5-turbo-0125"):
    client = OpenAI()

    client.fine_tuning.jobs.create(
        training_file=file_id,
        model=model_name
    )

# create_finetuning_file(filepath="/Users/mathe/Doutorado/github/math-plus-plus/exp1_o1mini_training_data.jsonl")

# run_training_epoch(
#     file_id="file-7RhFUJEXbhu8f1rE9aXGRY",
# )

# get_validation_inference()

# evaluate_inferences()

# get_validation_inference(model_name="ft:gpt-3.5-turbo-0125:prandini::B2OvLsN9")

evaluate_inferences()

# generate_adaptive_training_data()

# generate_initial_training_data()

# generate_initial_test_data()


# crew = GenerateInitialDataCrew().crew()
# crew.kickoff()
# print(f"crew: {crew}\n\n")
# print(crew.usage_metrics)


def calculate_costs(crew_usage_metrics, model="gpt4o"):
    if model == "gpt4o":
        model_input_price = 2.50
        model_output_price = 10.0
        unit_of_tokens = 1000000
    elif model == "gpt4o-mini":
        model_input_price = 0.15
        model_output_price = 0.6
        unit_of_tokens = 1000000

    prompt_tokens = crew_usage_metrics.get('prompt_tokens')
    completion_tokens = crew_usage_metrics.get('completion_tokens')
    
    input_cost = (prompt_tokens / unit_of_tokens) * model_input_price
    output_cost = (completion_tokens / unit_of_tokens) * model_output_price
    total_cost = input_cost + output_cost
    
    return {
        'total_cost': total_cost,
        'input_cost': input_cost,
        'output_cost': output_cost
    }

## Initial Training Data
    
# crew_usage_metrics = {
#     "total_tokens": 9395,
#     "prompt_tokens": 5569,
#     "completion_tokens": 3826,
#     "successful_requests": 4
# }

# gpt4omini_results = calculate_costs(
#     crew_usage_metrics=crew_usage_metrics,
#     model_input_price=0.15,
#     model_output_price=0.6,
#     unit_of_tokens=1000000
# )

# print(f"gpt4omini results: {gpt4omini_results}")
# gpt4omini results: {'total_cost': 0.00313095, 'input_cost': 0.00083535, 'output_cost': 0.0022956}


# gpt4o_results = calculate_costs(
#     crew_usage_metrics=crew.usage_metrics.model_dump(),
#     model_input_price=2.50,
#     model_output_price=10.0,
#     unit_of_tokens=1000000
# )

# print(f"gpt4o results: {gpt4o_results}")
# gpt4o results: {'total_cost': 0.08081, 'input_cost': 0.02276, 'output_cost': 0.058050000000000004}


## Adaptive Training Data

# crew: total_tokens=8039 prompt_tokens=3933 cached_prompt_tokens=0 completion_tokens=4106 successful_requests=3
# crew_usage_metrics = {
#     "total_tokens": 8039,
#     "prompt_tokens": 3933,
#     "completion_tokens": 4106,
#     "successful_requests": 3
# }
# gpt4o_results = calculate_costs(
#     crew_usage_metrics=crew_usage_metrics,
#     model="gpt4o"
# )

# print(f"gpt4o results: {gpt4o_results}")
# gpt4o results: {'total_cost': 0.0508925, 'input_cost': 0.0098325, 'output_cost': 0.04106}



# crew: total_tokens=12144 prompt_tokens=6713 cached_prompt_tokens=1024 completion_tokens=5431 successful_requests=5
# crew_usage_metrics = {
#     "total_tokens": 12144,
#     "prompt_tokens": 6713,
#     "completion_tokens": 5431,
#     "successful_requests": 5
# }
# gpt4omini_results = calculate_costs(
#     crew_usage_metrics=crew_usage_metrics,
#     model="gpt4o-mini"
# )

# print(f"gpt4o-mini results: {gpt4omini_results}")
# gpt4o-mini results: {'total_cost': 0.00426555, 'input_cost': 0.00100695, 'output_cost': 0.0032586}


## Knowledge Evaluator

# crew: total_tokens=16975 prompt_tokens=12610 cached_prompt_tokens=8192 completion_tokens=4365 successful_requests=10
# report_crew: total_tokens=1465 prompt_tokens=1012 cached_prompt_tokens=0 completion_tokens=453 successful_requests=1

# crew_usage_metrics = {
#     "total_tokens": 18840,
#     "prompt_tokens": 13622,
#     "completion_tokens": 4818,
#     "successful_requests": 11
# }
# gpt4o_results = calculate_costs(
#     crew_usage_metrics=crew_usage_metrics,
#     model="gpt4o"
# )

# print(f"gpt4o results: {gpt4o_results}")
# gpt4o results: {'total_cost': 0.082235, 'input_cost': 0.034055, 'output_cost': 0.04818}


# crew: total_tokens=16387 prompt_tokens=12610 cached_prompt_tokens=9216 completion_tokens=3777 successful_requests=10
# report_crew: total_tokens=1532 prompt_tokens=1012 cached_prompt_tokens=0 completion_tokens=520 successful_requests=1

# crew_usage_metrics = {
#     "total_tokens": 17919,
#     "prompt_tokens": 11598,
#     "completion_tokens": 4297,
#     "successful_requests": 11
# }
# gpt4o_results = calculate_costs(
#     crew_usage_metrics=crew_usage_metrics,
#     model="gpt4o-mini"
# )

# print(f"gpt4o-mini results: {gpt4o_results}")
# gpt4o-mini results: {'total_cost': 0.0043178999999999995, 'input_cost': 0.0017397, 'output_cost': 0.0025781999999999997}


## Total

# gpt4o initial data:  0,8081$ (10 iterations)
# gpt4o-mini initial data:  0,03131$ (10 iterations)

# gpt4o adaptive data:  0,1526775$ (3 iterations)
# gpt4o-mini initial data:  0,01279665$ (3 iterations)

# gpt4o knowledge evaluator:  0,082235$ (1 iteration)
# gpt4o-mini knowledge evaluator:  0,0043179$ (1 iteration)

##

# gpt4o Initial Data + 1 loop: 1,129345$
# gpt4o-mini Initial Data + 1 loop: 0,05274245$

##

import json
from openai import OpenAI
from dotenv import load_dotenv
from generate_initial_data_2.crew import GenerateInitialDataCrew
from generate_adaptive_data_2.crew import GenerateAdaptiveDataCrew
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
        final_answer_str = item['final_answer']
        scratch_json_str = json.dumps(item['solution'], ensure_ascii=False)
        assistant_message = {
            "role": "assistant",
            "content": f"Final Answer: {final_answer_str} \n Scratch Solution: Here's the Scratch solution as a JSON: {scratch_json_str}"
        }
        training_data.append({"messages": [system_message, user_message, assistant_message]})

    output_file = "/Users/mathe/Doutorado/github/math-plus-plus/training_data.jsonl"
    with open(output_file, 'w', encoding='utf-8') as file:
        for entry in training_data:
            json.dump(entry, file, ensure_ascii=False)
            file.write('\n')

    print(f"Training data saved to {output_file}")

def generate_initial_training_data_2():
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
        final_answer_str = item['math_solution']
        scratch_json_str = json.dumps(item['scratch_solution'], ensure_ascii=False)
        assistant_message = {
            "role": "assistant",
            "content": f"Final Answer: {final_answer_str} \n Scratch Solution: Here's the Scratch solution as a JSON: {scratch_json_str}"
        }
        training_data.append({"messages": [system_message, user_message, assistant_message]})

    output_file = "/Users/mathe/Doutorado/github/math-plus-plus/training_data.jsonl"
    with open(output_file, 'w', encoding='utf-8') as file:
        for entry in training_data:
            json.dump(entry, file, ensure_ascii=False)
            file.write('\n')

    print(f"Training data saved to {output_file}")

def generate_initial_validation_data():
    N_ITERATIONS = 2
    all_data = []
    for _ in range(N_ITERATIONS):
        GenerateInitialDataCrew().crew().kickoff()
        with open("/Users/mathe/Doutorado/github/math-plus-plus/final_result.json", 'r') as file:
            current_data = json.load(file)
            all_data.extend([item for item in current_data["items"]])
    
    validation_data = []
    for item in all_data:
        system_message = {"role": "system", "content": "You are a math and Scratch programming assistant. You are a developer and game designer specializing in MIT Scratch. You receive a gamified math problem and your task is to solve the problem and to implement the correct solution as an interactive game using a high level and simplified Scratch's block syntax in JSON format, ensuring it solves the problem, reaches the final answer, and provides a fun and educational experience."}
        user_message = {"role": "user", "content": item['question']}
        final_answer_str = item['final_answer']
        scratch_json_str = json.dumps(item['solution'], ensure_ascii=False)
        assistant_message = {
            "role": "assistant",
            "content": f"Final Answer: {final_answer_str} \n Scratch Solution: Here's the Scratch solution as a JSON: {scratch_json_str}"
        }
        validation_data.append({"messages": [system_message, user_message, assistant_message]})

    output_file = "/Users/mathe/Doutorado/github/math-plus-plus/validation_data.jsonl"
    with open(output_file, 'w', encoding='utf-8') as file:
        for entry in validation_data:
            json.dump(entry, file, ensure_ascii=False)
            file.write('\n')

    print(f"Validation data saved to {output_file}")

def generate_initial_validation_data_2():
    N_ITERATIONS = 2
    all_data = []
    for _ in range(N_ITERATIONS):
        GenerateInitialDataCrew().crew().kickoff()
        with open("/Users/mathe/Doutorado/github/math-plus-plus/final_result.json", 'r') as file:
            current_data = json.load(file)
            all_data.extend([item for item in current_data["items"]])

    validation_data = []
    for item in all_data:
        system_message = {"role": "system", "content": "You are a math and Scratch programming assistant. You are a developer and game designer specializing in MIT Scratch. You receive a gamified math problem and your task is to solve the problem and to implement the correct solution as an interactive game using a high level and simplified Scratch's block syntax in JSON format, ensuring it solves the problem, reaches the final answer, and provides a fun and educational experience."}
        user_message = {"role": "user", "content": item['question']}
        final_answer_str = item['math_solution']
        scratch_json_str = json.dumps(item['scratch_solution'], ensure_ascii=False)
        assistant_message = {
            "role": "assistant",
            "content": f"Final Answer: {final_answer_str} \n Scratch Solution: Here's the Scratch solution as a JSON: {scratch_json_str}"
        }
        validation_data.append({"messages": [system_message, user_message, assistant_message]})

    output_file = "/Users/mathe/Doutorado/github/math-plus-plus/validation_data.jsonl"
    with open(output_file, 'w', encoding='utf-8') as file:
        for entry in validation_data:
            json.dump(entry, file, ensure_ascii=False)
            file.write('\n')

    print(f"Validation data saved to {output_file}")

def generate_adaptive_training_data():
    N_ITERATIONS = 5

    # with open("/Users/mathe/Doutorado/github/math-plus-plus/report_result.json", 'r') as file:
    #     report_data = json.load(file)

    # inputs = {
    #     "report": report_data
    # }

    all_data = []
    for _ in range(N_ITERATIONS):
        GenerateAdaptiveDataCrew().crew().kickoff()
        with open("/Users/mathe/Doutorado/github/math-plus-plus/adaptive_result.json", 'r') as file:
            current_data = json.load(file)
            all_data.extend([item for item in current_data["items"]])
    
    training_data = []
    for item in all_data:
        system_message = {"role": "system", "content": "You are a math and Scratch programming assistant. You are a developer and game designer specializing in MIT Scratch. You receive a gamified math problem and your task is to solve the problem and to implement the correct solution as an interactive game using a high level and simplified Scratch's block syntax in JSON format, ensuring it solves the problem, reaches the final answer, and provides a fun and educational experience."}
        user_message = {"role": "user", "content": item['question']}
        final_answer_str = item['math_solution']
        scratch_json_str = json.dumps(item['scratch_solution'], ensure_ascii=False)
        assistant_message = {
            "role": "assistant",
            "content": f"Final Answer: {final_answer_str} \n Scratch Solution: Here's the Scratch solution as a JSON: {scratch_json_str}"
        }
        training_data.append({"messages": [system_message, user_message, assistant_message]})

    output_file = "/Users/mathe/Doutorado/github/math-plus-plus/adaptive_training_data.jsonl"
    with open(output_file, 'w', encoding='utf-8') as file:
        for entry in training_data:
            json.dump(entry, file, ensure_ascii=False)
            file.write('\n')

    print(f"Training data saved to {output_file}")

def get_validation_inference(model_name = "gpt-3.5-turbo-0125"):
    with open("/Users/mathe/Doutorado/github/math-plus-plus/validation_data.jsonl", "r") as f:
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
    with open("/Users/mathe/Doutorado/github/math-plus-plus/validation_data.jsonl", "r") as f:
        target_data = [json.loads(line) for line in f]
    with open("/Users/mathe/Doutorado/github/math-plus-plus/inference_data.jsonl", "r") as f:
        predicted_data = [json.loads(line) for line in f]

    assert len(target_data) == len(predicted_data), f"inference_data must contains the same number of elements ({len(predicted_data)}) as validation_data ({len(target_data)})"

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

    results = EvaluateKnowledgeCrew().crew().kickoff_for_each(inputs=evaluation_data_converted)

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
    ReportGeneratorCrew().crew().kickoff(inputs=report_generator_inputs)

    print(f"len score data: {score_data}")
    all_scores = [item["score"] for item in score_data]
    mean_score = sum(all_scores) / len(all_scores)

    print(f"Results: \n\n All Scores: {all_scores} \n Mean Score: {mean_score}")

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

# create_finetuning_file(filepath="/Users/mathe/Doutorado/github/math-plus-plus/extended_training_data.jsonl")

# run_training_epoch(
#     file_id="file-YV4DgmECscsEoFVgnvBXEr",
#     model_name="ft:gpt-3.5-turbo-0125:neospace::AainbYqn"
# )

# run_training_epoch(
#     file_id="file-YV4DgmECscsEoFVgnvBXEr",
# )

# get_validation_inference(model_name="ft:gpt-3.5-turbo-0125:neospace::AajGUS4p")

evaluate_inferences()

# generate_adaptive_training_data()
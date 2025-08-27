import json
import re
import pandas as pd
import os
import subprocess
import sys
from openai import OpenAI
from generation_models import message_construct_func, message_construct_llama_func, GPT_response, count_total_tokens, \
    extract_code, extract_and_check, LLM_answer_code_checker, paraphrase_with_GPT4, log_run_info
import random
import math
import json
from typing import List, Tuple, Dict
import time
import numpy as np
import ast
from prompt import *
from argparse import ArgumentParser
from symbolic_code_check import analyze_code_and_explain
import random
import string
from LLaMA_Factory.src.llamafactory.chat.chat_model import run_response

### To do, add tasks to import related functions
from Logic_Game_func import load_task_dataset, verify_solution_func_gather, multi_round_answer_sampling

def save_file_func_CodeSteer(save_code_dir, response_list, user_prompt_list, question, user_prompt_list_save, response_list_save):
    data = {
        'question': question,
        'response_list': response_list,
        'user_prompt_list': user_prompt_list,
        'user_prompt_list_save': user_prompt_list_save,
        'response_list_save': response_list_save
    }

    output_file = os.path.join(save_code_dir, 'conversation_data.json')

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Data successfully saved to {output_file}")
    except Exception as e:
        print(f"Error saving data: {str(e)}")

def save_file_func_baselines(save_code_dir, response_list, user_prompt_list, question, system_message):
    data = {
        'question': question,
        'response_list': response_list,
        'user_prompt_list': user_prompt_list,
        'system_message': system_message
    }

    output_file = os.path.join(save_code_dir, 'conversation_data.json')

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Data successfully saved to {output_file}")
    except Exception as e:
        print(f"Error saving data: {str(e)}")

def answer_sampling_func(model_name, args_path, user_prompt_list, response_list, code_interpreter=False):
    if model_name in ['o3-mini-2025-01-31', 'o1', "o1-preview", 'o1-mini', 'gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo',
                      "claude-3-5-sonnet-20241022", "claude-3-sonnet-20240229",
                      "claude-3-opus-20240229", "claude-3-haiku-20240307", 'open-mixtral-8x7b', "mistral-large-latest",
                      'DeepSeek-R1', 'DeepSeek-V3',
                      'R1_CI_14B_NIM', 'R1_CI_7B_NIM', 'R1_CI_3B_NIM', 'Qwen_NIM']:
        response = GPT_response('', user_prompt_list[-1], model_name=model_name, code_interpreter=code_interpreter,
                                user_prompt_list=user_prompt_list, response_total_list=response_list, logprobs=False)
    else:
        messages = message_construct_llama_func(user_prompt_list, response_list)
        response = run_response(messages, args_path)
    return response

def code_interpreter_func(save_code_dir, response, round_num):
    code_block_list = extract_code(response)
    if not code_block_list:
        return '', ''

    code = code_block_list[0]
    try:
        # Launch Python, read the script from stdin ("-")
        result = subprocess.run(
            ["python3", "-"],
            input=code,
            capture_output=True,
            text=True,
            timeout=60
        )
        output = result.stdout
        errors = result.stderr
    except subprocess.TimeoutExpired as e:
        try:
            output = e.stdout or ""
            errors = (e.stderr or "") + (
                f"\nTimeoutExpired: Command '{e.cmd}' timed out after {e.timeout} seconds"
            )
        except:
            output = ""
            errors = f"TimeoutExpired: Command timed out after 60 seconds"
    return output[:1000], errors[:1000]

def run_logic_game_baselines(task_name, gather_save_input_dir, model_name, baseline_method_name, args_path, start_index, max_sample_num, max_round_num, train_test = 'together'):
    log_file_result = os.path.join(gather_save_input_dir, f"acc_result_log_{model_name}_{train_test}.txt")

    task_exists = False
    # Open and read the log file
    try:
        with open(log_file_result, 'r') as file:
            content = file.read()

        # Check if the file is empty
        if not content.strip():
            print(f"The file {log_file_result} is empty.")
            task_exists = False
        else:
            # Check if task_name is in the content
            task_exists = task_name in content

            if task_exists:
                print(f"Task '{task_name}' found in the log file.")
            else:
                print(f"Task '{task_name}' not found in the log file.")

    except FileNotFoundError:
        print(f"Error: File {log_file_result} not found.")
    except Exception as e:
        print(f"Error reading file: {e}")

    if task_exists:
        print(f"Task '{task_name}' has already been completed. Skipping...")
        return

    print('\n' + '*'*30)
    total_sample_num = 0
    total_correct_num = 0

    # Load dataset
    solution_list, question_list, target_list, puzzles, solution_data_list, question_constrained_list, question_matrix_list, number_list, word_list, letter_list, save_input_dir = load_task_dataset(
        task_name, model_name)

    print(f'length of number_list: {len(number_list)}')
    base_save_code_dir = save_input_dir + f'/result_{task_name}_{baseline_method_name}_{model_name}'
    if not os.path.exists(base_save_code_dir):
        os.makedirs(base_save_code_dir)

    ### Remain unchanged
    question_num_total = len(question_list)

    for i in range(min(start_index, question_num_total-max_sample_num), max_sample_num + min(start_index, question_num_total-max_sample_num)):
        solution = ''; question = ''; number_list_item = ''; target = ''; word = ''; letter = ''
        question = question_list[i]
        if len(solution_list) > 0:
            solution = solution_list[i]
        if len(number_list) > 0:
            number_list_item = number_list[i]
        if len(target_list) > 0:
            target = target_list[i]
        if len(word_list) > 0:
            word = word_list[i]
        if len(letter_list) > 0:
            letter = letter_list[i]

        total_sample_num += 1

        save_code_dir = os.path.join(base_save_code_dir, f"Test_sample_{i}/")
        if not os.path.exists(save_code_dir):
            os.makedirs(save_code_dir)

        print('-------###-------###-------###-------')
        print(f"\nTest_sample_{i}, total: {min(len(solution_list), max_sample_num)}/")

        system_message = ''
        response_list = []
        code_usage_round_num = 0
        code_interpreter = False
        if baseline_method_name in ['1_only_ques', 'All_code_CoT', 'All_text', 'Code_Interpreter']:
            if baseline_method_name == '1_only_ques':
                user_prompt_list = [question]
            elif baseline_method_name == 'All_code_CoT':
                user_prompt_list = [with_COT_code_output_prompt + question]
            elif baseline_method_name == 'All_text':
                user_prompt_list = [text_output_prompt + question]
            elif baseline_method_name == 'Code_Interpreter':
                user_prompt_list = [question]
                code_interpreter = True

            response = answer_sampling_func(model_name, args_path, user_prompt_list, response_list, code_interpreter)
            response_list.append(response)

        elif baseline_method_name == 'R1_code_interpreter_test':
            for round_num in range(max_round_num):
                if round_num == 0:
                    question = R1_code_interpreter_data_syn_prompt1 + 'question: ' + question + '\n'
                    user_prompt_list = [question]
                    response = answer_sampling_func(model_name, args_path, user_prompt_list, response_list)
                    response_list.append(response)
                    print(f"\n########## Round {round_num} ##########\n response: {response}")
                else:
                    response = answer_sampling_func(model_name, args_path, user_prompt_list, response_list)
                    response_list.append(response)
                    print(f"\n########## Round {round_num} ##########\n response: {response}")

                output, errors = code_interpreter_func(save_code_dir, response, round_num)
                if len(output) > 0 or len(errors) > 0:
                    code_usage_round_num += 1
                    user_prompt_list.append(f'Code output: {output}\nErrors: {errors}')
                else:
                    break
        elif baseline_method_name == 'R1_code_interpreter_data_syn_1':
            for round_num in range(max_round_num):
                if round_num == 0:
                    user_prompt_list = [R1_code_interpreter_data_syn_prompt1 + question]
                    response = answer_sampling_func(model_name, args_path, user_prompt_list, response_list)
                    response_list.append(response)
                    print(f"\n########## Round {round_num} ##########\n response: {response}")
                else:
                    response = answer_sampling_func(model_name, args_path, user_prompt_list, response_list)
                    response_list.append(response)
                    print(f"\n########## Round {round_num} ##########\n response: {response}")

                output, errors = code_interpreter_func(save_code_dir, response, round_num)
                if len(output) > 0 or len(errors) > 0:
                    code_usage_round_num += 1
                    user_prompt_list.append(f'Code output: {output}\nErrors: {errors}')
                else:
                    break
        elif baseline_method_name == 'R1_code_interpreter_data_syn_code':
            for round_num in range(max_round_num):
                if round_num == 0:
                    user_prompt_list = [R1_code_interpreter_data_syn_prompt1_code + question]
                    response = answer_sampling_func(model_name, args_path, user_prompt_list, response_list)
                    response_list.append(response)
                    print(f"\n########## Round {round_num} ##########\n response: {response}")
                else:
                    response = answer_sampling_func(model_name, args_path, user_prompt_list, response_list)
                    response_list.append(response)
                    print(f"\n########## Round {round_num} ##########\n response: {response}")

                output, errors = code_interpreter_func(save_code_dir, response, round_num)
                if len(output) > 0 or len(errors) > 0:
                    code_usage_round_num += 1
                    user_prompt_list.append(f'Code output: {output}\nErrors: {errors}')
                else:
                    break
        elif baseline_method_name == 'R1_code_interpreter_data_syn_hint':
            for round_num in range(max_round_num):
                if round_num == 0:
                    user_prompt_list = [question]
                    user_prompt_list_input = [R1_code_interpreter_data_syn_prompt2 + question]
                    response = answer_sampling_func(model_name, args_path, user_prompt_list_input, response_list)
                    response_list.append(response)
                    print(f"\n########## Round {round_num} ##########\n response: {response}")
                else:
                    response = answer_sampling_func(model_name, args_path, user_prompt_list_input, response_list)
                    response_list.append(response)
                    print(f"\n########## Round {round_num} ##########\n response: {response}")

                output, errors = code_interpreter_func(save_code_dir, response, round_num)
                if len(output) > 0 or len(errors) > 0:
                    code_usage_round_num += 1
                    user_prompt_list_input.append(f'{R1_code_interpreter_data_syn_intermediate_step}Code output: {output}\nErrors: {errors}')
                    user_prompt_list.append(f'Code output: {output}\nErrors: {errors}')
                else:
                    break
        elif baseline_method_name == 'CodeSteer':
            CodeSteer_output_prompt_guidance_list = [];
            CodeSteer_input_prompt_list = [code_text_choice_prompt + question];
            CodeSteer_input_prompt_training_list = [code_text_choice_prompt + question]

            starting_prompt_choice = with_COT_code_output_prompt

            print(f'Starting prompt choice: {starting_prompt_choice}')
            user_prompt_list = [starting_prompt_choice + question]
            CodeSteer_output_prompt_guidance_list.append(starting_prompt_choice)
            response = answer_sampling_func(model_name, args_path, user_prompt_list, response_list)
            response_list.append(response)

            ############ Further rounds of guidance ############
            for round_num in range(max_round_num):
                code_block_list = extract_code(response)
                if len(code_block_list) > 0:
                    code_complexity_summary, code_complexity_score = analyze_code_and_explain(code_block_list[0])
                    if code_complexity_score <= 2:
                        code_complexity_summary += '\nThe generated code may not be complex enough to carry out symbolic computing for solving the task.'
                    with open(save_code_dir + f"/code_1_{round_num}.py", "w") as f:
                        f.write(code_block_list[0])

                    try:
                        result = subprocess.run(
                            ["python3", save_code_dir + f"/code_1_{round_num}.py"],
                            capture_output=True, text=True, timeout=60
                        )
                        output = result.stdout
                        errors = result.stderr
                    except subprocess.TimeoutExpired as e:
                        output = e.stdout if e.stdout else ""
                        errors = e.stderr if e.stderr else ""
                        errors += f"\nTimeoutExpired: Command '{e.cmd}' timed out after {e.timeout} seconds"
                    output = output[:1000]
                    errors = errors[:1000]
                    response = response + f'\nThe execution result from the generated code is:\noutput: {output}, errors: {errors}'

                    if isinstance(output, str):
                        if count_total_tokens([output + errors], []) > 8000:
                            response = response + f'\nThe execution result from the generated code is too long to be displayed.'
                        else:
                            response = response + f'\nThe execution result from the generated code is:\noutput: {output}, errors: {errors}'
                    else:
                        response = response + f'\nThe execution result from the generated code is:\nerrors: {errors}'

                check_code_saving_path = save_code_dir + f"/check_code_1_{round_num}.py"
                check_result = LLM_answer_code_checker(question, response, check_code_saving_path, model_name=model_name)

                CodeSteer_input_prompt_head = f'''{decision_prompt_complex_code} {question}\n'''
                if len(code_block_list) > 0:
                    print('\n############True#############\n')
                    CodeSteer_input_prompt = f'''The response from TaskLLM is: {response}\n\nThe feedback from the checking agent is:\n{check_result}\n\nThe summary of generated code complexity is: {code_complexity_summary}\n\n''' + \
                                             f'''The final returned guidance prompt should be of the format <<<guidance prompt content>>>.'''

                else:
                    CodeSteer_input_prompt = f'''The response from TaskLLM is: {response}\n\nThe feedback from the checking agent is:\n{check_result}\n\n''' + \
                                             f'''The final returned guidance prompt should be of the format <<<guidance prompt content>>>.'''

                CodeSteer_input_prompt_total = CodeSteer_input_prompt_head + CodeSteer_input_prompt
                CodeSteer_input_prompt_list.append(CodeSteer_input_prompt_total)
                CodeSteer_input_prompt_training_list.append(CodeSteer_input_prompt)
                response_text = GPT_response("", '', model_name=model_name, code_interpreter=False,
                                             user_prompt_list=CodeSteer_input_prompt_list,
                                             response_total_list=CodeSteer_output_prompt_guidance_list,
                                             logprobs=False)
                matches = re.findall(r'<<<(.*?)>>>', response_text, re.DOTALL)
                guidance_prompt = matches[-1] if matches else response_text

                print(f'\nGuidance prompt_{round_num + 1}: {guidance_prompt}\n')

                CodeSteer_output_prompt_guidance_list.append(guidance_prompt)
                if '<<<Code>>>' in guidance_prompt:
                    guidance_prompt = with_COT_code_output_prompt
                elif '<<<Text>>>' in guidance_prompt:
                    guidance_prompt = text_output_prompt
                elif '<<<Return Answer>>>' in guidance_prompt or 'Return Answer' in guidance_prompt or '<<<Terminate>>>' in guidance_prompt or 'Terminate' in guidance_prompt:
                    break
                user_prompt_list.append(guidance_prompt)
                response = answer_sampling_func(model_name, args_path, user_prompt_list, response_list)

                response_list.append(response)
        elif baseline_method_name == 'CodeSteer_wo_self-answer_checker':
            CodeSteer_output_prompt_guidance_list = [];
            CodeSteer_input_prompt_list = [code_text_choice_prompt + question];
            CodeSteer_input_prompt_training_list = [code_text_choice_prompt + question]

            starting_prompt_choice = with_COT_code_output_prompt

            print(f'Starting prompt choice: {starting_prompt_choice}')
            user_prompt_list = [starting_prompt_choice + question]
            user_prompt_list_save = [question]
            CodeSteer_output_prompt_guidance_list.append(starting_prompt_choice)
            response = answer_sampling_func(model_name, args_path, user_prompt_list, response_list)
            response_list.append(response)
            response_list_save = [response]
            response_save = ''

            ############ Further rounds of guidance ############
            for round_num in range(max_round_num):
                code_block_list = extract_code(response)
                if len(code_block_list) > 0:
                    code_complexity_summary, code_complexity_score = analyze_code_and_explain(code_block_list[0])
                    if code_complexity_score <= 2:
                        code_complexity_summary += '\nThe generated code may not be complex enough to carry out symbolic computing for solving the task.'
                    with open(save_code_dir + f"/code_1_{round_num}.py", "w") as f:
                        f.write(code_block_list[0])

                    try:
                        result = subprocess.run(
                            ["python3", save_code_dir + f"/code_1_{round_num}.py"],
                            capture_output=True, text=True, timeout=60
                        )
                        output = result.stdout
                        errors = result.stderr
                    except subprocess.TimeoutExpired as e:
                        output = e.stdout if e.stdout else ""
                        errors = e.stderr if e.stderr else ""
                        errors += f"\nTimeoutExpired: Command '{e.cmd}' timed out after {e.timeout} seconds"
                    output = output[:1000]
                    errors = errors[:1000]
                    response = response + f'\nThe execution result from the generated code is:\noutput: {output}, errors: {errors}'

                    if isinstance(output, str):
                        if count_total_tokens([output + errors], []) > 8000:
                            response = response + f'\nThe execution result from the generated code is too long to be displayed.'
                        else:
                            response = response + f'\nThe execution result from the generated code is:\noutput: {output}, errors: {errors}'
                    else:
                        response = response + f'\nThe execution result from the generated code is:\nerrors: {errors}'

                CodeSteer_input_prompt_head = f'''{decision_prompt_complex_code} {question}\n'''
                if len(code_block_list) > 0:
                    print('\n############True#############\n')
                    CodeSteer_input_prompt = f'''The response from TaskLLM is: {response}\n\nThe summary of generated code complexity is: {code_complexity_summary}\n\n''' + \
                                             f'''The final returned guidance prompt should be of the format <<<guidance prompt content>>>.'''

                else:
                    CodeSteer_input_prompt = f'''The response from TaskLLM is: {response}\n\n''' + \
                                             f'''The final returned guidance prompt should be of the format <<<guidance prompt content>>>.'''

                CodeSteer_input_prompt_total = CodeSteer_input_prompt_head + CodeSteer_input_prompt
                CodeSteer_input_prompt_list.append(CodeSteer_input_prompt_total)
                CodeSteer_input_prompt_training_list.append(CodeSteer_input_prompt)
                response_text = GPT_response("", '', model_name=model_name, code_interpreter=False,
                                             user_prompt_list=CodeSteer_input_prompt_list,
                                             response_total_list=CodeSteer_output_prompt_guidance_list,
                                             logprobs=False)
                matches = re.findall(r'<<<(.*?)>>>', response_text, re.DOTALL)
                guidance_prompt = matches[-1] if matches else response_text

                print(f'\nGuidance prompt_{round_num + 1}: {guidance_prompt}\n')

                CodeSteer_output_prompt_guidance_list.append(guidance_prompt)
                if '<<<Code>>>' in guidance_prompt:
                    guidance_prompt = with_COT_code_output_prompt
                elif '<<<Text>>>' in guidance_prompt:
                    guidance_prompt = text_output_prompt
                elif '<<<Return Answer>>>' in guidance_prompt or 'Return Answer' in guidance_prompt or '<<<Terminate>>>' in guidance_prompt or 'Terminate' in guidance_prompt:
                    break
                user_prompt_list.append(guidance_prompt)
                response = answer_sampling_func(model_name, args_path, user_prompt_list, response_list)
                if len(code_block_list) > 0:
                    user_prompt_list_save.append(f'Code output: {output}\nErrors: {errors}')
                    response_list_save.append(response_save + f'{guidance_prompt}\n\n{response}\n\n')
                    response_save = ''
                else:
                    response_save += f'{guidance_prompt}\n\n{response}\n\n'

                response_list.append(response)


        if 'CodeSteer' in baseline_method_name:
            save_file_func_CodeSteer(save_code_dir, response_list, user_prompt_list, question, user_prompt_list_save, response_list_save)
        else:
            save_file_func_baselines(save_code_dir, response_list, user_prompt_list, question, system_message)

        response = response_list[-1]
        response = response[:10000]
        token_len_response = count_total_tokens([response], [])
        print(f'token_len_response: {token_len_response}')
        True_false_result_1, True_false_result_2 = verify_solution_func_gather(i, task_name, response,
                                                                               save_code_dir, question, solution,
                                                                               target, puzzles, solution_data_list,
                                                                               solution_list, question_constrained_list,
                                                                               question_matrix_list, number_list_item, word, letter)

        ### Remain unchanged
        if True_false_result_1 == False and True_false_result_2 == False:
            print('False')
            with open(save_code_dir + f"/success_failure.txt", "w") as f:
                f.write('False')
        else:
            print('True')
            with open(save_code_dir + f"/success_failure.txt", "w") as f:
                f.write('True')
            total_correct_num += 1

        print(f'\ntotal_sample_num: {total_sample_num}')
        print(f'total_correct_num: {total_correct_num}\n')
        print(f'Code usage round number: {code_usage_round_num}\n')

        with open(base_save_code_dir + f"/acc_result_log_{model_name}_{baseline_method_name}_{train_test}.txt", "w") as f:
            f.write(f"correct/all:{total_correct_num}/{total_sample_num}\n")

    run_info = f"CodeSteer, {task_name}, {baseline_method_name}, {model_name}\n"
    run_info_result = f'correct/all:{total_correct_num}/{total_sample_num}\n'
    log_file_result = os.path.join(gather_save_input_dir, f"acc_result_log_{model_name}_{train_test}.txt")
    log_run_info(log_file_result, run_info + run_info_result)

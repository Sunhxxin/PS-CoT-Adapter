'''
Adapted from https://github.com/lupantech/ScienceQA
'''

import os
import json
import argparse
import warnings
import pandas as pd
from sentence_transformers import SentenceTransformer
from evaluations import caculate_bleu, caculate_rouge, caculate_similariry

warnings.filterwarnings('ignore')

def get_acc_with_contion(res_pd, key, values):
    if isinstance(values, list):
        total_pd = res_pd[res_pd[key].isin(values)]
    else:
        total_pd = res_pd[res_pd[key] == values]
    correct_pd = total_pd[total_pd['true_false'] == True]
    acc = "{:.2f}".format(len(correct_pd) / len(total_pd) * 100)
    return acc

def get_acc_num_with_contion(res_pd, key, values):
    if isinstance(values, list):
        total_pd = res_pd[res_pd[key].isin(values)]
    else:
        total_pd = res_pd[res_pd[key] == values]
    correct_pd = total_pd[total_pd['true_false'] == True]
    acc_num = "{:.2f}".format((len(total_pd)-len(correct_pd)) / 100 * 100)
    return acc_num

def get_scores(result_data, rationale_data, results_reference, data_file):
    # read result file
    results = result_data
    num = len(results)
    #assert num == 4241
    print("number of questions:", num)

    # read data file
    sqa_data = json.load(open(data_file))

    # construct pandas data
    sqa_pd = pd.DataFrame(sqa_data).T
    res_pd = sqa_pd[sqa_pd['split'] == 'test']  # test set

    # update data
    for index, row in res_pd.iterrows():

        res_pd.loc[index, 'no_context'] = True if (not row['hint'] and not row['image']) else False
        res_pd.loc[index, 'has_text'] = True if row['hint'] else False
        res_pd.loc[index, 'has_image'] = True if row['image'] else False
        res_pd.loc[index, 'has_text_image'] = True if (row['hint'] and row['image']) else False

        label = row['answer']
        pred = int(results[index])
        res_pd.loc[index, 'pred'] = pred
        res_pd.loc[index, 'true_false'] = (label == pred)

    # accuracy scores
    acc_average = len(res_pd[res_pd['true_false'] == True]) / num * 100
    #assert result_file.split('_')[-1] == "{:.3f}.json".format(acc_average)


    # Group by 'skill' for further analysis
    grouped_by_skill = res_pd.groupby('skill')

    # ...

    # Calculate accuracy for each skill category
    scores_by_skill = {}
    for skill, skill_group in grouped_by_skill:
        acc_num_skill = get_acc_num_with_contion(skill_group, 'skill', skill)
        scores_by_skill[f'acc_{skill.lower()}'] = acc_num_skill

    # Convert accuracy strings to float
    float_values = [float(value) for value in scores_by_skill.values()]

    # Calculate min and max accuracy
    min_acc = min(float_values)
    max_acc = max(float_values)

    # Normalize the accuracy scores
    normalized_scores = {key: (float(value) - min_acc) / (max_acc - min_acc) for key, value in scores_by_skill.items()}

    # Find the three smallest normalized scores
    top_3_skills = sorted(normalized_scores, key=normalized_scores.get, reverse=True)[:3]

    # rationale quality

    ## BLEU
    bleu1 = caculate_bleu(rationale_data, results_reference, gram=1)
    bleu4 = caculate_bleu(rationale_data, results_reference, gram=4)

    ## Rouge-L
    rouge = caculate_rouge(rationale_data, results_reference)

    ## Similarity
    model = SentenceTransformer('/data/mm-cot-main/all-MiniLM-L6-v2').cuda()
    similariry = caculate_similariry(rationale_data, results_reference, model)

    scores = {
            "answer":{
                'acc_natural':
                get_acc_with_contion(res_pd, 'subject', 'natural science'),
                'acc_social':
                get_acc_with_contion(res_pd, 'subject', 'social science'),
                'acc_language':
                get_acc_with_contion(res_pd, 'subject', 'language science'),
                'acc_has_text':
                get_acc_with_contion(res_pd, 'has_text', True),
                'acc_has_image':
                get_acc_with_contion(res_pd, 'has_image', True),
                'acc_no_context':
                get_acc_with_contion(res_pd, 'no_context', True),
                'acc_grade_1_6':
                get_acc_with_contion(res_pd, 'grade', ['grade1', 'grade2', 'grade3', 'grade4', 'grade5', 'grade6']),
                'acc_grade_7_12':
                get_acc_with_contion(res_pd, 'grade', ['grade7', 'grade8', 'grade9', 'grade10', 'grade11', 'grade12']),
                'acc_average':
                "{:.2f}".format(acc_average),
            },
            "rationale":{
                'bleu1': bleu1 * 100,
                'bleu4': bleu4 * 100,
                'rouge': rouge * 100,
                'similariry': similariry * 100,
            }
    }
    # Add scores_by_skill information to the existing 'answer' dictionary
    scores["answer"]["scores_by_skill"] = scores_by_skill
    # Save normalized scores to the 'answer' dictionary
    scores["answer"]["normalized_scores_by_skill"] = normalized_scores
    scores["answer"]["top_3_skills"] = top_3_skills

    return scores


def print_scores(scores):
    latex_output = ""
    for key, score in scores.items():
        print(f"{key[4:]}: \t{score}")
        latex_output += f"& {score} "
    latex_output += "\\\\"
    print(latex_output)

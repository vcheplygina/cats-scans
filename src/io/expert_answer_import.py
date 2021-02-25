# Import packages
from statistics import mode
import numpy as np
import xlrd

def expert_answers(expert_answer_path):
    """Loads in all the answers from the experts from the excel file.
    Takes the mode of every question and compares these answers between the datasets to make a vector containing 0, 0.5 and 1.
    Subsequently calculates the length of each vectors and stores it in the similarity matrix."""

    # Load in answers

    workbook = xlrd.open_workbook(expert_answer_path+'/expert_answers.xlsx')

    worksheet = workbook.sheet_by_index(0)

    datasets = ['chest_xray', 'ISIC2018', 'stl-10', 'dtd', 'pcam']

    num_experts = 14
    table_space = 2

    # Make empty question lists

    q1 = []
    q2 = []
    q3 = []
    q4 = []
    q5 = []
    q6 = []
    q7 = []
    q8 = []
    q9 = []
    q10 = []

    q_numbers = [q1, q2, q3, q4, q5, q6, q7, q8, q9, q10]

    # Create dictionary for later use

    answer_dict = {}

    # Get dataset names into dictionary

    for index in range(len(datasets)):
        answer_dict[datasets[index]] = []

    # Define space between tables in excel

    data_gap = num_experts + table_space + 1

    # Add the mode of each of the 10 answers to the right dataset in the dictionary

    for data_index in range(len(datasets)):
        mode_list = []
        for q_index in range(len(q_numbers)):
            for expert_index in range(1, num_experts+1):
                answers_per_question = []
                answers_per_question.append(worksheet.cell(expert_index + data_gap * data_index, q_index).value)
            mode_list.append(mode(answers_per_question))
        answer_dict[datasets[data_index]] = mode_list

    # Make similarity vector based on the differences in answers and calculate its length to then store it in the similarity matrix

    sim_mat = np.zeros((len(datasets), len(datasets)), dtype=float)

    initial_vector = np.zeros((1, len(q_numbers)), dtype=float)

    for data_index_columns in range(len(datasets)):
        dataset_answers = answer_dict[datasets[data_index_columns]]
        for data_index_rows in range(len(datasets)):
            comparison_answers = answer_dict[datasets[data_index_rows]]
            for ans_index in range(len(dataset_answers)):
                if dataset_answers[ans_index] == comparison_answers[ans_index]:
                    vec_value = 0
                elif dataset_answers[ans_index] != comparison_answers[ans_index]:
                    if dataset_answers[ans_index] == 'Both' or comparison_answers[ans_index] == 'Both':
                        vec_value = 0.5
                    else:
                        vec_value = 1
                initial_vector[0][ans_index] = vec_value
            sim_mat[data_index_rows][data_index_columns] = np.around(np.linalg.norm(initial_vector), decimals=2)

    return sim_mat

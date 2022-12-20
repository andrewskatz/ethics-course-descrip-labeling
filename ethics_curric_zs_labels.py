# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 21:56:30 2022

@author: akatz4
"""







# import transformers


# from transformers import pipeline


# from personal_utilities import embed_cluster as ec
from personal_utilities import zs_labeling as zsl

import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns


import pickle


"""

Import data and preprocess

"""




os.getcwd()

# for jee df with abstracts and full article information
proj_path = 'G:\\Shared drives\\Project - Engineering ethics curricula\\Engineering ethics curricula - new\\analysis\\jedm 2022\\label testing'

os.chdir(proj_path)
os.listdir()


text_df = pd.read_csv("course_descrip_sampled_top_score_labels_matched_mpnet_20221117.csv")

print(text_df.columns)

text_col_name = 'original_sent_text'

# =============================================================================
# Load labels
# =============================================================================


# labels_df = pd.read_csv("[old] course_descrip_sampled_labeled_ex_bank_20221113 - ex_bank.csv")
labels_df = pd.read_csv("course description labels - final_labels_v2.csv")
print(labels_df.columns)

level_one_labels = list(labels_df['level_one_label'].unique())
print(len(level_one_labels))

level_two_labels = list(labels_df['level_two_label'].unique())
print(len(level_two_labels))




"""

Labeling

"""


filtered_df = text_df.dropna(subset=[text_col_name])
filtered_df['new_sent_id'] = filtered_df.index


test_df = filtered_df.head(100)



# =============================================================================
# using the utility file zs_labeling.py and label_df_with_zs()
# =============================================================================

zs_threshold = 0.1

text_col_name = 'original_sent_text'
id_col_name = 'new_sent_id'
multi_label = False
keep_top_n = False
top_n=5

total_results_df = zsl.label_df_with_zs(test_df, 
                                        text_col_name, 
                                        id_col_name, 
                                        level_two_labels, 
                                        zs_threshold, 
                                        multi_label=multi_label,
                                        keep_top_n=keep_top_n,
                                        top_n=top_n)


save_date = "20221218"
zs_threshold_save = str(zs_threshold).replace('.', '-')
if multi_label == True:
    total_results_df.to_csv(f"ethics_course_zs_label_v2_{zs_threshold_save}thresh_multi_{save_date}.csv", index = False)

if multi_label == False:
    total_results_df.to_csv(f"ethics_course_zs_label_v2_{zs_threshold_save}thresh_no-multi_{save_date}.csv", index = False)











# =============================================================================
# using new second_round_zsl() function
# =============================================================================

t2_df, level_one_used = zsl.prepare_second_round(total_results_df)


# r2_test_df = t2_df.head(20)
text_col = "sequence"
id_col = "original_id"
zs_threshold = 0.15

r2_df, used = zsl.second_round_zsl(t2_df, 
                             labels_df, 
                             text_col, 
                             id_col, 
                             zs_threshold, 
                             multi_label=multi_label, 
                             keep_top_n=keep_top_n, 
                             top_n=top_n)



save_date = "20221218"
zs_threshold_save = str(zs_threshold).replace('.', '-')
if multi_label == True:
    r2_df.to_csv(f"ethics_course_zs_two-level_v2_label_{zs_threshold_save}thresh_multi_{save_date}.csv", index = False)

if multi_label == False:
    r2_df.to_csv(f"ethics_course_zs_two-level_v2_label_{zs_threshold_save}thresh_no-multi_{save_date}.csv", index = False)
















# =============================================================================
# old school way (without the utility file and label_df_with_zs)
# =============================================================================



classifier = pipeline(task = 'zero-shot-classification', model = 'facebook/bart-large-mnli')


zs_threshold = 0.1

total_results_df = pd.DataFrame(columns=['sequence', 'labels', 'scores', 'new_sent_id'])




for index, row in filtered_df.iterrows():
  row_text = row[text_col]
  new_sent_id = row['new_sent_id']
  
  print(f"working on item {index} with id {new_sent_id}: {row_text}")
  
  classifier_results = classifier(row_text, class_labels)
  results_df = pd.DataFrame(classifier_results)
  results_df['new_sent_id'] = new_sent_id
  
  results_df = results_df[results_df['scores'] > zs_threshold]

  total_results_df = pd.concat([total_results_df, results_df])



zs_threshold_save = str(zs_threshold).replace('.', '-')
total_results_df.to_csv(f"ethics_course_zs_label_{zs_threshold}thresh.csv", index = False)





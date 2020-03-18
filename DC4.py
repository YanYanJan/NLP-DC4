import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

data =  pd.read_csv('/Users/yanyan/PycharmProjects/DC4/quora_duplicate_questions.tsv', sep='\t',error_bad_lines=False)
data = data.drop(['id','qid1','qid2'], axis = 1)

#print(data)

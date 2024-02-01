import pandas as pd 
from constraints.heloc import evaluate_numpy_heloc
from evaluation.constraints import constraint_satisfaction

df = pd.read_csv("data/heloc/train_data.csv")
constraint_satisfaction(df.to_numpy(), use_case="heloc")
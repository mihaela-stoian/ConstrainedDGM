from datasets import load_dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics  

def test_accuracy(X_train, y_train, X_test, y_test):
    clf = RandomForestClassifier(n_estimators = 100)  
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))
    print("AUC OF THE MODEL: ", metrics.roc_auc_score(y_test, y_pred))


dataset = load_dataset("mstz/heloc")["train"]
df = pd.DataFrame(dataset)
df.drop_duplicates(inplace=True)

#http://didawiki.cli.di.unipi.it/lib/exe/fetch.php/bigdataanalytics/bda/credit_risk_prediction_heloc_case.pdf
# 3418 rows out of 9872 that have a negative value of -8
# print(df["net_fraction_of_installment_burden"].value_counts())
# df.drop(columns=["net_fraction_of_installment_burden"], inplace=True)
print((df < 0).sum())
print(df.isna().sum())
print(df.shape)
print(df.head())
print(df.describe())

train_ratio = 0.8
validation_ratio = 0.10
test_ratio = 0.10
ratio_remaining = 1 - test_ratio
ratio_val_adjusted = validation_ratio/ratio_remaining



x_train, x_test = train_test_split(df, test_size=1 - train_ratio, random_state=1)

x_val, x_test = train_test_split(x_test, test_size=ratio_val_adjusted, random_state=1)
print(x_train.shape, x_val.shape, x_test.shape)
print("Train", x_train["is_at_risk"].value_counts())
print("Val", x_val["is_at_risk"].value_counts())
print("Test", x_test["is_at_risk"].value_counts())


x_train.to_csv("data/heloc/train_data.csv", index=False)
x_test.to_csv("data/heloc/test_data.csv", index=False)
x_val.to_csv("data/heloc/val_data.csv", index=False)


test_accuracy(x_train.iloc[:,:-1], x_train.iloc[:,-1], x_test.iloc[:,:-1], x_test.iloc[:,-1])
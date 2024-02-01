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
    print("AUC OF THE MODEL: ", metrics.roc_auc_score(y_test, y_pred,  multi_class='ovo'))


dataset = pd.read_csv("./data/news/news.csv")
df = pd.DataFrame(dataset)
df.drop_duplicates(inplace=True)

df.drop(columns=["url"], inplace=True)
new_col_names = {elem:elem[1:] if elem[0]==' ' else elem for elem in df.columns}
df.rename(columns = new_col_names, inplace = True)
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

print(df.columns)
target_col = 'shares'
print("Original", df[target_col].value_counts())


train_data, test_data = train_test_split(df, test_size=1 - train_ratio, random_state=1)
val_data, test_data = train_test_split(test_data, test_size=ratio_val_adjusted, random_state=1)
print(train_data.shape, val_data.shape, test_data.shape)
print("Train", train_data[target_col].value_counts())
print("Val", val_data[target_col].value_counts())
print("Test", test_data[target_col].value_counts())

train_data.to_csv("data/news/train_data.csv", index=False)
test_data.to_csv("data/news/test_data.csv", index=False)
val_data.to_csv("data/news/val_data.csv", index=False)

test_accuracy(train_data.iloc[:,:-1], train_data.iloc[:,-1], test_data.iloc[:,:-1], test_data.iloc[:,-1])
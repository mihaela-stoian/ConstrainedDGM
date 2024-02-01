from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics  
seed=0
np.random.seed(0)
def test_accuracy(X_train, y_train, X_test, y_test):
    clf = RandomForestClassifier(n_estimators = 100)  
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))
    print("AUC OF THE MODEL: ", metrics.roc_auc_score(y_test, y_pred,  multi_class='ovo'))


def collapse_ohe_cols(original_data: pd.DataFrame, target_cols: list[str]):
    data = original_data.copy()
    data = data.drop(columns=target_cols)
    targets = []
    for i, row in original_data.iterrows():
        found_target = False
        for target in target_cols:
            if row[target]:
                found_target = True
                target_id = target_cols.index(target)
                targets.append(target_id)
        if not found_target:
            targets.append(-1)
    data = data.assign(faults=targets)
    return data


dataset = pd.read_csv("./data/faults/faults.csv")
df = pd.DataFrame(dataset)
df.drop_duplicates(inplace=True)

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


target_cols = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
df = collapse_ohe_cols(df, target_cols)
target_col = 'faults'
print("Original", df["faults"].value_counts())

train_ids = df.groupby(target_col, group_keys=False).apply(lambda x: x.sample(frac=train_ratio)).sample(frac=1).index #.reset_index(drop=True)
train_data = df.iloc[train_ids].reset_index(drop=True)
df = df.drop(index=train_ids).reset_index(drop=True)

test_ids = df.groupby(target_col, group_keys=False).apply(lambda x: x.sample(frac=0.5)).sample(frac=1).index #.reset_index(drop=True)
test_data = df.iloc[test_ids].reset_index(drop=True)
df = df.drop(index=test_ids).reset_index(drop=True)
val_data = df
print("Train", train_data["faults"].value_counts())
print("Val", val_data["faults"].value_counts())
print("Test", test_data["faults"].value_counts())

# x_train, x_test = train_test_split(df, test_size=1 - train_ratio, random_state=1)
#
# x_val, x_test = train_test_split(x_test, test_size=ratio_val_adjusted, random_state=1)
print(train_data.shape, val_data.shape, test_data.shape)

train_data.to_csv("data/faults/train_data.csv", index=False)
test_data.to_csv("data/faults/test_data.csv", index=False)
val_data.to_csv("data/faults/val_data.csv", index=False)

test_accuracy(train_data.iloc[:,:-1], train_data.iloc[:,-1], test_data.iloc[:,:-1], test_data.iloc[:,-1])
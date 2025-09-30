import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import explained_variance_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import ParameterGrid
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from tqdm import tqdm

CATEGORICAL = "categorical"
CONTINUOUS = "continuous"
ORDINAL = "ordinal"

_MODELS = {
    'binary_classification': [ # 184
         {
             'class': DecisionTreeClassifier, # 48
             'kwargs': {
                 'max_depth': [4, 8, 16, 32], 
                 'min_samples_split': [2, 4, 8],
                 'min_samples_leaf': [1, 2, 4, 8]
             }
         },
         {
             'class': AdaBoostClassifier, # 4
             'kwargs': {
                 'n_estimators': [10, 50, 100, 200]
             }
         },
         {
            'class': LogisticRegression, # 36
            'kwargs': {
                 'solver': ['lbfgs'],
                 'n_jobs': [-1],
                 'max_iter': [10, 50, 100, 200],
                 'C': [0.01, 0.1, 1.0],
                 'tol': [1e-01, 1e-02, 1e-04]
             }
         },
        {
            'class': MLPClassifier, # 12
            'kwargs': {
                'hidden_layer_sizes': [(100, ), (200, ), (100, 100)],
                'max_iter': [50, 100],
                'alpha': [0.0001, 0.001]
            }
        },
        {
            'class': RandomForestClassifier, # 48
            'kwargs': {
                 'max_depth': [8, 16, None],
                 'min_samples_split': [2, 4, 8],
                 'min_samples_leaf': [1, 2, 4, 8],
                'n_jobs': [-1]

            }
        },
        {
            'class': XGBClassifier, # 36
            'kwargs': {
                 'n_estimators': [10, 50, 100],
                 'min_child_weight': [1, 10],
                 'max_depth': [5, 10, 20],
                 'gamma': [0.0, 1.0],
                 'objective': ['binary:logistic'],
                 'nthread': [-1]
            },
        }

    ],
    'multiclass_classification': [ # 132
        
        {
            'class': MLPClassifier, # 12
            'kwargs': {
                'hidden_layer_sizes': [(100, ), (200, ), (100, 100)],
                'max_iter': [50, 100],
                'alpha': [0.0001, 0.001]
            }
        },
         {
             'class': DecisionTreeClassifier, # 48
             'kwargs': {
                 'max_depth': [4, 8, 16, 32],
                 'min_samples_split': [2, 4, 8],
                 'min_samples_leaf': [1, 2, 4, 8]
             }
         },
        {
            'class': RandomForestClassifier, # 36
            'kwargs': {
                 'max_depth': [8, 16, None],
                 'min_samples_split': [2, 4, 8],
                 'min_samples_leaf': [1, 2, 4, 8],
                 'n_jobs': [-1]

            }
        },
        {
            'class': XGBClassifier, # 36
            'kwargs': {
                 'n_estimators': [10, 50, 100],
                 'min_child_weight': [1, 10], 
                 'max_depth': [5, 10, 20],
                 'gamma': [0.0, 1.0],
                 'objective': ['binary:logistic'],
                 'nthread': [-1]
            }
        }

    ],
    'regression': [ # 84
        {
            'class': LinearRegression,
        },
        {
            'class': MLPRegressor, # 12
            'kwargs': {
                'hidden_layer_sizes': [(100, ), (200, ), (100, 100)],
                'max_iter': [50, 100],
                'alpha': [0.0001, 0.001]
            }
        },
        {
            'class': XGBRegressor, # 36 
            'kwargs': {
                 'n_estimators': [10, 50, 100],
                 'min_child_weight': [1, 10], 
                 'max_depth': [5, 10, 20],
                 'gamma': [0.0, 1.0],
                 'objective': ['reg:linear'],
                 'nthread': [-1]
            }
        },
        {
            'class': RandomForestRegressor, # 36
            'kwargs': {
                 'max_depth': [8, 16, None],
                 'min_samples_split': [2, 4, 8],
                 'min_samples_leaf': [1, 2, 4, 8],
                 'n_jobs': [-1]
            }
        }
    ]
}


class FeatureMaker:

    def __init__(self, metadata, label_column='label', label_type='int', sample=50000):
        self.columns = metadata['columns']
        self.label_column = label_column
        self.label_type = label_type
        self.sample = sample
        self.encoders = dict()

#     def make_features(self, data):
#         data = data.copy()
#         np.random.shuffle(data)
#         data = data[:self.sample]

#         features = []
#         labels = []

#         for index, cinfo in enumerate(self.columns):
#             col = data[:, index]
#             if cinfo['name'] == self.label_column:
#                 if self.label_type == 'int':
#                     labels = col.astype(int)
#                 elif self.label_type == 'float':
#                     labels = col.astype(float)
#                 else:
#                     assert 0, 'unkown label type'
#                 continue

#             if cinfo['type'] == CONTINUOUS:
#                 cmin = cinfo['min']
#                 cmax = cinfo['max']
#                 if cmin >= 0 and cmax >= 1e3:
#                     feature = np.log(np.maximum(col, 1e-2))

#                 else:
#                     feature = (col - cmin) / (cmax - cmin) * 5

#             elif cinfo['type'] == ORDINAL:
#                 feature = col

#             else:
#                 if cinfo['size'] <= 2:
#                     feature = col

#                 else:
#                     encoder = self.encoders.get(index)
#                     col = col.reshape(-1, 1)
#                     if encoder:
#                         feature = encoder.transform(col)
#                     else:
#                         encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
#                         self.encoders[index] = encoder
#                         feature = encoder.fit_transform(col)

#             features.append(feature)

#         features = np.column_stack(features)

#         return features, labels


# def _prepare_ml_problem(train, val, test, metadata): 
#     fm = FeatureMaker(metadata)
#     x_trains, y_trains = [], []

#     for i in train:
#         x_train, y_train = fm.make_features(i)
#         x_trains.append(x_train)
#         y_trains.append(y_train)

#     x_val, y_val = fm.make_features(val)
#     x_test, y_test = fm.make_features(test)
#     model = _MODELS[metadata['problem_type']]

#     return x_trains, y_trains, x_val, y_val, x_test, y_test, model


def _weighted_f1(y_test, pred):
    report = classification_report(y_test, pred, output_dict=True)
    classes = list(report.keys())[:-3]
    proportion = [  report[i]['support'] / len(y_test) for i in classes]
    weighted_f1 = np.sum(list(map(lambda i, prop: report[i]['f1-score']* (1-prop)/(len(classes)-1), classes, proportion)))
    return weighted_f1 


@ignore_warnings(category=ConvergenceWarning)
def _evaluate_multi_classification(x_trains, y_trains, x_valids, y_valids, x_tests, y_tests, task, size):
    # x_trains, y_trains, x_valid, y_valid, x_test, y_test, classifiers = _prepare_ml_problem(fake, train, test, metadata)
    classifiers = _MODELS["multiclass_classification"]

    
    best_f1_scores = []
    best_weighted_scores = []
    best_auroc_scores = []
    best_acc_scores = []
    best_avg_scores = []

    for model_spec in classifiers:
        model_class = model_spec['class']
        model_kwargs = model_spec.get('kwargs', dict())
        model_repr = model_class.__name__

        unique_labels = np.unique(y_trains[0])

        param_set = list(ParameterGrid(model_kwargs))

        results = []
        for param in tqdm(param_set):
            model = model_class(**param)
            try:
                model.fit(x_trains[0], y_trains[0])
            except:
                pass 
            
            if len(unique_labels) != len(np.unique(y_valids[0])):
                pred = [unique_labels[0]] * len(x_valids[0])
                pred_prob = np.array([1.] * len(x_valids[0]))
            else:
                pred = model.predict(x_valids[0])
                pred_prob = model.predict_proba(x_valids[0])

            macro_f1 = f1_score(y_valids[0], pred, average='macro')
            weighted_f1 = _weighted_f1(y_valids[0], pred)
            acc = accuracy_score(y_valids[0], pred)

            # 3. auroc
            rest_label = set(range(size)) - set(unique_labels)
            tmp = []
            j = 0
            for i in range(size):
                if i in rest_label:
                    tmp.append(np.array([0] * y_valids[0].shape[0])[:,np.newaxis])
                else:
                    try:
                        tmp.append(pred_prob[:,[j]])
                    except:
                        tmp.append(pred_prob[:, np.newaxis])
                    j += 1
            roc_auc = roc_auc_score(np.eye(size)[y_valids[0]], np.hstack(tmp), multi_class='ovr')
                
            results.append(
                {   
                    "name": model_repr,
                    "param": param,
                    "macro_f1": macro_f1,
                    "weighted_f1": weighted_f1,
                    "roc_auc": roc_auc, 
                    "accuracy": acc
                }
            )

        results = pd.DataFrame(results)
        results['avg'] = results.loc[:, ['macro_f1', 'weighted_f1', 'roc_auc']].mean(axis=1)        
        best_f1_param = results.param[results.macro_f1.idxmax()]
        best_weighted_param = results.param[results.weighted_f1.idxmax()]
        best_auroc_param = results.param[results.roc_auc.idxmax()]
        best_acc_param = results.param[results.accuracy.idxmax()]
        best_avg_param = results.param[results.avg.idxmax()]


        # test the best model
        results = pd.DataFrame(results)
        # best_param = results.param[results.macro_f1.idxmax()]

        def _calc(best_model, task):
            best_scores = []
            i = 0
            for x_train, y_train in zip(x_trains, y_trains):
                if task=="utility":
                    x_test = x_tests[0]
                    y_test = y_tests[0]
                elif task=="detection":
                    x_test = x_tests[i]
                    y_test = y_tests[i]
                try:
                    best_model.fit(x_train, y_train)
                    learned_classes = best_model.classes_
                except:
                    pass 
                
                if len(unique_labels) != len(np.unique(y_test)):
                    pred = [unique_labels[0]] * len(x_test)
                    pred_prob = np.array([1.] * len(x_test))
                else:
                    pred = best_model.predict(x_test)
                    pred_prob = best_model.predict_proba(x_test)

                macro_f1 = f1_score(y_test, pred, average='macro')
                weighted_f1 = _weighted_f1(y_test, pred)
                acc = accuracy_score(y_test, pred)

                # 3. auroc
                #size = [a["size"] for a in metadata["columns"] if a["name"] == "label"][0]

                rest_label = set(unique_labels) - set(learned_classes)
                tmp = []
                j = 0
                for i in range(size):
                    if i in rest_label:
                        tmp.append(np.array([0] * pred_prob.shape[0])[:,np.newaxis])
                    else:
                        try:
                            tmp.append(pred_prob[:,[j]])
                        except:
                            ## This is not correct and it should fail but at same time the try expression should always work
                            tmp.append(pred_prob[:, np.newaxis])
                        j += 1


                # Workaround
                try:
                    roc_auc = roc_auc_score(np.eye(size)[y_test], np.hstack(tmp), multi_class='ovr')
                except ValueError as v:
                    print(v)
                    roc_auc = 0.5
                    
                best_scores.append(
                    {   
                        "name": model_repr,
                        "macro_f1": macro_f1,
                        "weighted_f1": weighted_f1,
                        "roc_auc": roc_auc, 
                        "accuracy": acc
                    }
                )
                i += 1
            return pd.DataFrame(best_scores).mean(axis=0)

        def _df(dataframe):
            return {
                "name": model_repr,
                "macro_f1": dataframe.macro_f1,
                "roc_auc": dataframe.roc_auc,
                "weighted_f1": dataframe.weighted_f1,
                "accuracy": dataframe.accuracy,
            }

        best_f1_scores.append(_df(_calc(model_class(**best_f1_param), task)))
        best_weighted_scores.append(_df(_calc(model_class(**best_weighted_param), task)))
        best_auroc_scores.append(_df(_calc(model_class(**best_auroc_param), task)))
        best_acc_scores.append(_df(_calc(model_class(**best_acc_param), task)))
        best_avg_scores.append(_df(_calc(model_class(**best_avg_param), task)))

    return pd.DataFrame(best_f1_scores), pd.DataFrame(best_weighted_scores), pd.DataFrame(best_auroc_scores)


@ignore_warnings(category=ConvergenceWarning)
def _evaluate_binary_classification( x_trains, y_trains, x_valids, y_valids, x_tests, y_tests, task, size=2):
    ## Train using train set (synth), evaluate all of them on validation, select best and evaluate on test
    ## Do a hyperparameter search for real train data as well separatly from synth
    

    classifiers = _MODELS["binary_classification"]
    #x_trains, y_trains, x_valid, y_valid, x_test, y_test, classifiers = _prepare_ml_problem(fake, train, test, metadata)

    best_f1_scores = []
    best_weighted_scores = []
    best_auroc_scores = []
    best_acc_scores = []
    best_avg_scores = []

    for model_spec in classifiers:
        print("Model", model_spec)
        model_class = model_spec['class']
        model_kwargs = model_spec.get('kwargs', dict())
        model_repr = model_class.__name__

        unique_labels = np.unique(y_trains[0])

        param_set = list(ParameterGrid(model_kwargs))

        results = []
        for param in tqdm(param_set):
            model = model_class(**param)
            
            try:
                model.fit(x_trains[0], y_trains[0])
            except ValueError as ve:
                pass

            if len(unique_labels) == 1:
                pred = [unique_labels[0]] * len(x_valids[0])
                pred_prob = np.array([1.] * len(x_valids[0]))
            else:
                pred = model.predict(x_valids[0])
                pred_prob = model.predict_proba(x_valids[0])

            binary_f1 = f1_score(y_valids[0], pred, average='binary')
            weighted_f1 = _weighted_f1(y_valids[0], pred)
            acc = accuracy_score(y_valids[0], pred)
            precision = precision_score(y_valids[0], pred, average='binary')
            recall = recall_score(y_valids[0], pred, average='binary')
            macro_f1 = f1_score(y_valids[0], pred, average='macro')

            # auroc
            #size = [a["size"] for a in metadata["columns"] if a["name"] == "label"][0]
            rest_label = set(range(size)) - set(unique_labels)
            tmp = []
            j = 0
            for i in range(size):
                if i in rest_label:
                    tmp.append(np.array([0] * y_valids[0].shape[0])[:,np.newaxis])
                else:
                    try:
                        tmp.append(pred_prob[:,[j]])
                    except:
                        tmp.append(pred_prob[:, np.newaxis])
                    j += 1
            roc_auc = roc_auc_score(np.eye(size)[y_valids[0]], np.hstack(tmp))

            results.append(
                {   
                    "name": model_repr,
                    "param": param,
                    "binary_f1": binary_f1,
                    "weighted_f1": weighted_f1,
                    "roc_auc": roc_auc, 
                    "accuracy": acc, 
                    "precision": precision, 
                    "recall": recall, 
                    "macro_f1": macro_f1
                }
            )

        # test the best model
        results = pd.DataFrame(results)
        results['avg'] = results.loc[:, ['binary_f1', 'weighted_f1', 'roc_auc']].mean(axis=1)        
        best_f1_param = results.param[results.binary_f1.idxmax()]
        best_weighted_param = results.param[results.weighted_f1.idxmax()]
        best_auroc_param = results.param[results.roc_auc.idxmax()]
        best_acc_param = results.param[results.accuracy.idxmax()]
        best_avg_param = results.param[results.avg.idxmax()]

        def _calc(best_model, task):
            best_scores = []
            i = 0
            for x_train, y_train in zip(x_trains, y_trains):
                if task=="utility":
                    x_test = x_tests[0]
                    y_test = y_tests[0]
                elif task=="detection":
                    x_test = x_tests[i]
                    y_test = y_tests[i]
                try:
                    best_model.fit(x_train, y_train)
                    learned_classes = best_model.classes_

                except ValueError:
                    pass

                if len(unique_labels) == 1:
                    pred = [unique_labels[0]] * len(x_test)
                    pred_prob = np.array([1.] * len(x_test))
                else:
                    pred = best_model.predict(x_test)
                    pred_prob = best_model.predict_proba(x_test)

                binary_f1 = f1_score(y_test, pred, average='binary')
                weighted_f1 = _weighted_f1(y_test, pred)
                acc = accuracy_score(y_test, pred)
                precision = precision_score(y_test, pred, average='binary')
                recall = recall_score(y_test, pred, average='binary')
                macro_f1 = f1_score(y_test, pred, average='macro')

                # auroc
                #size = [a["size"] for a in metadata["columns"] if a["name"] == "label"][0]
                #size = 2 ##Due to binary
                rest_label = set(unique_labels) - set(learned_classes)
                tmp = []
                j = 0
                for i in range(size):
                    if i in rest_label:
                        tmp.append(np.array([0] * y_test.shape[0])[:,np.newaxis])
                    else:
                        try:
                            tmp.append(pred_prob[:,[j]])
                        except:
                            tmp.append(pred_prob[:, np.newaxis])
                        j += 1
                ## Workaround
                try:
                    roc_auc = roc_auc_score(np.eye(size)[y_test], np.hstack(tmp))
                except ValueError:
                    #tmp[1] = tmp[1].reshape(20000, 1)
                    #roc_auc = roc_auc_score(np.eye(size)[y_test], np.hstack(tmp))
                    roc_auc = 0.5

                best_scores.append(
                    {   
                        "name": model_repr,
                        # "param": param,
                        "binary_f1": binary_f1,
                        "weighted_f1": weighted_f1,
                        "roc_auc": roc_auc, 
                        "accuracy": acc, 
                        "precision": precision, 
                        "recall": recall, 
                        "macro_f1": macro_f1
                    }
                )
                # print(len(best_scores))
                i += 1
            return pd.DataFrame(best_scores).mean(axis=0)

        def _df(dataframe):
            return {
                "name": model_repr,
                "binary_f1": dataframe.binary_f1,
                "roc_auc": dataframe.roc_auc,
                "weighted_f1": dataframe.weighted_f1,
                "accuracy": dataframe.accuracy,
            }

        best_f1_scores.append(_df(_calc(model_class(**best_f1_param), task)))
        best_weighted_scores.append(_df(_calc(model_class(**best_weighted_param), task)))
        best_auroc_scores.append(_df(_calc(model_class(**best_auroc_param), task)))
        best_acc_scores.append(_df(_calc(model_class(**best_acc_param), task)))
        best_avg_scores.append(_df(_calc(model_class(**best_avg_param), task)))

    return pd.DataFrame(best_f1_scores), pd.DataFrame(best_weighted_scores), pd.DataFrame(best_auroc_scores)


@ignore_warnings(category=ConvergenceWarning)
def _evaluate_regression(x_trains, y_trains, x_valids, y_valids, x_tests, y_tests, task, size=None):
    #x_trains, y_trains, x_valid, y_valid, x_test, y_test, regressors = _prepare_ml_problem(fake, train, test, metadata)
    regressors = _MODELS["regression"]
    best_r2_scores = []
    best_ev_scores = []
    best_mae_scores = []
    best_rmse_scores = []
    best_avg_scores = []

    y_trains = [np.log(np.clip(i, 1, 20000)) for i in y_trains]
    y_test = np.log(np.clip(y_tests[0], 1, 20000))

    for model_spec in regressors:
        model_class = model_spec['class']
        model_kwargs = model_spec.get('kwargs', dict())
        model_repr = model_class.__name__

        param_set = list(ParameterGrid(model_kwargs))

        results = []
        for param in tqdm(param_set):
            model = model_class(**param)
            model.fit(x_trains[0], y_trains[0])
            pred = model.predict(x_valids[0])

            r2 = r2_score(y_valids[0], pred)
            explained_variance = explained_variance_score(y_valids[0], pred)
            mean_squared = mean_squared_error(y_valids[0], pred)
            root_mean_squared = mean_squared_error(y_valids[0], pred, squared=False)
            mean_absolute = mean_absolute_error(y_valids[0], pred)

            results.append(
                {   
                    "name": model_repr,
                    "param": param,
                    "r2": r2,
                    "explained_variance": explained_variance,
                    "mean_squared": mean_squared, 
                    "mean_absolute": mean_absolute, 
                    "rmse": root_mean_squared
                }
            )

        results = pd.DataFrame(results)
        # results['avg'] = results.loc[:, ['r2', 'rmse']].mean(axis=1)        
        best_r2_param = results.param[results.r2.idxmax()]
        best_ev_param = results.param[results.explained_variance.idxmax()]
        best_mae_param = results.param[results.mean_absolute.idxmin()]
        best_rmse_param = results.param[results.rmse.idxmin()]
        # best_avg_param = results.param[results.avg.idxmax()]

        def _calc(best_model, task):
            best_scores = []
            i = 0
            for x_train, y_train in zip(x_trains, y_trains):
                if task=="utility":
                    x_test = x_tests[0]
                    y_test = y_tests[0]
                elif task=="detection":
                    x_test = x_tests[i]
                    y_test = y_tests[i]
                best_model.fit(x_train, y_train)
                pred = best_model.predict(x_test)

                r2 = r2_score(y_test, pred)
                explained_variance = explained_variance_score(y_test, pred)
                mean_squared = mean_squared_error(y_test, pred)
                root_mean_squared = mean_squared_error(y_test, pred, squared=False)
                mean_absolute = mean_absolute_error(y_test, pred)

                best_scores.append(
                    {   
                        "name": model_repr,
                        "param": param,
                        "r2": r2,
                        "explained_variance": explained_variance,
                        "mean_squared": mean_squared, 
                        "mean_absolute": mean_absolute, 
                        "rmse": root_mean_squared
                    }
                )
                i += 1
            return pd.DataFrame(best_scores).mean(axis=0)

        def _df(dataframe):
            return {
                "name": model_repr,
                "r2": dataframe.r2,
                "explained_variance": dataframe.explained_variance,
                "MAE": dataframe.mean_absolute,
                "RMSE": dataframe.rmse,
            }

        best_r2_scores.append(_df(_calc(model_class(**best_r2_param), task)))
        best_ev_scores.append(_df(_calc(model_class(**best_ev_param), task)))
        best_mae_scores.append(_df(_calc(model_class(**best_mae_param), task)))
        best_rmse_scores.append(_df(_calc(model_class(**best_rmse_param), task)))

    return pd.DataFrame(best_r2_scores), pd.DataFrame(best_rmse_scores), None



_EVALUATORS = {
    'binary_classification': _evaluate_binary_classification,
    'multiclass_classification': _evaluate_multi_classification,
    'regression': _evaluate_regression
}



def prepare_detection_data(real_data, generated_data, columns):
    split = ["train", "val", "test"]
    all_data = [[], [], []]
    for j in range(3):
        for i in range(len(generated_data["train"])):
            label_real = [0]*real_data[split[j]].shape[0]
            label_syn = [1]*real_data[split[j]].shape[0]
            labels = pd.DataFrame({"Synthetic":label_real +label_syn})
            gen_data_i = pd.DataFrame(generated_data[split[j]][i], columns=columns)
            data = pd.concat([real_data[split[j]], gen_data_i], axis=0).reset_index(drop=True)
            ## Shuffling
            idx = np.random.permutation(data.index)
            data = data.reindex(idx).reset_index(drop=True)
            labels = labels.reindex(idx).reset_index(drop=True)
            all_data[j].append((data,labels))
    return all_data

def compute_detection_scores(real_data, generated_data, problem_type, columns, target_column):
    all_data = prepare_detection_data(real_data, generated_data, columns)
    print('Start stasy detection eval for SYNTH data')

    x_trains, y_trains,  x_valids, y_valids, x_tests, y_tests = [], [], [], [], [], []
    for i in range(len(generated_data["train"])):
        x_train, y_train = all_data[0][i]
        x_trains.append(x_train)
        y_trains.append(y_train.squeeze())
        x_valid, y_valid = all_data[1][i]
        x_valids.append(x_valid)
        y_valids.append(y_valid.squeeze())
        x_test, y_test = all_data[2][i]
        x_tests.append(x_test)
        y_tests.append(y_test.squeeze())
    a, b, c = _EVALUATORS[problem_type](x_trains, y_trains, x_valids, y_valids, x_tests, y_tests, 'detection') 

    # det_avg = pd.DataFrame([a_all.mean(axis=0), a_all.std(axis=0)])
    det_avg = pd.DataFrame([a.mean(axis=0)])
    return (a, b, c, det_avg)


def compute_utility_scores(real_data, generated_data, problem_type, columns, target_column, target_size, eval_real=False):

    y_trains_real = [real_data["train"][target_column]]
    x_train = real_data["train"].drop(columns=[target_column])
    x_trains_real = [x_train]
    y_valid = real_data["val"][target_column]
    x_valid = real_data["val"].drop(columns=[target_column])
    y_test = real_data["test"][target_column]
    x_test = real_data["test"].drop(columns=[target_column])

    ## Real data evaluation
    if eval_real:
        print('Started stasy utility eval for REAL data')
        a_real, b_real, c_real = _EVALUATORS[problem_type](x_trains_real, y_trains_real, [x_valid], [y_valid], [x_test], [y_test], 'utility', target_size)
        # df_avg_real = pd.DataFrame([a_real.mean(axis=0), a_real.std(axis=0)])
        df_avg_real = pd.DataFrame([a_real.mean(axis=0)])
    else:
        a_real, b_real, c_real, df_avg_real = None, None, None, None

        ## Synthetic data evaluation
    print('Started stasy utility eval for SYNTH data')
    y_trains_syn, x_trains_syn = [], []
    for fake_data in generated_data:
        #fake_data = pd.DataFrame(fake_data, columns=columns)
        y_trains_syn.append(fake_data[target_column])
        fake_data = fake_data.drop(columns=[target_column])
        x_trains_syn.append(fake_data)
    a_syn, b_syn, c_syn = _EVALUATORS[problem_type](x_trains_syn, y_trains_syn, [x_valid], [y_valid], [x_test], [y_test], "utility", target_size) 
    # df_avg_syn = pd.DataFrame([a_syn.mean(axis=0), a_syn.std(axis=0)])
    df_avg_syn = pd.DataFrame([a_syn.mean(axis=0)])
    print('Finished stasy utility eval for SYNTH data')

    return (a_real, b_real, c_real, df_avg_real), (a_syn, b_syn, c_syn, df_avg_syn)

def compute_utility_real(real_data, generated_data, problem_type, columns, target_column, target_size, eval_real=False):
    y_trains_real = [real_data["train"][target_column]]
    x_train = real_data["train"].drop(columns=[target_column])
    x_trains_real = [x_train]
    y_valid = real_data["val"][target_column]
    x_valid = real_data["val"].drop(columns=[target_column])
    y_test = real_data["test"][target_column]
    x_test = real_data["test"].drop(columns=[target_column])
    print('Started stasy utility eval for REAL data')
    a_real, b_real, c_real = _EVALUATORS[problem_type](x_trains_real, y_trains_real, [x_valid], [y_valid], [x_test], [y_test], 'utility', target_size)
    # df_avg_real = pd.DataFrame([a_real.mean(axis=0), a_real.std(axis=0)])
    df_avg_real = pd.DataFrame([a_real.mean(axis=0)])
    return (a_real, b_real, c_real, df_avg_real)


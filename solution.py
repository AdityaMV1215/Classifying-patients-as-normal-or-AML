import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, make_scorer, recall_score, f1_score

df = pd.read_csv("AML_Classification_Data.csv")
df.drop(labels='Unnamed: 0', axis=1, inplace=True)
train_data = df.iloc[0:179,:]
final_test_data = df.iloc[179:,:]

def rf(X_train, y_train, X_test, y_test, metric, n_splits, random_state, print_info):
    n_estimators = [i for i in range(100, 200, 50)]
    criterion = ['gini', 'entropy']
    min_samples_split = [i for i in range(2, 10)]
    min_samples_leaf = [i for i in range(1, 10)]
    oob_score = [True, False]
    warm_start = [True, False]
    param_grid_rf = {'n_estimators': n_estimators, 'criterion': criterion, 'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf, 'oob_score': oob_score, 'warm_start': warm_start}
    scorer_rf = make_scorer(metric)
    skf_rf = StratifiedKFold(n_splits=n_splits)
    rf_model = RandomForestClassifier()
    grid_search_model_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, scoring=scorer_rf, cv=skf_rf)
    grid_search_model_rf.fit(X_train, y_train)
    y_pred_rf = grid_search_model_rf.predict(X_test)
    if print_info:
        tn_rf, fp_rf, fn_rf, tp_rf = confusion_matrix(y_test, y_pred_rf).ravel()
        print("Random Forest - Testing Data")
        print("Best paramaters = {}".format(grid_search_model_rf.best_params_))
        print("True negative = {}, False Positive = {}, False Negative = {}, True Positive = {}".format(tn_rf, fp_rf, fn_rf, tp_rf))
        print("Accuracy score = {}".format(accuracy_score(y_test, y_pred_rf)))
        print("Recall score = {}".format(recall_score(y_test, y_pred_rf)))
        print("AUC ROC score = {}".format(roc_auc_score(y_test, y_pred_rf)))
        print("F1 score = {}".format(f1_score(y_test, y_pred_rf)))
        print()

    return grid_search_model_rf, f1_score(y_test, y_pred_rf)

def rf_fixed(X_train, y_train, X_test, y_test, n_estimators, criterion, min_samples_split, min_samples_leaf, oob_score, warm_start):
    rf_model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, oob_score=oob_score, warm_start=warm_start)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    tn_rf, fp_rf, fn_rf, tp_rf = confusion_matrix(y_test, y_pred_rf).ravel()
    print("Random Forest - Testing Data")
    print("True negative = {}, False Positive = {}, False Negative = {}, True Positive = {}".format(tn_rf, fp_rf, fn_rf, tp_rf))
    print("Accuracy score = {}".format(accuracy_score(y_test, y_pred_rf)))
    print("Recall score = {}".format(recall_score(y_test, y_pred_rf)))
    print("AUC ROC score = {}".format(roc_auc_score(y_test, y_pred_rf)))
    print("F1 score = {}".format(f1_score(y_test, y_pred_rf)))
    print()
    return rf_model, f1_score(y_test, y_pred_rf)

def nn(X_train, y_train, X_test, y_test, metric, n_splits, random_state, print_info):

    activation_functions = ['identity', 'logistic', 'tanh', 'relu']
    learning_rate = ['constant', 'invscaling', 'adaptive']
    learning_rate_init = [0.001, 0.003, 0.004, 0.006, 0.009, 0.027, 0.01, 0.03, 0.04, 0.06, 0.09]
    hidden_layer_sizes = []
    for neurons in range(100, 500, 25):
        t = []
        val = neurons
        for size in range(0, 1):
            t.append(val)
            val = val + neurons

        hidden_layer_sizes.append(tuple(t))

    param_grid_nn = {'activation': activation_functions, 'hidden_layer_sizes': hidden_layer_sizes, 'learning_rate': learning_rate, 'learning_rate_init': learning_rate_init}
    scorer_nn = make_scorer(metric)
    skf_nn = StratifiedKFold(n_splits=n_splits)
    nn_model = MLPClassifier(max_iter=10000)
    grid_search_model_nn = GridSearchCV(estimator=nn_model, param_grid=param_grid_nn, scoring=scorer_nn, cv=skf_nn)
    grid_search_model_nn.fit(X_train, y_train)
    y_pred_nn = grid_search_model_nn.predict(X_test)
    if print_info:
        tn_nn, fp_nn, fn_nn, tp_nn = confusion_matrix(y_test, y_pred_nn).ravel()
        print("Neural Network - Testing Data")
        print("Best paramaters = {}".format(grid_search_model_nn.best_params_))
        print("True negative = {}, False Positive = {}, False Negative = {}, True Positive = {}".format(tn_nn, fp_nn, fn_nn, tp_nn))
        print("Accuracy score = {}".format(accuracy_score(y_test, y_pred_nn)))
        print("Recall score = {}".format(recall_score(y_test, y_pred_nn)))
        print("AUC ROC score = {}".format(roc_auc_score(y_test, y_pred_nn)))
        print("F1 score = {}".format(f1_score(y_test, y_pred_nn)))
        print()
    return grid_search_model_nn, f1_score(y_test, y_pred_nn)

def nn_fixed(X_train, y_train, X_test, y_test, activation, hls, lr, lrinit):
    nn_model = MLPClassifier(activation=activation, hidden_layer_sizes=hls, learning_rate=lr, learning_rate_init=lrinit, max_iter=10000)
    nn_model.fit(X_train, y_train)
    y_pred_nn_fixed = nn_model.predict(X_test)
    tn_nn, fp_nn, fn_nn, tp_nn = confusion_matrix(y_test, y_pred_nn_fixed).ravel()
    print("Neural Network - Testing Data")
    print("True negative = {}, False Positive = {}, False Negative = {}, True Positive = {}".format(tn_nn, fp_nn, fn_nn, tp_nn))
    print("Accuracy score = {}".format(accuracy_score(y_test, y_pred_nn_fixed)))
    print("Recall score = {}".format(recall_score(y_test, y_pred_nn_fixed)))
    print("AUC ROC score = {}".format(roc_auc_score(y_test, y_pred_nn_fixed)))
    print("F1 score = {}".format(f1_score(y_test, y_pred_nn_fixed)))
    print()
    return nn_model, f1_score(y_test, y_pred_nn_fixed)

multiple_predictions = []
relevant_features = []
for i in [0,1,2,4,5,8,10,13,17,19,23]:
    f1 = 0
    x = 0
    count = 0
    threshold = 1.0
    X_train, X_test, y_train, y_test = train_test_split(train_data.iloc[:, 0:-1], train_data.iloc[:, -1], test_size=0.30, random_state=i)
    while f1 < threshold:
        if count > 30:
            x = 0
            count = 0
            threshold = threshold - 0.02

        if x == 0:
            rfc = RandomForestClassifier(n_estimators=100)
            selector = SelectFromModel(estimator=rfc)
            selector.fit(X_train, y_train)
            selected_feature_mask = selector.get_support()
            temp = np.array(list(X_train.columns.values))
            selected_features = set(temp[selected_feature_mask])
            X_train_new = selector.transform(X_train)
            X_test_new = selector.transform(X_test)

            print("Current random state = {}, current count = {}, current threshold = {}".format(i, count, threshold))
            print("number of aml samples in train = {}, number of aml samples in test = {}".format(y_train[y_train == 1].shape[0], y_test[y_test == 1].shape[0]))

            #model, f1 = rf(X_train_new, y_train, X_test_new, y_test, roc_auc_score, 5, 0, 1)
            model, f1 = nn(X_train_new, y_train, X_test_new, y_test, roc_auc_score, 5, 0, 1)
            best_params = model.best_params_

            X_test_final = selector.transform(final_test_data.iloc[:,0:-1])
            y_pred_final = model.predict(X_test_final)
            x = x + 1

        else:
            print("Current random state = {}, current count = {}, current threshold = {}".format(i, count, threshold))
            print("number of aml samples in train = {}, number of aml samples in test = {}".format(y_train[y_train == 1].shape[0], y_test[y_test == 1].shape[0]))
            #model, f1 = rf_fixed(X_train_new, y_train, X_test_new, y_test, best_params['n_estimators'], best_params['criterion'], best_params['min_samples_split'], best_params['min_samples_leaf'], best_params['oob_score'], best_params['warm_start'])
            model, f1 = nn_fixed(X_train_new, y_train, X_test_new, y_test, best_params['activation'], best_params['hidden_layer_sizes'], best_params['learning_rate'], best_params['learning_rate_init'])
            y_pred_final = model.predict(X_test_final)

        count = count + 1

    multiple_predictions.append(y_pred_final)
    if f1 == 1.0:
        relevant_features.append(selected_features)

final_vote = []
confidence = [1] * 1432
for i in range(0,len(multiple_predictions[0])):
    count_one = 0
    count_zero = 0
    for j in range(0,len(multiple_predictions)):
        if multiple_predictions[j][i] == 1:
            count_one = count_one + 1
        else:
            count_zero = count_zero + 1

    if count_one > count_zero:
        final_vote.append(1)
        confidence.append(count_one / (count_zero+count_one))
        confidence.append(count_one / (count_zero + count_one))
        confidence.append(count_one / (count_zero + count_one))
        confidence.append(count_one / (count_zero + count_one))
        confidence.append(count_one / (count_zero + count_one))
        confidence.append(count_one / (count_zero + count_one))
        confidence.append(count_one / (count_zero + count_one))
        confidence.append(count_one / (count_zero + count_one))
    else:
        final_vote.append(0)
        confidence.append(count_zero / (count_zero+count_one))
        confidence.append(count_zero / (count_zero + count_one))
        confidence.append(count_zero / (count_zero + count_one))
        confidence.append(count_zero / (count_zero + count_one))
        confidence.append(count_zero / (count_zero + count_one))
        confidence.append(count_zero / (count_zero + count_one))
        confidence.append(count_zero / (count_zero + count_one))
        confidence.append(count_zero / (count_zero + count_one))

print()
print("Final vote = {} and final vote has {} positive samples".format(final_vote, final_vote.count(1)))
print("Confidence values = {}".format(confidence))

best_features = relevant_features[0]
for i in range(1,len(relevant_features)):
    best_features = best_features.intersection(relevant_features[i])

print("The selected features are {}".format(best_features))
df_ans = pd.read_csv("AMLTraining.csv")

temp1 = df_ans.loc[df_ans['Label'] != 'normal']
temp2 = temp1.loc[temp1['Label'] != 'aml']
current = min(temp2.index.values)
for i in range(0,len(final_vote)):
    for j in range(0,8):
        if final_vote[i] == 1:
            df_ans.iloc[current+j, -1] = 'aml'
        else:
            df_ans.iloc[current+j, -1] = 'normal'
    current = current + 8

df_ans['Confidence'] = confidence
df_ans.to_csv("answer.csv")
df = pd.read_csv("answer.csv")
df.drop(labels='Unnamed: 0', axis=1, inplace=True)
df.to_csv("predictions.csv", index=False)
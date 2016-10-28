from __future__ import division
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score

import matplotlib.pyplot as plt

plt.style.use('ggplot')

np.random.seed(0)

N_ESTIMATORS_RANDOM_FOREST = 1024
RUN_ON_TEST_SET = True



df = pd.read_csv('../extracted_features/features.csv', index_col=False) #names=["name","label","subset"]+['f'+str(x) for x in range(100)], )
print "Featureset shape", df.shape
#Take train and validation subset

df = df.fillna(value=0)
df_train = df[df['subset'] < 2]
print "After test set removal", df.shape

labels_original = df_train['label'].values
features = df_train.drop(['name','label','subset'], axis=1).values



print "\nTrain set label counts"
print df[df['subset'] == 0].label.value_counts()

print "\nValidation set label counts"
print df[df['subset'] == 1].label.value_counts()



print features.shape

df_test = df[df['subset'] == 2]
labels_original_test = df_test['label'].values
features_test = df_test.drop(['name','label','subset'], axis=1).values

print "N TEST SAMPLES", len(labels_original_test)
print "N TEST BENIGN", np.sum(labels_original_test==0)
print "N TEST DCIS", np.sum(labels_original_test==1)
print "N TEST IDC", np.sum(labels_original_test==2), '\n'






#ROC
plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

results={}

for problem in ['BINARY','TERNARY']:


    labels = np.copy(labels_original)

    if problem == 'BINARY':
        labels = labels_original > 0

    X = features
    #X = features[:,:10]
    y = labels

    cl = RandomForestClassifier(n_estimators=N_ESTIMATORS_RANDOM_FOREST, random_state=0)
    kf = KFold(len(X), len(X))

    scores = []

    targets = []
    preds = []
    preds_proba = []


    for train, test in tqdm(kf):
        #print("%s %s" % (train, test))
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

        cl.fit(X_train, y_train)
        score = cl.score(X_test, y_test)

        targets.append(y_test)
        preds.append(cl.predict(X_test))
        preds_proba.append(cl.predict_proba(X_test))



        scores.append(score)
        #print cl.score(X_test, y_test)

    print np.mean(scores)
    print confusion_matrix(targets, preds)

    targets_roc = np.array(targets)[:,0]

    if problem == 'BINARY':
        preds_proba = np.array(preds_proba)[:,0,1]
        results['2CVACC'] = np.mean(scores)
        results['2CVAUC'] = roc_auc_score(targets_roc, preds_proba)
        results['2CVKAPPA'] = cohen_kappa_score(targets_roc, preds)

        fpr, tpr, thresholds = roc_curve(targets_roc, preds_proba)
        plt.plot(fpr, tpr, color='b', linestyle='dashed')

    else:
        results['3CVACC'] = np.mean(scores)
        results['3CVKAPPA'] = cohen_kappa_score(targets, preds)

print results




if RUN_ON_TEST_SET:

    # Finally on the test set 
    cl = RandomForestClassifier(n_estimators=N_ESTIMATORS_RANDOM_FOREST, random_state=0)
    cl.fit(features, labels_original)
    predictions = cl.predict(features_test)

    print "TEST SET CONFUSION MATRIX\n", confusion_matrix(labels_original_test, predictions)

    results['3TACC'] = accuracy_score(labels_original_test, predictions)
    results['3TKAPPA'] = cohen_kappa_score(labels_original_test, predictions)

    labels_binary = labels_original > 0
    labels_binary_test = labels_original_test > 0

    cl = RandomForestClassifier(n_estimators=N_ESTIMATORS_RANDOM_FOREST, random_state=0)
    cl.fit(features, labels_binary)
    predictions = cl.predict(features_test)
    predictions_proba = cl.predict_proba(features_test)[:,1]

    results['2TACC'] = accuracy_score(labels_binary_test, predictions)
    results['2TAUC'] = roc_auc_score(labels_binary_test, predictions_proba)
    results['2TKAPPA'] = cohen_kappa_score(labels_binary_test, predictions)

    fpr, tpr, thresholds = roc_curve(labels_binary_test, predictions_proba)
    plt.plot(fpr, tpr, color='b')
    plt.savefig('../figures/output/roc.png',bbox_inches='tight')
    plt.show()

    






    table = r"""
    \begin{table}[h]
    \renewcommand{\arraystretch}{1.1}
    \caption{Results of whole-slide image label prediction}
    \label{table_results_single_label}
    \centering
    \begin{tabular}{|lrrrr|}
    \hline
    \textsc{Labels}&\textsc{Result set}&\textsc{Acc}&\textsc{Kappa}&\textsc{AUC}\\
    \hline
    \textit{Benign, Cancer}&&&&\\
    &Leave-one out CV&2CVACC&2CVKAPPA&2CVAUC \\
    &Independent test set&2TACC&2TKAPPA&2TAUC \\
    \textit{Benign, DCIS, IDC}&&&&\\
    &Leave-one out CV&3CVACC&3CVKAPPA&- \\
    &Independent test set&3TACC&3TKAPPA&- \\
    \hline

    \end{tabular}
    \end{table}

    """


    for k, v in results.iteritems():
        table = table.replace(k, '{0:.4f}'.format(v))


    with open('../figures/output/table_rf.tex', 'w') as f:
        f.write(table)

#print table


from __future__ import division
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix

np.random.seed(0)





df = pd.read_csv('../extracted_features/features.csv', index_col=False) #names=["name","label","subset"]+['f'+str(x) for x in range(100)], )
print "Featureset shape", df.shape
#Take train and validation subset
df = df[df['subset'] < 2]
df = df.fillna(value=0)
print "After test set removal", df.shape

labels = df['label'].values
print labels.shape
features = df.drop(['name','label','subset'], axis=1).values
print features.shape



#labels = labels > 0

X = features
#X = features[:,:10]
y = labels

cl = RandomForestClassifier(n_estimators=100)
#cl = LogisticRegression()
kf = KFold(len(X), len(X))

scores = []

targets = []
preds = []


for train, test in tqdm(kf):
     #print("%s %s" % (train, test))
     X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

     cl.fit(X_train, y_train)
     score = cl.score(X_test, y_test)

     targets.append(y_test)
     preds.append(cl.predict(X_test))
     scores.append(score)
     #print cl.score(X_test, y_test)

print np.mean(scores)
print confusion_matrix(targets, preds)





import pandas as pd
import numpy as np
import os

import pickle

from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv("data/processed/train_processed.csv")

X_train = train_data.drop(columns=['potability'],axis=1)
y_train = train_data['potability']

clf= RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

pickle.dump(clf,open("model.pkl","wb"))
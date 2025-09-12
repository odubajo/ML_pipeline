import pandas as pd
import numpy as np
import os
import yaml

import pickle

from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv("data/processed/train_processed.csv")

n_estimators = yaml.safe_load(open("params.yaml"))['model_building']['n_estimators']
min_samples_split = yaml.safe_load(open("params.yaml"))['model_building']['min_samples_split']

y_train = train_data['Potability']
X_train = train_data.drop(columns=['Potability'],axis=1)

clf= RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split ,random_state=42)
clf.fit(X_train, y_train)

pickle.dump(clf,open("model.pkl","wb"))
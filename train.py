# Spark configuration and packages specification. The dependencies defined in
# this file will be automatically provisioned for each run that uses Spark.

import sys
import os
import argparse

# Import frameworks
import pandas as pd
import pickle
import numpy as np
import xgboost

# Sklearn utils / metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import roc_curve, average_precision_score,precision_recall_curve

# For plotting
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# Load AML dataprep and logging
from azureml.dataprep.package import run
from azureml.logging import get_azureml_logger

# initialize the logger
logger = get_azureml_logger()

# add experiment arguments
parser = argparse.ArgumentParser()
# parser.add_argument('--arg', action='store_true', help='My Arg')
args = parser.parse_args()

# create the outputs folder
os.makedirs('./outputs', exist_ok=True)

print("Loading dataset...")
print()

################################################################
##### Load dataset from a DataPrep package as a pandas DataFrame
################################################################

df = run('train.dprep', dataflow_idx=0, spark=False)
df = df.dropna(how='any',axis=0)

# One hot encoding
df.loc[:,'one_way'] = df.one_way.astype('uint8')
ohe_fields=['one_way','surface_type','street_type','hour','weekday','month']
df_ohe = pd.get_dummies(df,columns=ohe_fields)

# Get the one-hot variable names
ohe_feature_names = pd.get_dummies(df[ohe_fields],columns=ohe_fields).columns.tolist()

# Names of the continuous features
float_feature_names = [
    'accident_counts',
    'speed_limit',
    'aadt',
    'surface_width',
    'sinuosity',
    'euclidean_length',
    'segment_length',
    'road_orient_approx',
    'precip_depth',
    'snow_depth',
    'temperature',
    'visibility',
    'wind_speed',
    'proximity_to_billboard',
    'proximity_to_major_road',
    'proximity_to_signal',
    'proximity_to_nearest_intersection',
    'population_density',
    'solar_azimuth',
    'solar_elevation',
]

# Sinuosity is typically close to 1, even for moderately curvy roads. A high sinuosity means a longer road.
feature_transforms = {
    'sinuosity': np.log
}
for feature,transform in feature_transforms.items():
    df_ohe[feature] = transform(df_ohe[feature])

# Get these features from the data frame
float_features = df_ohe.xs(float_feature_names,axis=1).values
# Scale to zero mean, unit variance using sci-kit learn's StandardScaler
scaler = StandardScaler()
# Fit the data transformation
float_scaled = scaler.fit_transform(float_features)

# Replace values with the scaled version
df_ohe[float_feature_names] = float_scaled

# Target Values
y = df['target'].values

binary_feature_names = [
    'snowing',
    'raining',
    'icy',
    'thunderstorm',
    'hailing',
    'foggy',
    'at_intersection',
]

# All the feature names
feature_names = float_feature_names+ohe_feature_names+binary_feature_names

# The scaled/transsformed training set
df_ohe = df_ohe.xs(feature_names,axis=1)

df.xs(float_feature_names+ohe_fields+binary_feature_names,axis=1).head(2).to_csv("./outputs/sample.csv")

# Save off data transformations
wrangler = {
    'scaler': scaler,
    'float_feature_names': float_feature_names,
    'ohe_fields': ohe_fields,
    'feature_names': feature_names,
    'feature_transforms': feature_transforms 
}
with open('./outputs/wrangler.pkl','wb') as fp:
    pickle.dump(wrangler,fp)


X = df_ohe.values

#########################################################################
#### Train Model
#########################################################################

print("Training Model...")

# Split data into 90% train, 10% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
# Load into XGBoost DMatrices
dtrain = xgboost.DMatrix(X_train,label=y_train,feature_names=feature_names)
dtest =  xgboost.DMatrix(X_test,label=y_test,feature_names=feature_names)

params = {
    'max_depth':6,
    'min_child_weight': 5.0,
    'reg_lambda': 1.0,
    'reg_alpha':0.0,
    'scale_pos_weight':1.0,
    'eval_metric':'auc',
    'objective':'binary:logistic',
    'eta':0.3
}
print ("Running xgboost.train...")
booster = xgboost.train(params,dtrain,
    evals = [(dtest, 'eval')],
    num_boost_round=3000,
    early_stopping_rounds=25
)

# Save model to outputs
booster.save_model('./outputs/0001.xgbmodel')

# Visualizations
fig = plt.figure(figsize=(20,30))
ax1 = fig.add_subplot(111)

xgboost.plot_importance(booster,ax=ax1,importance_type='weight')
plt.savefig('./outputs/importance.png',bbox_inches='tight')

y_pred_test = booster.predict(dtest) > 0.2 # Threshold with a decent guess as good precision/recall
logger.log('test_accuracy:',accuracy_score(y_test,y_pred_test))
logger.log('test_F1:',f1_score(y_test,y_pred_test))
logger.log('test_precision:',precision_score(y_test,y_pred_test))
logger.log('test_ap:',average_precision_score(y_test,y_pred_test))
logger.log('test_recall:',recall_score(y_test,y_pred_test))
y_pred_test = booster.predict(dtest)
logger.log('test_auc:',roc_auc_score(y_test,y_pred_test))


y_pred_train = booster.predict(dtrain) > 0.2
logger.log('train_accuracy:',accuracy_score(y_train,y_pred_train))
logger.log('train_F1:',f1_score(y_train,y_pred_train))
logger.log('train_precision:',precision_score(y_train,y_pred_train))
logger.log('train_ap:',average_precision_score(y_train,y_pred_train))
logger.log('train_recall:',recall_score(y_train,y_pred_train))
y_pred_train = booster.predict(dtrain)
logger.log('train_auc:',roc_auc_score(y_train,y_pred_train))

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test,y_pred_test)
fpr_train, tpr_train, thresholds_train = roc_curve(y_train,y_pred_train)
fig,ax = plt.subplots()
plt.plot([0,1],[0,1],'r-',label='Random Guess',color='orange',lw=3)
plt.plot(fpr,tpr,label='ROC (Test)',lw=3)
plt.plot(fpr_train,tpr_train,'r:',label='ROC (Train)',color='steelblue',lw=3)
plt.grid()
plt.legend()
plt.savefig('./outputs/roc.png',bbox_inches='tight')

plt.figure(figsize=(15,15))
plt.plot(thresholds,tpr,'r-',label='TPR (Test)',color='orange',lw=3)
plt.plot(thresholds_train,tpr_train,'r:',label='TPR (Train',color='orange',lw=3)
plt.plot(thresholds,fpr,'r-',label='FPR (Test)',color='steelblue',lw=3)
plt.plot(thresholds_train,fpr_train,'r:',label='FPR (Train)',color='steelblue',lw=3)
plt.gca().set_xbound(lower=0,upper=1)
plt.grid()
plt.legend()
plt.savefig('./outputs/tpr_fpr.png',bbox_inches='tight')

plt.figure(figsize=(15,15))

y_pred_test = booster.predict(dtest)
y_pred_train = booster.predict(dtrain)

xprecision,xrecall,xthresholds = precision_recall_curve(y_test,y_pred_test)
precision_train, recall_train, thresholds_train = precision_recall_curve(y_train,y_pred_train)
fig,ax = plt.subplots()
plt.plot(xprecision,xrecall,label='PR (Test)',lw=3)
plt.plot(precision_train,recall_train,label='PR (Train)',lw=3)

plt.grid()
plt.legend()
plt.savefig('./outputs/pr_curve.png',bbox_inches='tight')

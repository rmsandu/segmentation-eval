# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""

from sklearn.ensemble import RandomForestRegressor
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import utils.graphing as gh
from utils.scatter_plot import scatter_plot
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from utils import evaluate_performance

sns.set(style="ticks")
plt.style.use('ggplot')

#%%
df = pd.read_excel(r"C:\develop\segmentation-eval\Radiomics_Radii_MAVERRIC.xlsx")

idx_comments = df.columns.get_loc('Proximity_to_vessels')
# len(df.columns)
df_x = df.iloc[:, idx_comments:len(df.columns)].copy()
df_x.drop(columns=['Comments',
                   'Device_name', 'Ablation_Radii_Brochure', 'Ablation_Volume_Brochure', 'ablation_date',
                   'Ablation Volume [ml]_PCA_axes'], inplace=True)
# df['Ablation_Volume_Brochure'].replace(0, np.nan)
df_x.dropna(inplace=True)
y = df_x['Ablation Volume [ml]']
df_x.drop(columns=['Ablation Volume [ml]'], inplace=True)
X = df_x
# split into train and test size
#%%
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.30)
clf = RandomForestRegressor(n_estimators=100,
                              random_state=1, min_samples_leaf=10, min_samples_split=10,  oob_score=True)
# fit the classifier on the entire input data
clf.fit(X, y)
# predict the on the hold-out test data
predicted_labels = clf.predict(X_test)
# Calculate the absolute errors
errors = abs(predicted_labels - y_test)# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
#%%
print("Score of the training dataset obtained using an out-of-bag estimate:  %0.2f" % clf.oob_score_)

importances = list(clf.feature_importances_)# List of tuples with variable and importance
feature_list = X.columns.to_list()
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)# Print out the feature and importances
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
# evaluate_performance.plot_ROC_curve(clf, X, y)
# #  pr curve
# evaluate_performance.plot_PR_curve(clf, X, y,)
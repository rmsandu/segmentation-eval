# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from sklearn.model_selection import train_test_split, cross_val_predict

import utils.graphing as gh

sns.set(style="ticks")
plt.style.use('ggplot')

# %%
df_rf = pd.read_excel(r"C:\develop\segmentation-eval\Random_Forest\Radiomics_Radii_Chemo_LTP_RF.xlsx", sheet_name='rf')
df_centers = pd.read_excel(r"C:\develop\segmentation-eval\Random_Forest\Radiomics_MAVERRIC_random_forest_input.xlsx")
loocv_flag = True

x_label = 'Measured Ablation Volume [ml]'
y_label = 'Predicted Ablation Volume [ml]'
# %%

#%% extract needle error

df_centers['center_tumor'] = df_centers[
    ['center_of_mass_x_tumor', 'center_of_mass_y_tumor', 'center_of_mass_z_tumor']].values.tolist()
df_centers['TP_needle_1'] = df_centers['ValidationTargetPoint'].map(lambda x: x[1:len(x)-1])
df_centers['TP_needle'] = df_centers['TP_needle_1'].map(lambda x: np.array([float(i) for i in x.split()]))
# df_error_needle = np.linalg.norm(df_centers['TP_needle'] - df_centers['center_tumor'])
list_errors_needle = []
for row in df_centers.itertuples():
    try:
        needle_error = np.linalg.norm(row.TP_needle - row.center_tumor)
    except Exception:
        needle_error = np.nan
    list_errors_needle.append(needle_error)
print(len(list_errors_needle))
#%% RANDOM FOREST
df_rf['needle_error'] = list_errors_needle
df_rf['Device_name'] = df_centers['Device_name']
# q = df_rf['Ablation Volume [ml]'].quantile(0.99)
df1_no_outliers = df_rf[df_rf['Ablation Volume [ml]'] <= 100]
df1_no_outliers.reset_index(inplace=True, drop=True)
df = df1_no_outliers.copy()
# TODO: per device
# TODO: only non-subcapsular
df = df[df['Device_name'] == 'Angyodinamics (Acculis)']
idx_comments = df.columns.get_loc('Ablation Volume [ml]')
df_x = df.iloc[:, 0:idx_comments].copy()
df_x['Ablation Volume [ml]'] = df['Ablation Volume [ml]']
df_x['needle_error'] = df['needle_error']
df_x.dropna(inplace=True)
df_x['chemo_before_ablation'].replace('No', False, inplace=True)
df_x['chemo_before_ablation'].replace('Yes', True, inplace=True)
y = df_x['Ablation Volume [ml]']
df_x.drop(columns=['Ablation Volume [ml]'], inplace=True)
X = df_x
print('No. of feature used:', len(X.columns))
print('No. of training samples:', len(df_x))
# split into train and test size
# %%
n_estimators = 100
min_samples_leaf = 2
min_sample_split = 2
clf_h = RandomForestRegressor(n_estimators=n_estimators,
                              random_state=0,
                              min_samples_leaf=min_samples_leaf,
                              min_samples_split=min_samples_leaf,
                              oob_score=True)

# %%  PREDICT ON THE HOLD_OUT TEST DATA
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=0)

print('X train shape and Y train shape:', X_train.shape, y_train.shape)
print('X test shape and Y test shape:', X_test.shape, y_test.shape)
clf_h.fit(X_train, y_train)
importances = list(clf_h.feature_importances_)
predicted_labels_holdout_test = clf_h.predict(X_test)
print("Score of the training dataset obtained using an out-of-bag estimate:  %0.2f" % clf_h.oob_score_)
# print("Prediction computed with out-of-bag estimate on the training set:  " % clf.oob_prediction_)


feature_list = X_train.columns.to_list()
feature_importances = [(feature, round(importance, 2)) for feature, importance in
                       zip(feature_list, importances)]
feature_importances = sorted(feature_importances, key=lambda x: x[1],
                             reverse=True)
[print('Variable Train/Test Set 70/30: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

fig, ax = plt.subplots()
ax.scatter(y_test, predicted_labels_holdout_test, edgecolors=(0, 0, 0))
regr = linear_model.LinearRegression()
X_arr = np.array(y_test)
Y_arr = np.array(predicted_labels_holdout_test)
X_arr = X_arr.reshape(len(X_arr), 1)
Y_arr = Y_arr.reshape(len(Y_arr), 1)
regr.fit(X_arr, Y_arr)
SS_tot = np.sum((Y_arr - np.mean(Y_arr)) ** 2)
residuals = Y_arr - regr.predict(X_arr)
SS_res = np.sum(residuals ** 2)
r_squared = 1 - (SS_res / SS_tot)
correlation_coef = np.corrcoef(X_arr[:, 0], Y_arr[:, 0])[0, 1]
label = r'$R^2: $ {0:.2f}; r: {1:.2f}'.format(r_squared, correlation_coef)
plt.plot(X_arr, regr.predict(X_arr), color='black', lw=3, label=label)
# plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Ideal Fit')
plt.title('Prediction for Random Forest Model on Hold-Out. '
          'Train Samples:' + str(X_train.shape[0]) + '. Test Samples: ' + str(X_test.shape[0]) + '. Solero '
          , fontsize=6)
ax.set_xlabel(x_label, fontsize=10, color='k')
ax.set_ylabel(y_label, fontsize=10, color='k')
plt.legend(fontsize=10)
plt.tick_params(labelsize=10, color='black')
ax.tick_params(colors='black', labelsize=10)
plt.xlim([0, 100])
plt.ylim([0, 100])
plt.show()
figpathHist = os.path.join("figures", "Random_Forest_Model_Accuracy_Hold_OutTrain_Test_" + 'Min_samples_leaf_' + str(
    min_samples_leaf) + '_No_estimators_' + str(n_estimators))
gh.save(figpathHist, ext=['png'], close=True)
# %%
n_estimators = 100
min_samples_leaf = 2
min_sample_split = 2
clf = RandomForestRegressor(n_estimators=n_estimators,
                            random_state=0,
                            min_samples_leaf=min_samples_leaf,
                            min_samples_split=min_samples_leaf,
                            oob_score=True)

clf.fit(X, y)
# print("Score of the training dataset obtained using an out-of-bag estimate:  %0.2f" % clf.oob_score_)
# print("Prediction computed with out-of-bag estimate on the training set:  " % clf.oob_prediction_)
importances = list(clf.feature_importances_)
feature_list = X.columns.to_list()
feature_importances = [(feature, round(importance, 2)) for feature, importance in
                       zip(feature_list, importances)]
feature_importances = sorted(feature_importances, key=lambda x: x[1],
                             reverse=True)
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

# %%
n_folds = 10
if loocv_flag is True:
    n_folds = len(X)
predicted_crossval = cross_val_predict(clf, X, y, cv=n_folds)
fig, ax = plt.subplots()
plt.scatter(y, predicted_crossval, edgecolors=(0, 0, 0))
regr = linear_model.LinearRegression()
X_arr = np.array(y)
Y_arr = np.array(predicted_crossval)
X_arr = X_arr.reshape(len(X_arr), 1)
Y_arr = Y_arr.reshape(len(Y_arr), 1)
regr.fit(X_arr, Y_arr)
SS_tot = np.sum((Y_arr - np.mean(Y_arr)) ** 2)
residuals = Y_arr - regr.predict(X_arr)
SS_res = np.sum(residuals ** 2)
r_squared = 1 - (SS_res / SS_tot)
correlation_coef = np.corrcoef(X_arr[:, 0], Y_arr[:, 0])[0, 1]
label = r'$R^2: $ {0:.2f}; r: {1:.2f}'.format(r_squared, correlation_coef)
plt.plot(X_arr, regr.predict(X_arr), color='black', lw=3, label=label)
plt.xlim([0, 100])
plt.ylim([0, 100])
# ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
if n_folds == len(X):
    plt.title('Leave One Out Cross Validated Prediction for Random Forest Model. Number of folds: ' + str(
        n_folds) + '. Solero',
              fontsize=6)
else:
    plt.title(
        'Cross Validated Prediction for Random Forest Model. Number of folds: ' + str(n_folds) + '. Solero',
        fontsize=6)
ax.set_xlabel(x_label, fontsize=10, color='k')
ax.set_ylabel(y_label, fontsize=10, color='k')
plt.legend(fontsize=10)
plt.tick_params(labelsize=10, color='black')
ax.tick_params(colors='black', labelsize=10)
plt.show()
if n_folds == len(X):
    title = "Random_Forest_Model_Accuracy_LOOCV" + '_Min_samples_leaf_' + str(min_samples_leaf) + \
            " _No_estimators_" + str(n_estimators)
    figpathHist = os.path.join("figures", title)
else:
    title = "Random_Forest_Model_Accuracy_" + 'No_Of_Folds_' + str(n_folds) + "_Min_samples_leaf_" + str(
        min_samples_leaf) + "_No_estimators_ " + str(n_estimators)
gh.save(figpathHist, ext=['png'], close=True)

# %% Calculate the absolute errors
errors = abs(predicted_labels_holdout_test - y_test)  # Print out the mean absolute error (mae)
# print('Mean Absolute Error:', round(np.mean(errors), 2))
# # Calculate mean absolute percentage error (MAPE)
# mape = 100 * (errors / y_test)  # Calculate and display accuracy
# accuracy = 100 - np.mean(mape)
# print('Accuracy:', round(accuracy, 2), '%.')
print('Number of folds:' + str(n_folds) + ". Accuracy score:", round(clf.score(X_test, y_test), 2))
r2 = r2_score(y_test, predicted_labels_holdout_test)
print('R-square, coeff of determination:', round(r2, 2), '%.')
median_err = median_absolute_error(y_test, predicted_labels_holdout_test)
print('Median Squared Error:', round(median_err, 2), '%.')
mean_sq_err = mean_squared_error(y_test, predicted_labels_holdout_test)
print('Mean Squared Error:', round(mean_sq_err, 2), '%.')
mean_err = mean_absolute_error(y_test, predicted_labels_holdout_test)
print('Mean Abs Error:', round(mean_err, 2), '%.')
# %%

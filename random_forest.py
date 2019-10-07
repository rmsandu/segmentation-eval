# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import utils.graphing as gh
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import mean_absolute_error, max_error, explained_variance_score, mean_squared_error, r2_score, median_absolute_error


sns.set(style="ticks")
plt.style.use('ggplot')

# %%
df = pd.read_excel(r"C:\develop\segmentation-eval\Radiomics_Radii_Chemo_MAVERRIC.xlsx")

idx_comments = df.columns.get_loc('Proximity_to_vessels')
# len(df.columns)
df_x = df.iloc[:, idx_comments:len(df.columns)].copy()
df_x.drop(columns=['Comments', 'Device_name', 'Ablation_Radii_Brochure', 'ablation_date',
                   'Ablation Volume [ml]_PCA_axes', 'Tumour coverage ratio',
                   'Tumour residual volume [ml]', 'Volume Overlap Error',
                   'Volume Similarity', 'diameter2D_col_ablation', 'diameter2D_row_ablation',
                   'diameter2D_slice_ablation', 'diameter3D_ablation', 'elongation_ablation', 'sphericity_ablation',
                   'gray_lvl_nonuniformity_ablation', 'gray_lvl_variance_ablation', 'intensity_mean_ablation',
                   'intensity_uniformity_ablation', 'intensity_variance_ablation',
                   'least_axis_length_ablation', 'minor_axis_length_ablation',
                   'major_axis_length_ablation', 'safety_margin_distribution_0', 'safety_margin_distribution_10',
                   'safety_margin_distribution_5', 'first_axis_ablation_brochure',
                   'second_axis_ablation_brochure', 'third_axis_ablation_brochure', 'Ablation_Volume_Brochure'

                   ], inplace=True)

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
clf = RandomForestRegressor(n_estimators=n_estimators,
                            random_state=1,
                            min_samples_leaf=min_samples_leaf,
                            min_samples_split=min_samples_leaf,
                            oob_score=True)

clf.fit(X, y)
print("Score of the training dataset obtained using an out-of-bag estimate:  %0.2f" % clf.oob_score_)
print("Prediction computed with out-of-bag estimate on the training set:  " % clf.oob_prediction_)
importances = list(clf.feature_importances_)
feature_list = X.columns.to_list()
feature_importances = [(feature, round(importance, 2)) for feature, importance in
                       zip(feature_list, importances)]
feature_importances = sorted(feature_importances, key=lambda x: x[1],
                             reverse=True)
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

# %%# predict the on the hold-out test data
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3)

print('X train shape and Y train shape:', X_train.shape, y_train.shape)
print('X test shape and Y test shape:', X_test.shape, y_test.shape)

predicted_labels_holdout_test = clf.predict(X_test)


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
# ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
plt.title('Prediction for Random Forest Model on Hold-Out Train/Test Sample: 70/30. '
          'Min samples leaf:' + str(min_samples_leaf) + '. No. estimators: ' + str(n_estimators)
          , fontsize=10)
ax.set_xlabel('Measured Ablation Volume [ml]', fontsize=10, color='k')
ax.set_ylabel('Predicted Ablation Volume [ml]', fontsize=10, color='k')
plt.legend(fontsize=10)
plt.tick_params(labelsize=10, color='black')
ax.tick_params(colors='black', labelsize=10)
plt.show()
figpathHist = os.path.join("figures", "Random_Forest_Model_Accuracy_Hold_OutTrain_Test_" + 'Min_samples_leaf_' + str(
    min_samples_leaf) + '_No_estimators_' + str(n_estimators))
gh.save(figpathHist, ext=['png'], close=True)
# %%
n_folds = 3
predicted_crossval = cross_val_predict(clf, X, y, cv=n_folds)
fig, ax = plt.subplots()
ax.scatter(y, predicted_crossval, edgecolors=(0, 0, 0))
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
# ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
if n_folds == len(X):
    plt.title('Leave One Out Cross Validated Prediction for Random Forest Model. Number of folds: ' + str(n_folds),
          fontsize=10)
else:
    plt.title('Cross Validated Prediction for Random Forest Model. Number of folds: ' + str(n_folds),
          fontsize=10)
ax.set_xlabel('Measured Ablation Volume [ml]', fontsize=10, color='k')
ax.set_ylabel('Predicted Ablation Volume [ml]', fontsize=10, color='k')
plt.legend(fontsize=10)
plt.tick_params(labelsize=10, color='black')
ax.tick_params(colors='black', labelsize=10)
plt.show()
if n_folds == len(X):
    title = "Random_Forest_Model_Accuracy_LOOCV" + '_Min_samples_leaf_' + str(min_samples_leaf) + \
            " _No_estimators_" + str(n_estimators)
    figpathHist = os.path.join("figures", title)
else:
    title = "Random_Forest_Model_Accuracy_" + 'No_Of_Folds_'+ str(n_folds) + "_Min_samples_leaf_" + str(min_samples_leaf) + "_No_estimators_ " + str(n_estimators)
gh.save(figpathHist, ext=['png'], close=True)
# %% Calculate the absolute errors
errors = abs(predicted_labels_holdout_test - y_test)  # Print out the mean absolute error (mae)

# print('Mean Absolute Error:', round(np.mean(errors), 2))
# # Calculate mean absolute percentage error (MAPE)
# mape = 100 * (errors / y_test)  # Calculate and display accuracy
# accuracy = 100 - np.mean(mape)
# print('Accuracy:', round(accuracy, 2), '%.')
print("Accuracy score:", round(clf.score(X_test, y_test), 2))
r2 = r2_score(y_test, predicted_labels_holdout_test)
print('R-square, coeff of determination:', round(r2, 2), '%.')
median_err = median_absolute_error(y_test, predicted_labels_holdout_test)
print('Median Squared Error:', round(median_err, 2), '%.')
mean_sq_err = mean_squared_error(y_test, predicted_labels_holdout_test)
print('Mean Squared Error:', round(mean_sq_err, 2), '%.')
mean_err = mean_absolute_error(y_test, predicted_labels_holdout_test)
print('Mean Abs Error:', round(mean_err, 2), '%.')
# %%

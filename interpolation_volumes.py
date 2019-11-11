# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""
import os
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import utils.graphing as gh


def interpolation_fct(df_ablation, df_radiomics, title):
    x = np.asarray(df_ablation['Energy_brochure'])
    y = np.asarray(df_ablation['Ablation Volume [ml]_brochure'])
    f = interp1d(x, y, fill_value="extrapolate")
    df_radiomics.dropna(subset=['Ablation Volume [ml]'], inplace=True)
    energy = np.asarray(df_radiomics['Energy [kj]'])
    print('No of samples for ' + title + ': ', len(energy))
    ablation_vol_predicted_brochure = f(energy)
    fig, ax = plt.subplots()
    plt.title(title)
    plt.plot(x, y, 'o', energy, f(energy), '*')
    plt.legend(['data ' + title, 'linear interpolation'], loc='best')
    plt.xlabel('Energy [kJ]')
    plt.ylim([0, 120])
    plt.xlim([0, 80])
    plt.ylabel('Predicted Ablation Volume Brochure [ml]')
    plt.grid('on')
    plt.show()
    figpathHist = os.path.join("figures", title + '_interpolation')
    gh.save(figpathHist, ext=["png"], close=True)
    # PREDICTED VS MEASURED
    ablation_vol_calculated = np.asarray(df_radiomics['Ablation Volume [ml]'])
    print(len(ablation_vol_calculated))
    fig, ax = plt.subplots()
    size = [50*n for n in df_radiomics['Tumour Volume [ml]']]
    plt.scatter(ablation_vol_predicted_brochure, ablation_vol_calculated, marker='o',alpha=0.5, s=size)
    plt.ylabel('Measured Ablation Volume [ml]')
    plt.xlabel('Predicted Ablation Volume Brochure [ml]')
    plt.ylim([0, 100])
    plt.xlim([0, 100])
    plt.title(title + '  Nr. Samples: ' + str(len(ablation_vol_calculated)))
    plt.grid('on')
    X = ablation_vol_predicted_brochure.reshape(len(ablation_vol_predicted_brochure), 1)
    Y = ablation_vol_calculated.reshape(len(ablation_vol_calculated), 1)
    regr = linear_model.LinearRegression()
    regr.fit(X, Y)
    SS_tot = np.sum((Y - np.mean(Y)) ** 2)
    residuals = Y - regr.predict(X)
    SS_res = np.sum(residuals ** 2)
    r_squared = 1 - (SS_res / SS_tot)
    correlation_coef = np.corrcoef(X[:, 0], Y[:, 0])[0, 1]
    label = r'$R^2: $ {0:.2f}; r: {1:.2f}'.format(r_squared, correlation_coef)
    plt.plot(X, regr.predict(X), color='orange', linewidth=1.5, label=label)
    plt.legend(fontsize=10)
    figpathHist = os.path.join("figures", title + '_measured_vs_predicted_volume')
    gh.save(figpathHist, ext=["png"], close=True)

    # return energy_interp


if __name__ == '__main__':

    df_ablation = pd.read_excel(r"C:\develop\segmentation-eval\Ellipsoid_Brochure_Info.xlsx")
    df_radiomics = pd.read_excel(r"C:\develop\segmentation-eval\Radiomics_MAVERRIC_111119.xlsx")
    # sort values
    df_ablation.sort_values(by=['Energy_brochure'], inplace=True)
    df_amica = df_ablation[df_ablation['Device_name'] == 'Amica (Probe)']
    df_angyodinamics = df_ablation[df_ablation['Device_name'] == 'Angyodinamics (Acculis)']
    df_covidien = df_ablation[df_ablation['Device_name'] == 'Covidien (Covidien MWA)']
    df_radiomics.dropna(subset=['Energy [kj]'], inplace=True)
    df_radiomics.sort_values(by=['Energy [kj]'], inplace=True)
    df_radiomics = df_radiomics[(df_radiomics['Energy [kj]'] > 0) & (df_radiomics['Energy [kj]'] <= 70)]
    df_radiomics_amica = df_radiomics[df_radiomics['Device_name'] == 'Amica (Probe)']
    df_radiomics_angyodinamics = df_radiomics[df_radiomics['Device_name'] == 'Angyodinamics (Acculis)']
    df_radiomics_covidien = df_radiomics[df_radiomics['Device_name'] == 'Covidien (Covidien MWA)']

    interpolation_fct(df_amica, df_radiomics_amica, 'Amica')
    interpolation_fct(df_angyodinamics, df_radiomics_angyodinamics, 'Angyodinamics (Solero)')
    interpolation_fct(df_covidien, df_radiomics_covidien, 'Covidien')

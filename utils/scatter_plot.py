# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""
import utils.graphing as gh
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
from sklearn import linear_model
from sklearn.metrics import r2_score

sns.set(style="ticks")
plt.style.use('ggplot')


def scatter_plot(df1, **kwargs):
    """
    df, x_data, y_data, title, x_label=False, y_label='', lin_reg=''
    :param df:
    :param x_data:
    :param x_data:
    :param title:
    :param x_label:
    :param x_label:
    :param lin_reg:
    :return:
    """
    fig, ax = plt.subplots()
    if kwargs.get('x_data') is None:
        print('No X input data to plot')
        return
    if kwargs.get('y_data') is None:
        print('No Y input data to plot')
        return
    df = df1.copy()
    df.dropna(subset=[kwargs["x_data"]], inplace=True)
    df.dropna(subset=[kwargs["y_data"]], inplace=True)
    df.plot.scatter(x=kwargs["x_data"], y=kwargs["y_data"], s=80)
    if kwargs.get('x_label') is not None and kwargs.get('y_label') is None:
        plt.xlabel(kwargs['x_label'], fontsize=20)
        plt.ylabel(kwargs['y_data'], fontsize=20)
    elif kwargs.get('y_label') is not None and kwargs.get('x_label') is None:
        plt.ylabel(kwargs['y_label'], fontsize=20)
        plt.xlabel(kwargs['x_data'], fontsize=20)
    elif kwargs.get('x_label') is None and kwargs.get('y_label') is None:
        plt.xlabel(kwargs['x_data'], fontsize=20)
        plt.ylabel(kwargs['y_data'], fontsize=20)

    if kwargs.get('lin_reg') is not None:
        X = np.array(df[kwargs['x_data']])
        Y = np.array(df[kwargs['y_data']])
        regr = linear_model.LinearRegression()
        X = X.reshape(len(X), 1)
        Y = Y.reshape(len(Y), 1)
        regr.fit(X, Y)
        SS_tot = np.sum((Y-np.mean(Y))**2)
        residuals = Y - regr.predict(X)
        SS_res = np.sum(residuals**2)
        r_squared = 1-(SS_res/SS_tot)
        r_square_sklearn = r2_score(Y, regr.predict(X))
        print('R-square manual:', r_squared)
        print('R-square  sklearn:', r_square_sklearn)
        label = r'$R^2: $' + '{:.2f}'.format(r_squared)
        plt.plot(X, regr.predict(X), color='orange', linewidth=3, label=label)

    nr_samples = ' Nr. samples: ' + str(len(df))
    plt.title(kwargs['title'] + nr_samples, fontsize=20)
    plt.legend(fontsize=20)
    plt.tick_params(labelsize=20, color='black')
    ax.tick_params(colors='black', labelsize=20)
    figpathHist = os.path.join("figures", kwargs['title'])
    gh.save(figpathHist, width=18, height=16, ext=["png"], close=True)
    plt.close('all')



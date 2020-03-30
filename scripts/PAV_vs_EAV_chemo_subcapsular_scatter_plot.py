# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.interpolate import griddata

import utils.graphing as gh


def interpolation_fct(df_ablation, df_radiomics):
    """

    :param device:
    :param flag_hue:
    :param df_ablation:
    :param df_radiomics:
    :param fontsize:
    :return:
    """
    # perform interpolation as a function of  power and time (multivariate interpolation)
    points_power = np.asarray(df_ablation['Power']).reshape((len(df_ablation), 1))
    points_time = np.asarray(df_ablation['Time_Duration_Applied']).reshape((len(df_ablation), 1))
    power_and_time_brochure = np.hstack((points_power, points_time))
    ablation_vol_brochure = np.asarray(df_ablation['Predicted Ablation Volume (ml)']).reshape((len(df_ablation), 1))
    df_radiomics.dropna(subset=['Power', 'Time_Duration_Applied'], inplace=True)
    grid_x = df_radiomics['Power'].to_numpy()
    grid_y = df_radiomics['Time_Duration_Applied'].to_numpy()
    grid_x = np.array(pd.to_numeric(grid_x, errors='coerce'))
    grid_y = np.array(pd.to_numeric(grid_y, errors='coerce'))
    grid_x = grid_x.reshape(len(grid_x), 1)
    grid_y = grid_y.reshape(len(grid_y), 1)
    power_and_time_effective = np.asarray(np.hstack((grid_x, grid_y)))

    ablation_vol_interpolated_brochure = griddata(power_and_time_brochure, ablation_vol_brochure,
                                                  power_and_time_effective, method='linear')
    ablation_vol_interpolated_brochure = ablation_vol_interpolated_brochure.reshape(len(df_radiomics), )
    ablation_vol_measured = np.asarray(df_radiomics['Ablation Volume [ml]']).reshape(len(df_radiomics), )
    # sanity check that both EAV and PAV have the same length (even if NaNs present)
    if len(ablation_vol_interpolated_brochure) != len(ablation_vol_measured):
        print("something's not right")
        return None
    return ablation_vol_interpolated_brochure

    # # %% PLOT BOXPLOTS
    # plot_boxplots_volumes(ablation_vol_interpolated_brochure, ablation_vol_measured,
    #                                                flag_subcapsular=False)


def edit_save_plot(ax=None, p=None, flag_hue=None, xlabel='PAV', ylabel='EAV', device='Acculis pMTA MWA System',
                   r_1=None, r_2=None,
                   label_2=None, label_3=None,
                   ratio_flag=False):
    """

    :param ax:
    :param p:
    :param flag_hue:
    :param ylabel:
    :param device:
    :param r_1:
    :param r_2:
    :param label_2:
    :param label_3:
    :return:
    """
    fontsize = 20
    if flag_hue in ['vessels', 'subcapsular', 'chemotherapy', 'Tumor_Vol']:
        ax = p.axes[0, 0]
        ax.legend(fontsize=fontsize, title_fontsize=fontsize, title=device, loc='upper left')
        leg = ax.get_legend()
        L_labels = leg.get_texts()
        label_line_1 = r'$R^2:{0:.2f}$'.format(r_1)
        label_line_2 = r'$R^2:{0:.2f}$'.format(r_2)
        L_labels[0].set_text(label_line_1)
        L_labels[1].set_text(label_line_2)
        L_labels[2].set_text(label_2)
        L_labels[3].set_text(label_3)
    else:
        ax.legend(fontsize=fontsize, title_fontsize=fontsize, title=device, loc='upper left')

    if ratio_flag is False:
        plt.xlim([0, 50])
        plt.ylim([0, 50])

    # plt.xlim([8, 46])
    # plt.ylim([-0.2, 3.5])
    # plt.xlim([1, 60])
    # plt.ylim([-0.5, 60])
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)

    figpath = os.path.join("figures", device + '__EAV_parametrized_PAV_groups_' + str(flag_hue))
    gh.save(figpath, width=12, height=12, ext=["png"], close=True, tight=True, dpi=600)


def plot_scatter_group_var_chemo(df_radiomics, ablation_vol_interpolated_brochure, ratio_flag=False):
    """

    :param df_radiomics:
    :param ablation_vol_interpolated_brochure:
    :param ratio_flag:
    :return:
    """
    df_radiomics.loc[df_radiomics.no_chemo_cycle > 0, 'no_chemo_cycle'] = 'Yes'
    df_radiomics.loc[df_radiomics.no_chemo_cycle == 0, 'no_chemo_cycle'] = 'No'
    # create new pandas DataFrame for easier plotting
    df = pd.DataFrame()
    df['PAV'] = ablation_vol_interpolated_brochure
    df['EAV'] = df_radiomics['Ablation Volume [ml]']
    df['Energy (kJ)'] = df_radiomics['Energy [kj]']
    df['Chemotherapy'] = df_radiomics['no_chemo_cycle']
    df['R(EAV:PAV)'] = df['EAV'] / df['PAV']
    df.dropna(inplace=True)
    print('Nr Samples used:', str(len(df)))
    label_2 = 'Chemotherapy: No'
    label_3 = 'Chemotherapy: Yes'
    chemo_true = df[df['Chemotherapy'] == 'Yes']
    chemo_false = df[df['Chemotherapy'] == 'No']
    if ratio_flag is True:
        p = sns.lmplot(y="R(EAV:PAV)", x="Energy (kJ)", hue="Chemotherapy", data=df, markers=["s", "s"],
                       palette=['mediumvioletred', 'green'],
                       ci=None, scatter_kws={"s": 150, "alpha": 0.8}, line_kws={'label': 'red'},
                       legend=True, legend_out=False)
        x1 = chemo_false['Energy (kJ)']
        y1 = chemo_false['R(EAV:PAV)']
        x2 = chemo_true['Energy (kJ)']
        y2 = chemo_true['R(EAV:PAV)']
        slope, intercept, r_2, p_value, std_err = stats.linregress(x1, y1)
        slope, intercept, r_1, p_value, std_err = stats.linregress(x2, y2)
        edit_save_plot(p=p, flag_hue='chemotherapy', ylabel='R(EAV:PAV)', xlabel='Energy (kJ)', device='Acculis pMTA',
                       r_1=r_1, r_2=r_2, label_2=label_2, label_3=label_3, ratio_flag=True)
    elif ratio_flag is False:
        p = sns.lmplot(y="EAV", x="PAV", hue="Chemotherapy", data=df, markers=["s", "s"],
                       palette=['mediumvioletred', "green"],
                       ci=None, scatter_kws={"s": 150, "alpha": 0.8}, line_kws={'label': 'red'},
                       legend=True, legend_out=False)
        x1 = chemo_false['PAV']
        y1 = chemo_false['EAV']
        x2 = chemo_true['PAV']
        y2 = chemo_true['EAV']
        slope, intercept, r_2, p_value, std_err = stats.linregress(x1, y1)
        slope, intercept, r_1, p_value, std_err = stats.linregress(x2, y2)
        edit_save_plot(p=p, flag_hue='chemotherapy', ylabel='Effective Ablation Treatment (mL)',
                       xlabel='Predicted Ablation Treatment (mL)', device='Acculis pMTA',
                       r_1=r_1, r_2=r_2, label_2=label_2, label_3=label_3)


def plot_scatter_group_var_tumor_vol(df_radiomics, ablation_vol_interpolated_brochure):
    """

    :param df_radiomics:
    :param ablation_vol_interpolated_brochure:
    :return:
    """
    df = pd.DataFrame()
    df['Tumor_Vol'] = df_radiomics['Tumour Volume [ml]']
    df['PAV'] = ablation_vol_interpolated_brochure
    df['EAV'] = df_radiomics['Ablation Volume [ml] (parametrized_formula)']
    df['Energy (kJ)'] = df_radiomics['Energy [kj]']
    df['R(EAV:PAV)'] = df['EAV'] / df['PAV']
    df.dropna(inplace=True)
    print('Nr Samples used:', str(len(df)))

    p = sns.lmplot(y="R(EAV:PAV)", x="Tumor_Vol", hue="Chemotherapy", data=df, markers=["X", "s"],
                   palette=['orange'],
                   ci=None, scatter_kws={"s": 150, "alpha": 0.5}, line_kws={'label': 'red'},
                   legend=True, legend_out=False)
    chemo_true = df[df['Chemotherapy'] == 'Yes']
    chemo_false = df[df['Chemotherapy'] == 'No']
    slope, intercept, r_1, p_value, std_err = stats.linregress(chemo_false['Tumor_Vol'],
                                                               chemo_false['R(EAV:PAV)'])
    slope, intercept, r_2, p_value, std_err = stats.linregress(chemo_true['Tumor_Vol'],
                                                               chemo_true['R(EAV:PAV)'])
    label_2 = 'Chemotherapy: No'
    label_3 = 'Chemotherapy: Yes'
    edit_save_plot(p=p, flag_hue='Tumor_Vol', ylabel='R(EAV:PAV)', xlabel='Tumor Volume (ml)', device='Acculis pMTA',
                   r_1=r_1, r_2=r_2, label_2=label_2, label_3=label_3)


def plot_scatter_group_var_subcapsular(df_radiomics, ablation_vol_interpolated_brochure, ratio_flag=True):
    """

    :param df_radiomics:
    :param ablation_vol_interpolated_brochure:
    :return:
    """
    df = pd.DataFrame()
    df['PAV'] = ablation_vol_interpolated_brochure
    df['EAV'] = df_radiomics['Ablation Volume [ml]']
    df['Energy (kJ)'] = df_radiomics['Energy [kj]']
    df['Subcapsular'] = df_radiomics['Proximity_to_surface']
    df['R(EAV:PAV)'] = df['EAV'] / df['PAV']
    df.dropna(inplace=True)
    print('Nr Samples used:', str(len(df)))
    label_2 = 'Deep Tumors'
    label_3 = 'Subcapsular'
    subcapsular_true = df[df['Subcapsular'] == True]
    subcapsular_false = df[df['Subcapsular'] == False]
    if ratio_flag is True:
        p = sns.lmplot(y="R(EAV:PAV)", x="Energy (kJ)", hue="Subcapsular", data=df, markers=["*", "*"],
                       palette=['cornflowerblue', 'orange'],
                       ci=None, scatter_kws={"s": 150, "alpha": 0.8}, line_kws={'label': 'red'},
                       legend=True, legend_out=False)
        slope, intercept, r_1, p_value, std_err = stats.linregress(subcapsular_false['Energy (kJ)'],
                                                                   subcapsular_false['R(EAV:PAV)'])
        slope, intercept, r_2, p_value, std_err = stats.linregress(subcapsular_true['Energy (kJ)'],
                                                                   subcapsular_true['R(EAV:PAV)'])
        edit_save_plot(p=p, flag_hue='subcapsular', ylabel='R(EAV:PAV)', xlabel='Energy (kJ)', device='Acculis pMTA',
                       r_1=r_1, r_2=r_2, label_2=label_2, label_3=label_3)
    else:
        p = sns.lmplot(y="EAV", x="PAV", hue="Subcapsular", data=df, markers=["*", "*"],
                       palette=['cornflowerblue', 'orange'],
                       ci=None, scatter_kws={"s": 150, "alpha": 0.8}, line_kws={'label': 'red'},
                       legend=True, legend_out=False)

        slope, intercept, r_1, p_value, std_err = stats.linregress(subcapsular_false['PAV'],
                                                                   subcapsular_false['EAV'])
        slope, intercept, r_2, p_value, std_err = stats.linregress(subcapsular_true['PAV'],
                                                                   subcapsular_true['EAV'])
        edit_save_plot(ax=None, p=p, flag_hue='subcapsular', ylabel='Effective Ablation Volume (mL)',
                       xlabel='Predicted Ablation Volume (mL)', device='Acculis pMTA',
                       r_1=r_1, r_2=r_2, label_2=label_2, label_3=label_3)


def plot_scatter_group_var_vessels(df_radiomics, ablation_vol_interpolated_brochure):
    """

    :param df_radiomics: DataFrame
    :param ablation_vol_interpolated_brochure:
    :return: nothing
    """
    df = pd.DataFrame()
    df['PAV'] = ablation_vol_interpolated_brochure
    df['EAV'] = df_radiomics['Ablation Volume [ml] (parametrized_formula)']
    df['Energy (kJ)'] = df_radiomics['Energy [kj]']
    df['Proximity_to_vessels'] = df_radiomics['Proximity_to_vessels']
    df['R(EAV:PAV)'] = df['EAV'] / df['PAV']
    df.dropna(inplace=True)
    print('Nr Samples used:', str(len(df)))
    p = sns.lmplot(y="R(EAV:PAV)", x="Energy (kJ)", hue="Proximity_to_vessels", data=df, markers=["*", "s"],
                   ci=None, scatter_kws={"s": 150, "alpha": 0.5}, line_kws={'label': 'red'},
                   legend=True, legend_out=False)
    subcapsular_true = df[df['Proximity_to_vessels'] == False]
    subcapsular_false = df[df['Proximity_to_vessels'] == True]
    slope, intercept, r_1, p_value, std_err = stats.linregress(subcapsular_false['Energy (kJ)'],
                                                               subcapsular_false['R(EAV:PAV)'])
    slope, intercept, r_2, p_value, std_err = stats.linregress(subcapsular_true['Energy (kJ)'],
                                                               subcapsular_true['R(EAV:PAV)'])
    label_2 = 'Proximity to vessels: No'
    label_3 = 'Proximity to vessels: Yes'
    # TODO: add the code for plotting (create function for plotting)


def plot_scatter_pav_eav(df_radiomics,
                         ablation_vol_interpolated_brochure, ratio_flag=False,
                         linear_regression=True):
    """

    :param df_radiomics:
    :param ablation_vol_interpolated_brochure:
    :param ratio_flag:
    :return:
    """
    df = pd.DataFrame()
    df['PAV'] = ablation_vol_interpolated_brochure
    df['EAV'] = df_radiomics['Ablation Volume [ml]']
    df['Energy (kJ)'] = df_radiomics['Energy [kj]']

    if ratio_flag is False:
        if linear_regression is True:
            df.dropna(inplace=True)
            slope, intercept, r_square, p_value, std_err = stats.linregress(df['EAV'], df['PAV'])
            ax = sns.regplot(x="PAV", y="EAV", data=df, scatter_kws={"s": 150, "alpha": 0.8},
                             color=sns.xkcd_rgb["violet"],
                             line_kws={'label': r'$R^2:{0:.4f}$'.format(r_square)})
        else:
            ax = sns.scatterplot(x="PAV", y="EAV", data=df, s=200, alpha=0.8,
                                 color=sns.xkcd_rgb["violet"])

        edit_save_plot(ax=ax, ylabel="Effective Ablation Volume (mL)", xlabel="Predicted Ablation Volume (mL)")

    else:
        df['R(EAV:PAV)'] = df['EAV'] / df['PAV']
        df.dropna(inplace=True)
        slope, intercept, r_square, p_value, std_err = stats.linregress(df['R(EAV:PAV)'], df['Energy (kJ)'])

        if linear_regression is True:
            ax = sns.regplot(y="R(EAV:PAV)", x="Energy (kJ)", data=df, color=sns.xkcd_rgb["violet"],
                             line_kws={'label': r'$R^2:{0:.4f}$'.format(r_square)},
                             scatter_kws={"s": 150, "alpha": 0.8})
        else:
            ax = sns.scatterplot(x="R(EAV:PAV)", y="Energy (kJ)", data=df, scatter_kws={"s": 150, "alpha": 0.8},
                                 color=sns.xkcd_rgb["violet"])

        edit_save_plot(ax=ax, ylabel="R(EAV:PAV)", xlabel="Energy (kJ)", device='Acculis pMTA')


def connect_points(x, y, p1, p2):
    x1, x2 = x[p1], x[p2]
    y1, y2 = y[p1], y[p2]
    plt.plot([x, x2], [x, y2], 'k-')


def connected_mev_miv(df_radiomics, ablation_vol_interpolated_brochure):
    """
    Plots ablation volumes.
    :param df_radiomics: DataFrame containing tabular radiomics features
    :param ablation_vol_interpolated_brochure:  column-like interpolated ablation volume from the brochure
    :return: Plot, saved as a PNG image
    """
    font = {'family': 'DejaVu Sans',
            'size': 16}

    matplotlib.rc('font', **font)
    df = pd.DataFrame()
    df['PAV'] = ablation_vol_interpolated_brochure
    df['EAV'] = df_radiomics_acculis['Ablation Volume [ml]']
    # MIV not ready
    df['MIV'] = df_radiomics['Inner Ellipsoid Volume']
    df['MEV'] = df_radiomics['Outer Ellipsoid Volume']
    df = df[df['MEV'] < 100]
    df.dropna(inplace=True)
    # plot scatter plots on the same y axis then connect them with a vertical line
    fig, ax = plt.subplots()
    MIV = np.asarray(df['MIV'])
    MEV = np.asarray(df['MEV'])
    PAV = np.asarray(df['PAV'])
    EAV = np.asarray(df['EAV'])
    # x = df['PAV']
    x = np.asarray([i for i in range(1, len(MEV) + 1)])
    ax.scatter(x, MIV, marker='o', color='green', label='Maximum Inscribed Ellipsoid')
    # ax.scatter(x, EAV, marker='o', color='orange', label='Effective Ablation Volume')
    ax.scatter(x, MEV, marker='o', color='blue', label='Minimum Enclosing Ellipsoid')
    plt.legend(loc='upper left')
    plt.ylabel('Volume (mL)')

    for i in np.arange(0, len(x)):
        x1, x2 = x[i], x[i]
        y1, y2 = MIV[i], MEV[i]
        plt.plot([x1, x2], [y1, y2], 'k-')

    # plt.ylim([0, 200])
    labels = np.round(np.asarray(df['PAV']))
    # plt.xticks(x, labels, rotation=45, fontsize=24)
    # plt.xlabel('Predicted Ablation Volume (ml)', fontsize=20)

    gh.save(r'C:\develop\segmentation-eval\figures\MIV_MEV_ellipsoids_1', width=12, height=12, ext=["png"], close=True,
            tight=True, dpi=600)

    # df.columns = ['Predicted Ablation', 'Effective Ablation', 'Maximum Inscribed Ellipsoid', 'Minimum Enclosing Ellipsoid']
    b = sns.boxplot(data=df)
    plt.ylabel('Volume (mL)')
    # b.set_xlabel(fontsize=20)
    # plt.ylim([])
    # plt.grid()
    # plt.show()
    gh.save(r'C:\develop\segmentation-eval\figures\boxplots_ellipsoids', width=12, height=12, ext=["png"], close=True,
            tight=True, dpi=600)


if __name__ == '__main__':
    df_ablation_brochure = pd.read_excel(r"C:\develop\segmentation-eval\Ellipsoid_Brochure_Info.xlsx")
    df_radiomics = pd.read_excel(r"C:\develop\segmentation-eval\Radiomics_MAVERRIC_153011-20200313_.xlsx")
    df_acculis = df_ablation_brochure[df_ablation_brochure['Device_name'] == 'Angyodinamics (Acculis)']
    df_acculis.reset_index(inplace=True)
    df_radiomics_acculis = df_radiomics[df_radiomics['Device_name'] == 'Angyodinamics (Acculis)']
    df_radiomics_acculis.reset_index(inplace=True)
    # df_radiomics = df_radiomics[df_radiomics['Proximity_to_surface'] == False]
    df_radiomics_acculis = df_radiomics_acculis[(df_radiomics_acculis['Comments'].isnull())]

    # flag_hue='chemotherapy'
    # %% extract the needle error
    ablation_vol_interpolated_brochure = interpolation_fct(df_acculis, df_radiomics_acculis)

    df_stats = pd.DataFrame()
    df_stats['PAV'] = ablation_vol_interpolated_brochure
    df_stats['EAV'] = df_radiomics_acculis['Ablation Volume [ml]']
    df_stats['Energy'] = df_radiomics_acculis['Energy [kj]']
    df_stats['Outer_Ellipsoid_Volume'] = df_radiomics_acculis['Outer Ellipsoid Volume']
    df_stats['Sphericity'] = df_radiomics_acculis['sphericity_ablation']
    df_stats['major_axis'] = df_radiomics_acculis['major_axis_length_ablation']
    df_stats_1 = df_stats.describe()
    filepath_excel = 'Radiomics_MAVERRIC_Descriptive_Stats.xlsx'
    writer = pd.ExcelWriter(filepath_excel)
    df_stats_1.to_excel(writer, sheet_name='radiomics', index=True, float_format='%.2f')
    writer.save()
    # %% PLOTS
    font = {'family': 'DejaVu Sans',
            'size': 20}

    matplotlib.rc('font', **font)
    connected_mev_miv(df_radiomics, ablation_vol_interpolated_brochure)
    # plot_scatter_pav_eav(df_radiomics_acculis, ablation_vol_interpolated_brochure, ratio_flag=False,
    #                      linear_regression=False)
    # plot_scatter_pav_eav(df_radiomics_acculis, ablation_vol_interpolated_brochure, ratio_flag=True,
    #                      linear_regression=True)
    # plot_scatter_group_var_subcapsular(df_radiomics_acculis, ablation_vol_interpolated_brochure, ratio_flag=False)
    # plot_scatter_group_var_subcapsular(df_radiomics_acculis, ablation_vol_interpolated_brochure, ratio_flag=True)

    # plot_scatter_group_var_chemo(df_radiomics_acculis, ablation_vol_interpolated_brochure, ratio_flag=False)
    # plot_scatter_group_var_chemo(df_radiomics_acculis, ablation_vol_interpolated_brochure, ratio_flag=True)

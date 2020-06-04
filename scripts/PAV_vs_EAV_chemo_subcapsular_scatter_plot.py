# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""
import os
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import iqr
import scripts.subplots_pav_eav_scatter as subplots_pav_eav
import scripts.mev_miv_scatter as mev_miv_scatter
import utils.graphing as gh


# # %% PLOT BOXPLOTS
# plot_boxplots_volumes(ablation_vol_interpolated_brochure, ablation_vol_measured,
#                                                flag_subcapsular=False)


def edit_save_plot(ax=None, p=None, flag_hue=None, xlabel='PAV', ylabel='EAV', device='',
                   r_1=None, r_2=None,
                   label_1=None, label_2=None,
                   ratio_flag=False):
    """

    :param ax:
    :param p:
    :param flag_hue:
    :param ylabel:
    :param device:
    :param r_1:
    :param r_2:
    :param label_1:
    :param label_2:
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
        L_labels[2].set_text(label_1)
        L_labels[3].set_text(label_2)
    else:
        ax.legend(fontsize=fontsize, title_fontsize=fontsize, title=device, loc='upper right')
        # ax.legend(fontsize=fontsize,  loc='upper right')

    if ratio_flag is False:
        plt.xlim([0, 100])
        plt.ylim([0, 100])

    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    timestr = time.strftime("%H%M%S-%Y%m%d")
    figpath = os.path.join("figures", device + '__EAV_parametrized_PAV_groups_' + str(flag_hue) + '-' + timestr)
    gh.save(figpath, width=12, height=12, ext=["png"], close=True, tight=True, dpi=600)


def plot_scatter_group_var_chemo(df_radiomics, ratio_flag=False):
    """

    :param df_radiomics:
    :param ratio_flag:
    :return:
    """
    df_radiomics.loc[df_radiomics.no_chemo_cycle > 0, 'no_chemo_cycle'] = 'Yes'
    df_radiomics.loc[df_radiomics.no_chemo_cycle == 0, 'no_chemo_cycle'] = 'No'
    # create new pandas DataFrame for easier plotting
    df = pd.DataFrame()
    df['PAV'] = df_radiomics['Predicted_Ablation_Volume']
    df['EAV'] = df_radiomics['Ablation Volume [ml]']
    df['Energy (kJ)'] = df_radiomics['Energy [kj]']
    df['Chemotherapy'] = df_radiomics['no_chemo_cycle']
    df['R(EAV:PAV)'] = df['EAV'] / df['PAV']
    df.dropna(inplace=True)
    print('Nr Samples used:', str(len(df)))
    label_2 = 'Chemotherapy: No'
    label_3 = 'Chemotherapy: Yes'
    chemo_false = df[df['Chemotherapy'] == 'No']
    chemo_true = df[df['Chemotherapy'] == 'Yes']
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
        edit_save_plot(p=p, flag_hue='chemotherapy', ylabel='R(EAV:PAV)', xlabel='Energy (kJ)',
                       r_1=r_1, r_2=r_2, label_1=label_2, label_2=label_3, ratio_flag=True)
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
                       xlabel='Predicted Ablation Treatment (mL)',
                       r_1=r_1, r_2=r_2, label_1=label_2, label_2=label_3)


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
    edit_save_plot(p=p, flag_hue='Tumor_Vol', ylabel='R(EAV:PAV)', xlabel='Tumor Volume (ml)',
                   r_1=r_1, r_2=r_2, label_2=label_2, label_3=label_3)


def plot_scatter_group_var_subcapsular(df_radiomics, ratio_flag=False):
    """

    :param df_radiomics:
    :param ablation_vol_interpolated_brochure:
    :param ratio_flag:
    :return:
    """
    df = pd.DataFrame()
    df['PAV'] = df_radiomics['Predicted_Ablation_Volume']
    df['EAV'] = df_radiomics['Ablation Volume [ml]']
    df['Energy (kJ)'] = df_radiomics['Energy [kj]']
    df['Subcapsular'] = df_radiomics['Proximity_to_surface']
    df['R(EAV:PAV)'] = df['EAV'] / df['PAV']
    df.dropna(inplace=True)
    print('Nr Samples used:', str(len(df)))
    label_1 = 'Deep Tumors'  # Non-subcapsular aka Subcapsular False
    label_2 = 'Subcapsular'
    subcapsular_false = df[df['Subcapsular'] == False]
    subcapsular_true = df[df['Subcapsular'] == True]

    if ratio_flag is True:
        p = sns.lmplot(y="R(EAV:PAV)", x="Energy (kJ)", hue="Subcapsular", data=df, markers=["*", "*"],
                       palette=['cornflowerblue', 'orange'],
                       ci=None, scatter_kws={"s": 150, "alpha": 0.8}, line_kws={'label': 'red'},
                       legend=True, legend_out=False)
        slope, intercept, r_1, p_value, std_err = stats.linregress(subcapsular_false['Energy (kJ)'],
                                                                   subcapsular_false['R(EAV:PAV)'])
        slope, intercept, r_2, p_value, std_err = stats.linregress(subcapsular_true['Energy (kJ)'],
                                                                   subcapsular_true['R(EAV:PAV)'])
        edit_save_plot(p=p, flag_hue='subcapsular', ylabel='R(EAV:PAV)', xlabel='Energy (kJ)',
                       r_1=r_1, r_2=r_2, label_2=label_1, label_3=label_2)
    else:
        p = sns.lmplot(y="EAV", x="PAV", hue="Subcapsular", data=df, markers=["*", "*"],
                       palette=['cornflowerblue', 'orange'],
                       ci=None, scatter_kws={"s": 150, "alpha": 0.8}, line_kws={'label': 'red'},
                       legend=True, legend_out=False)
        # first legend false
        slope, intercept, r_1, p_value, std_err = stats.linregress(subcapsular_false['PAV'],
                                                                   subcapsular_false['EAV'])
        slope, intercept, r_2, p_value, std_err = stats.linregress(subcapsular_true['PAV'],
                                                                   subcapsular_true['EAV'])
        edit_save_plot(ax=None, p=p, flag_hue='subcapsular', ylabel='Effective Ablation Volume (mL)',
                       xlabel='Predicted Ablation Volume (mL)',
                       r_1=r_1, r_2=r_2, label_1=label_1, label_2=label_2)


def plot_scatter_group_var_vessels(df_radiomics):
    """

    :param df_radiomics: DataFrame
    :param ablation_vol_interpolated_brochure:
    :return: nothing
    """
    df = pd.DataFrame()
    df['PAV'] = df_radiomics['Predicted_Ablation_Volume']
    df['EAV'] = df_radiomics['Ablation Volume [ml]']
    df['Energy (kJ)'] = df_radiomics['Energy [kj]']
    df['Proximity to vessels > 5mm'] = df_radiomics['Proximity_to_vessels']
    df.dropna(inplace=True)
    print('Nr Samples used:', str(len(df)))

    ax = sns.scatterplot(x="PAV", y="EAV", data=df, s=200, alpha=0.8, hue='Proximity to vessels > 5mm')
    edit_save_plot(ax=ax, ylabel="Effective Ablation Volume (mL)", xlabel="Predicted Ablation Volume (mL)")


def plot_scatter_pav_eav(df_radiomics,
                         ratio_flag=False,
                         linear_regression=True):
    """

    :param df_radiomics:
    :param ratio_flag:
    :return:
    """
    df = pd.DataFrame()
    df['PAV'] = df_radiomics['Predicted_Ablation_Volume']
    df['EAV'] = df_radiomics['Ablation Volume [ml]']
    df['Energy (kJ)'] = df_radiomics['Energy [kj]']
    df['MWA Systems'] = df_radiomics['Device_name']
    df['major_axis_length_ablation'] = df_radiomics['major_axis_length_ablation']
    df.dropna(inplace=True)
    print('Nr Samples used for PAV vs EAV scatter plot:', str(len(df)))
    if ratio_flag is False:
        if linear_regression is True:
            df.dropna(inplace=True)
            slope, intercept, r_square, p_value, std_err = stats.linregress(df['EAV'], df['PAV'])
            ax = sns.regplot(x="PAV", y="EAV", data=df, scatter_kws={"s": 150, "alpha": 0.8},
                             color=sns.xkcd_rgb["violet"],
                             line_kws={'label': r'$r={0:.4f}$'.format(r_square)})
        else:
            ax = sns.scatterplot(x="PAV", y="EAV", data=df, s=200, alpha=0.8,
                                 color=sns.xkcd_rgb["violet"],  hue='MWA Systems')

        edit_save_plot(ax=ax, ylabel="Effective Ablation Volume (mL)", xlabel="Predicted Ablation Volume (mL)")

    else:
        df['R(EAV:PAV)'] = df['EAV'] / df['PAV']
        df.dropna(inplace=True)
        slope, intercept, r_square, p_value, std_err = stats.linregress(df['major_axis_length_ablation'], df['Energy (kJ)'])
        print('p-value R(EAV:PAV) vs Energy (kj):', p_value)
        if linear_regression is True:
            ax = sns.regplot(y="major_axis_length_ablation", x="Energy (kJ)", data=df, color=sns.xkcd_rgb["cobalt"],
                             line_kws={'label': r'$ r = {0:.2f}$'.format(r_square)},
                             scatter_kws={"s": 150, "alpha": 0.8})
        else:
            ax = sns.scatterplot(x="R(EAV:PAV)", y="Energy (kJ)", data=df, scatter_kws={"s": 150, "alpha": 0.8},
                                 color=sns.xkcd_rgb["violet"], hue='MWA Systems')

        edit_save_plot(ax=ax, ylabel="Maximum ablation diameter (mm)", xlabel="Energy (kJ)", ratio_flag=True)


def connected_mev_miv(df_radiomics):
    """
    Plots ablation volumes.
    :param df_radiomics: DataFrame containing tabular radiomics features
    :param ablation_vol_interpolated_brochure:  column-like interpolated ablation volume from the brochure
    :return: Plot, saved as a PNG image
    """

    df = pd.DataFrame()
    df['PAV'] = df_radiomics['Predicted_Ablation_Volume']
    df['EAV'] = df_radiomics['Ablation Volume [ml]']
    # df['MIV'] = df_radiomics['Inner Ellipsoid Volume']
    df['MIV'] = df_radiomics['Outer Ellipsoid Volume'] / 3
    df['MEV'] = df_radiomics['Outer Ellipsoid Volume']
    df = df[df['MEV'] < 150]
    # discard
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
    ax.scatter(x, EAV, marker='o', color='orange', label='Effective Ablation Volume')
    ax.scatter(x, PAV, marker='o', color='blue', label='Predicted Ablation Volume')
    ax.scatter(x, MEV, marker='o', color='red', label='Minimum Enclosing Ellipsoid')
    plt.legend(loc='upper left')
    plt.ylabel('Volume (mL)')

    for i in np.arange(0, len(x)):
        x1, x2 = x[i], x[i]
        y1, y2 = MIV[i], MEV[i]
        plt.plot([x1, x2], [y1, y2], 'k-')

    # plt.ylim([-1, 150])
    # labels = np.round(np.asarray(df['PAV']))
    # plt.xticks(x, labels, rotation=45, fontsize=24, color='white')
    timestr = time.strftime("%H%M%S-%Y%m%d")
    folder_path = r"C:\develop\segmentation-eval\figures\MIV_MEV_ellipsoids"
    fig_path = os.path.join(folder_path, timestr)
    gh.save(fig_path, width=12, height=12, ext=["png"],
            close=True,
            tight=True, dpi=600)

    # df.columns = ['Predicted Ablation', 'Effective Ablation', 'Maximum Inscribed Ellipsoid', 'Minimum Enclosing Ellipsoid']
    #b = sns.boxplot(data=df)
    #plt.ylabel('Volume (mL)')
    # b.set_xlabel(fontsize=20)
    # plt.ylim([])
    # plt.grid()
    # plt.show()
    #timestr = time.strftime("%H%M%S-%Y%m%d")
    #gh.save(r'C:\develop\segmentation-eval\figures\boxplots_ellipsoids', width=12, height=12, ext=["png"],close=True,tight=True, dpi=300)


def write_descriptive_stats(df_radiomics_PAV_EAV, device_name):
    """

    :param df_radiomics_PAV_EAV:
    :return: Excel file with descriptive statistics
    """
    df = pd.DataFrame()
    df['PAV'] = df_radiomics_PAV_EAV['Predicted_Ablation_Volume']
    df['EAV'] = df_radiomics_PAV_EAV['Ablation Volume [ml]']
    df['Tumor_Volume'] = df_radiomics_PAV_EAV['Tumour Volume [ml]']
    df['Energy'] = df_radiomics_PAV_EAV['Energy [kj]']

    df['Inner_Ellipsoid_Volume'] = df_radiomics_PAV_EAV['Inner Ellipsoid Volume']
    df['Outer_Ellipsoid_Volume'] = df_radiomics_PAV_EAV['Outer Ellipsoid Volume']
    # # drop outer volumes larger than 150 because they are probably erroneous
    df.loc[df['Outer_Ellipsoid_Volume'] > 150, 'Outer_Ellipsoid_Volume'] = np.nan
    df['MEV-MIV'] = df['Outer_Ellipsoid_Volume'] - df['Inner_Ellipsoid_Volume']
    # # drop the rows where MIV > MEV
    # # since the minimum inscribed ellipsoid (MIV) should always be smaller than the maximum enclosing ellipsoid (MEV)
    # df.loc[df['MEV-MIV'] < 0, 'Inner_Ellpsoid_Volume'] = np.nan
    df.loc[df['MEV-MIV'] < 0, 'MEV-MIV'] = np.nan
    df['Elongation'] = df_radiomics_PAV_EAV['elongation_ablation']
    df['Sphericity'] = df_radiomics_PAV_EAV['sphericity_ablation']
    df['major_axis_ablation'] = df_radiomics_PAV_EAV['major_axis_length_ablation']
    df['major_axis_tumor'] = df_radiomics_PAV_EAV['major_axis_length_tumor']
    df['safety_margin_distribution_0'] = df_radiomics_PAV_EAV['safety_margin_distribution_0']
    df['safety_margin_distribution_5'] = df_radiomics_PAV_EAV['safety_margin_distribution_5']
    df['safety_margin_distribution_10'] = df_radiomics_PAV_EAV['safety_margin_distribution_10']
    df_stats = df.describe()

    iqr = df_stats.loc['75%', :] - df_stats.loc['25%', :]
    iqr.name = 'IQR'
    medd = df.median().to_frame().T
    medd.rename({0: 'median'})
    # median.index.name = 'median'
    df_stats = df_stats.append(medd, ignore_index=False)
    df_stats.rename({0: 'median'})
    df_stats = df_stats.append(iqr)
    filepath_excel = device_name + '_Radiomics_MAVERRIC_Descriptive_Stats.xlsx'
    writer = pd.ExcelWriter(filepath_excel)
    df_stats.to_excel(writer, sheet_name='radiomics', index=True, float_format='%.2f')
    writer.save()

if __name__ == '__main__':
    df_radiomics = pd.read_excel(r"C:\develop\segmentation-eval\Radiomics_MAVERRIC----003328-20200523_.xlsx")
    # change the name of the device
    df_radiomics.loc[df_radiomics.Device_name == 'Angyodinamics (Acculis)', 'Device_name'] = 'Acculis'
    df_radiomics.loc[df_radiomics.Device_name == 'Covidien (Covidien MWA)', 'Device_name'] = 'Covidien'
    df_radiomics.loc[df_radiomics.Device_name == 'Amica (Probe)', 'Device_name'] = 'Amica'
    # select only those rows included in the PAV vs EAV Energy Manuscript
    df_radiomics_PAV_EAV = df_radiomics[df_radiomics['Inclusion_Energy_PAV_EAV'] == True]

    # df_radiomics_PAV_EAV = df_radiomics_PAV_EAV[df_radiomics_PAV_EAV['Inclusion_Margin_Analysis'] == 1]
    # df_radiomics_PAV_EAV = df_radiomics_PAV_EAV[df_radiomics_PAV_EAV['Proximity_to_surface'] == False]

    # plot ellipsoid approximations
    # connected_mev_miv(df_radiomics_PAV_EAV)

    #%%  SCATTER PLOTS
    # set font
    # font = {'family': 'DejaVu Sans',
    #         'size': 18}
    # matplotlib.rc('font', **font)
    #
    # subplots_pav_eav.plot_subplots(df_radiomics_PAV_EAV)
    # mev_miv_scatter.plot_mev_miv(df_radiomics_PAV_EAV)

    # plot_scatter_pav_eav(df_radiomics_PAV_EAV, ratio_flag=False, linear_regression=False)
    # plot_scatter_pav_eav(df_radiomics_PAV_EAV, ratio_flag=False, linear_regression=True)
    # plot_scatter_pav_eav(df_radiomics_PAV_EAV, ratio_flag=True, linear_regression=True)
    # plot_scatter_group_var_subcapsular(df_radiomics_PAV_EAV, ratio_flag=False)
    # plot_scatter_group_var_chemo(df_radiomics_PAV_EAV, ratio_flag=False)
    # plot_scatter_group_var_vessels(df_radiomics_PAV_EAV)

    #%% Descriptive statistics
    df_acculis = df_radiomics_PAV_EAV[df_radiomics_PAV_EAV['Device_name'] == 'Acculis']
    df_amica = df_radiomics_PAV_EAV[df_radiomics_PAV_EAV['Device_name'] == 'Amica']
    df_covidien = df_radiomics_PAV_EAV[df_radiomics_PAV_EAV['Device_name'] == 'Covidien']
    write_descriptive_stats(df_radiomics_PAV_EAV, device_name='All')
    write_descriptive_stats(df_acculis, device_name='Acculis')
    write_descriptive_stats(df_amica, device_name='Amica')
    write_descriptive_stats(df_covidien, device_name='Covidien')

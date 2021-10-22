import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def importing():
    exp1_data = pd.read_excel(
        '304.xlsx', sheet_name='exp1')
    exp2_data = pd.read_excel(
        '304.xlsx', sheet_name='exp2')
    exp3_data = pd.read_excel(
        '304.xlsx', sheet_name='exp3')
    return exp1_data, exp2_data, exp3_data


def exp1_Ux_I(exp1_data):
    x = (exp1_data['I, A'].tolist())
    A = np.vstack((x, np.ones_like(x))).T
    k, b = np.linalg.lstsq(A, exp1_data['Ux, мВ'].tolist(), rcond=None)[0]

    x1 = np.linspace(min(x), max(x))
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(x1, k * x1 + b, color='black', linewidth='1')
    ax1.scatter(x, exp1_data['Ux, мВ'].tolist(), color='black')
    ax1.set_xlabel('Сила тока, А')
    ax1.set_ylabel('Напряжение датчика Холла, мВ')
    ax1.set_xlim(0, 2)
    ax1.set_ylim(0, 60)
    ax1.tick_params(which='major', direction='in')
    ax1.grid(linewidth=0.5, alpha=0.7)
    # ax2 = ax1.twiny()
    # ax2.set_xlim(0, 2)
    # ax2.tick_params(which='major', direction='in')
    # ax2.plot(range(2), np.arange(0, 2), linewidth=0)
    return k


exp1_data, exp2_data, exp3_data = importing()
k_UI = exp1_Ux_I(exp1_data)
k_BI = 4 * np.pi * 1e-7 * \
    (exp1_data['Unnamed: 1'].tolist()[4]) / \
    (exp1_data['Unnamed: 1'].tolist()[2])
print('k_UI:', k_UI, 'мВ/А', '\nK_BI:', k_BI *
      1000, 'мГн/см^2\nk:', k_BI * 1000 / k_UI,'Тл/В')
# plt.legend()
# plt.show

# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


fig, ax = plt.subplots()
fontsize = 24
x = [8.109, 10.24, 39.15, 55.808, 41.032, 10.568, 8.851, 9.315, 4.543, 20.1, 12.45, 29.41, 17.45, 20.5,
     14.06,
     11.538,
     9.403,
     8.1088,
     10,
     12
     ]
y = [12.209, 14.24, 59.17, 71.808, 53.032, 13.568, 16.851, 12.315, 9.543, 28.429, 18.515, 34.49, 23.045, 27.078,
     24.06,
     20.538,
     18.403,
     16.1088,
     14.9669,
     18
     ]

chemo_cycles = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0 , 0, 0 , 12 , 8, 7,  12, 6]


df = pd.DataFrame(data=dict(x=x, y=y, a2=chemo_cycles))
bins = np.arange(start=0, stop=12, step=4)
print(np.digitize(df.a2, bins, right=True))
grouped = df.groupby(np.digitize(df.a2, bins, right=True))
sizes = [150 * (i + 1.) for i in range(4)]
labels = ['0', '1-4', '5-8', '9-12']

# bins = np.arange(start=df.a2.min(), stop=df.a2.max(), step=3)
# bins = np.linspace(df.a2.min(), df.a2.max(), M)

for i, (name, group) in enumerate(grouped):
    print(name)
    plt.scatter(group.x, group.y, s=sizes[i], alpha=0.5, label=labels[name])


plt.legend(labelspacing=1.5, borderpad=1.5, title='Tumor Volumes', handletextpad=3.5)
# plt.show()
#
#
# sc = ax.scatter(x, y, color='steelblue', marker='o',
#                 alpha=0.7, s=size)
# legend_1 = ax.legend(*sc.legend_elements("sizes", num=5, func=lambda x: x / 100 - 1, color='steelblue'),
#                      title='Chemo Cycles', labelspacing=1.5, borderpad=1.5, handletextpad=3.5,
#                      fontsize=fontsize, loc='upper right')
# legend_1.get_title().set_fontsize(str(fontsize))
# ax.add_artist(legend_1)
plt.ylabel('Effective Ablation Volume [ml]', fontsize=fontsize)
plt.xlabel('Predicted Ablation Volume Brochure [ml]', fontsize=fontsize)
plt.ylim([0, 100])
plt.xlim([0, 100])
plt.show()

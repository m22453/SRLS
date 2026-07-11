'''
Date: 2025-06-04 13:51:50
LastEditors: Ruina Bai
LastEditTime: 2025-06-05 06:49:20
FilePath: /code/2025-B-ViewCentric/tsne.py
'''
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler


seed = 2025
root_path = '/remote-home/TCCI17/code/2025-B-ViewCentric/results/CUB_msrt_0.0_train_mode_sf_beta_0.001_seed4-v/visualize/'
# paths = ['embeddings_ours_data_0.npy','embeddings_ours_0.npy', 'embeddings_ours_199.npy', 'embeddings_divide_199.npy']
paths = ['embeddings_view0_raw.npy', 'embeddings_view0_99.npy', 'embeddings_view0_199.npy', 
         'embeddings_view1_raw.npy','embeddings_view1_99.npy','embeddings_view1_199.npy']
# titles = ['raw data structure', 'initialized data structure', 'data structure via RWCL', 'data structure via DIVIDE']
method = 'ours' # choices = ['ours', 'divide']
titles = ['raw 1st view structure ',f'{method} 1st view structure', f'final {method} 1st view structure',
          'raw 2nd view structure ',f'{method} 2nd view structure', f'final {method} 2nd view structure']
# titles = ['Fused view of BBC', 'Original view 1 of BBC','Original view 2 of BBC','Trained view 1 of BBC','Trained view 2 of BBC']
target = np.load(root_path + 'embeddings_label.npy')
print(set(target))
print(target.shape)


def plot_v(data, title_str, target, mark):

    plt.figure(figsize=(10, 8))
    palette = plt.cm.get_cmap("tab10", len(np.unique(target)))  # 配色

    # View1
    scaler = StandardScaler()
    data_standardized_1 = scaler.fit_transform(data)
    X_tsne_1 = TSNE(n_components=2, random_state=seed).fit_transform(data_standardized_1)


    # 保存每类的中心点位置
    centers_1 = []

    # View1 样本点绘制
    for label in np.unique(target):
        idx = (target == label)
        points = X_tsne_1[idx]
        color = palette(label)
        plt.scatter(points[:, 0], points[:, 1],
                    marker=mark, label=f'View1 - Class {label+1}', alpha=0.7, color=color)
        centers_1.append(np.mean(points, axis=0))


    # 🔽 中心点最后绘制，确保不被遮挡
    for center in centers_1:
        plt.scatter(center[0], center[1], marker='x', color='black', s=70, linewidths=2.5, zorder=10)

    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=9)
    plt.tight_layout()
    plt.savefig(root_path + title_str + ".png", format='png', dpi=150)
    plt.close()



def plot_s(datas, title_str, target, marks=["o", "s"]):

    plt.figure(figsize=(10, 8))
    palette = plt.cm.get_cmap("tab10", len(np.unique(target)))  # 配色

    # View1
    scaler = StandardScaler()
    data_standardized_1 = scaler.fit_transform(datas[0])
    X_tsne_1 = TSNE(n_components=2, random_state=seed).fit_transform(data_standardized_1)


    # View2
    scaler = StandardScaler()
    data_standardized_2 = scaler.fit_transform(datas[1])
    X_tsne_2 = TSNE(n_components=2, random_state=seed).fit_transform(data_standardized_2)

    # 保存每类的中心点位置
    centers_1 = []
    centers_2 = []

    # View1 样本点绘制
    for label in np.unique(target):
        idx = (target == label)
        points = X_tsne_1[idx]
        color = palette(label)
        plt.scatter(points[:, 0], points[:, 1],
                    marker=marks[0], label=f'View1 - Class {label+1}', alpha=0.7, color=color)
        centers_1.append(np.mean(points, axis=0))

    # View2 样本点绘制
    for label in np.unique(target):
        idx = (target == label)
        points = X_tsne_2[idx]
        color = palette(label)
        plt.scatter(points[:, 0], points[:, 1],
                    marker=marks[1], label=f'View2 - Class {label+1}', alpha=0.7, color=color)
        centers_2.append(np.mean(points, axis=0))

    # 🔽 中心点最后绘制，确保不被遮挡
    for center in centers_1 + centers_2:
        plt.scatter(center[0], center[1], marker='x', color='black', s=70, linewidths=2.5, zorder=10)

    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=9)
    plt.tight_layout()
    plt.savefig(root_path + title_str + ".png", format='png', dpi=150)
    plt.close()



def stack():
    data1 = np.load(root_path + 'embeddings_view0_raw.npy')
    data2 = np.load(root_path + 'embeddings_view1_raw.npy')
    # raw_data = np.concatenate((data1, data2), axis=0)

    # plot(raw_data, 'raw_views_structure', target)
    plot_s([data1, data2], 'raw_views_structure1', target)

    data1 = np.load(root_path + 'embeddings_view0_99.npy')
    data2 = np.load(root_path + 'embeddings_view1_99.npy')
    # data_99 = np.concatenate((data1, data2), axis=0)
    # targetd = np.concatenate((target, target), axis=0)
    # plot(data_99, '99th_views_structure', targetd)

    plot_s([data1, data2], f'{method}_views_structure1', target)



    data1 = np.load(root_path + 'embeddings_view0_199.npy')
    data2 = np.load(root_path + 'embeddings_view1_199.npy')
    # data_199 = np.concatenate((data1, data2), axis=0)

    # plot(data_199, 'final_views_structure', targetd)
    plot_s([data1, data2], f'final_{method}_views_structure1', target)
def cat():
    
    
    data1 = np.load(root_path + 'embeddings_view0_raw.npy')
    data2 = np.load(root_path + 'embeddings_view1_raw.npy')
    raw_data = np.concatenate((data1, data2), axis=1)

    plot_v(raw_data, 'raw_instance_structure', target, mark='p')

    data1 = np.load(root_path + 'embeddings_view0_99.npy')
    data2 = np.load(root_path + 'embeddings_view1_99.npy')
    data_99 = np.concatenate((data1, data2), axis=1)

    plot_v(data_99, '99th_instance_structure', target, mark='p')

    data1 = np.load(root_path + 'embeddings_view0_199.npy')
    data2 = np.load(root_path + 'embeddings_view1_199.npy')
    data_199 = np.concatenate((data1, data2), axis=1)

    plot_v(data_199, 'final_instance_structure', target, mark='p')

    data_99 = np.load(root_path + 'embeddings_99.npy')
    data_199 = np.load(root_path + 'embeddings_199.npy')
    
    plot_v(data_99, '99th_fused_structure', target, mark='p')
    plot_v(data_199, 'final_fused_structure', target, mark='p')

        
        
def single():
    for i in range(len(paths)):
        data = np.load(root_path+paths[i])
        print( data.shape)
        if i < 3:
            plot_v(data, titles[i], target, mark='o')
        else:
            plot_v(data, titles[i], target, mark='s')

        

# single()
cat()
stack()
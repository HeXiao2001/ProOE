import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import matplotlib
import numpy as np
from matplotlib.colors import Normalize, LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import zscore
import matplotlib.gridspec as gridspec
from tqdm import tqdm

# 设置matplotlib的字体为新罗马
plt.rcParams["font.family"] = "Times New Roman"

def plot_all(dataset, Flows_suffix, K):
    def plot_single_gdf(gdf, ax):
        community_columns = [col for col in gdf.columns if col.startswith('c_')]
        colors = plt.cm.tab20c(np.linspace(0, 1, len(community_columns)))
        community_color_map = {community: colors[i] for i, community in enumerate(community_columns)}

        def mix_multiple_colors(colors):
            colors = np.array([color for color in colors if color[3] > 0])
            if len(colors) == 0:
                return (0, 0, 0, 0)
            total_alpha_weight = np.sum(colors[:, 3])
            r_mix = np.sum(colors[:, 0] * colors[:, 3]) / total_alpha_weight
            g_mix = np.sum(colors[:, 1] * colors[:, 3]) / total_alpha_weight
            b_mix = np.sum(colors[:, 2] * colors[:, 3]) / total_alpha_weight
            a_mix = 1
            return (r_mix, g_mix, b_mix, a_mix)

        def calculate_mixed_color(row):
            colors = [(community_color_map[community][0], community_color_map[community][1], community_color_map[community][2], row[community]) for community in community_columns]
            return mix_multiple_colors(colors)

        gdf['mixed_color'] = gdf.apply(calculate_mixed_color, axis=1)
        gdf.plot(ax=ax, color=gdf['mixed_color'], edgecolor='black', linewidth=0.01, markersize=100)
        ax.set_axis_off()
        ax.set_frame_on(True)  # 显示边框
        patches = []
        for col, color in community_color_map.items():
            col = col.replace('c_', 'Comm ')
            patches.append(mpatches.Patch(color=color, label=col))
        ax.legend(handles=patches, loc='best', fontsize=16)
        # title
        ax.set_title('result - Spatial Fuzzy Communities', fontsize=20)

    def plot_trip_volume_matrix(dataset, Flows_suffix, K, ax):
        df = pd.read_csv('./data/input/{}/Flows{}.csv'.format(dataset, Flows_suffix, str(K)), dtype={'O_id': str, 'D_id': str, 'flow': float})
        df.columns = ['O_id', 'D_id', 'data']
        gdf = gpd.read_file('./data/output/{}/U{}_{}.geojson'.format(dataset, Flows_suffix, str(K)))
        Ts = pd.read_csv('./data/output/{}/I{}_{}.csv'.format(dataset, Flows_suffix, str(K)), dtype={'I': float, 'c': str})
        community_columns = [col for col in gdf.columns if col.startswith('c_')]
        gdf['max_community'] = gdf[community_columns].idxmax(axis=1).apply(lambda x: x)
        gdf['max_community_value'] = gdf[community_columns].max(axis=1)
        gdf = gdf[gdf['max_community_value'] > 0]
        colors = plt.cm.tab20c(np.linspace(0, 1, len(community_columns)))
        community_color_map = {community: colors[i] for i, community in enumerate(community_columns)}
        community_dict = {}
        for community in community_columns:
            tempdata = gdf[gdf['max_community'] == community][['LocationID', community]]
            tempdata = tempdata.sort_values(by=community, ascending=False)
            community_dict[community] = tempdata['LocationID'].values
        units_list = []
        for community in community_columns:
            units_list.extend(community_dict[community])
        x = units_list
        y = units_list
        df_indexed = df.set_index(['O_id', 'D_id'])
        n = len(units_list)
        z = np.zeros((n, n))
        data_dict = df_indexed['data'].to_dict()
        for i in tqdm(range(n), desc="Processing rows"):
            for j in range(i + 1, n):
                key = (x[i], y[j])
                if key in data_dict:
                    value = data_dict[key]
                    z[i, j] = value
                    z[j, i] = value
        z[z == 0] = 1e-10
        c = ax.pcolormesh(z, cmap='viridis', edgecolors='none', linewidth=0, norm=matplotlib.colors.LogNorm(vmin=1, vmax=z.max()))
        data_y_range = ax.get_ylim()
        display_y_range1 = ax.get_position().height
        data_y_range = ax.get_ylim()
        display_y_range2 = ax.get_position().height
        suofang = display_y_range2 / display_y_range1
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('bottom', size='3%', pad=0.7)
        cbar = plt.colorbar(c, cax=cax, orientation='horizontal')
        cax.yaxis.set_ticks_position('left')
        cax.yaxis.set_label_position('left')
        cbar.set_label('Trip Volume', rotation=0, labelpad=0)
        cbar.set_ticks([c.get_clim()[0], c.get_clim()[1]])
        cbar.set_ticklabels(['Min', 'Max'])
        coum_borders = [len(community_dict[community_columns[0]])]
        for community in community_columns:
            if community != community_columns[0]:
                coum_borders.append(max(coum_borders) + len(community_dict[community]))
        for coum_border in coum_borders:
            ax.axhline(coum_border, color='white', linestyle='--', linewidth=0.5)
            ax.axvline(coum_border, color='white', linestyle='--', linewidth=0.5)
        ticks = []
        for i in range(len(community_columns)):
            if i > 0:
                ticks.append((coum_borders[i - 1] + coum_borders[i]) / 2)
            else:
                ticks.append(coum_borders[i] / 2)
        ticks = np.array(ticks)
        community_names = [comm.replace('c_', 'Comm ') for comm in community_columns]
        ax.set_xticks(ticks, minor=False)
        ax.set_yticks(ticks, minor=False)
        ax.set_xticklabels(community_names, rotation=90, fontsize=10)
        ax.set_yticklabels(community_names, fontsize=10)
        ax.tick_params(axis='both', which='both', length=0)
        for spine in ax.spines.values():
            spine.set_linewidth(0)
        for i in range(len(coum_borders)):
            if i == 0:
                ax.plot([0, coum_borders[i]], [coum_borders[i], coum_borders[i]], color='red', linewidth=1, zorder=12)
                ax.plot([0, 0], [coum_borders[i], 0], color='red', linewidth=1, zorder=12)
                ax.plot([0, coum_borders[i]], [0, 0], color='red', linewidth=1, zorder=12)
                ax.plot([coum_borders[i], coum_borders[i]], [0, coum_borders[i]], color='red', linewidth=1, zorder=12)
            else:
                ax.plot([coum_borders[i - 1], coum_borders[i]], [coum_borders[i], coum_borders[i]], color='red', linewidth=1)
                ax.plot([coum_borders[i - 1], coum_borders[i]], [coum_borders[i - 1], coum_borders[i - 1]], color='red', linewidth=1)
                ax.plot([coum_borders[i - 1], coum_borders[i - 1]], [coum_borders[i - 1], coum_borders[i]], color='red', linewidth=1)
                ax.plot([coum_borders[i], coum_borders[i]], [coum_borders[i - 1], coum_borders[i]], color='red', linewidth=1)
        ax.set_frame_on(True)  # 显示边框
        # title
        ax.set_title('result - Trip Volume Matrix', fontsize=20)

    def plot_confidence_index(dataset, Flows_suffix, K, ax):
        intensity = pd.read_csv(f'./data/output/{dataset}/I{Flows_suffix}_{str(K)}.csv', dtype={'I': float, 'c': str})
        intensity.columns = ['Intensity', 'c_s']
        intensity['zscore'] = zscore(intensity['Intensity'])
        intensity['confidence'] = 1 / (1 + np.exp(-intensity['zscore']))
        communitys = intensity['c_s'].unique()
        colors = plt.cm.tab20c(np.linspace(0, 1, len(communitys)))
        community_color_map = {community: colors[i] for i, community in enumerate(communitys)}
        intensity['color'] = intensity['c_s'].map(community_color_map)
        intensity = intensity.set_index('c_s')
        intensity['confidence'].plot.bar(x='c_s', y='confidence', ax=ax, color=intensity['color'], legend=False)
        ax.set_xticklabels([f'Comm {i}' for i in range(1, K + 1)], rotation=45, fontsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.set_ylabel('Confidence Index', fontsize=20)
        ax.set_frame_on(True)  # 显示边框
        # title
        ax.set_title('result - Confidence Index', fontsize=20)

    def plot_entropy_map(dataset, Flows_suffix, K, ax):
        gdf = gpd.read_file('./data/output/{}/U{}_{}.geojson'.format(dataset, Flows_suffix, str(K)))
        gdf = gdf.to_crs(epsg=3857)
        community_columns = [col for col in gdf.columns if col.startswith('c_')]
        gdf['max_value'] = gdf[community_columns].max(axis=1)
        gdf = gdf[gdf['max_value'] >= 0.01]
        gdf['entropy'] = gdf.apply(lambda x: np.sum([x[col]*np.log(x[col]) for col in community_columns if x[col] > 0]), axis=1)
        colors = ['#084484','#1472b2','#41a6ca','#79c9c7','#b1e1b8','#d6efd0','#f4faef']
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=10)
        cmap = cmap.reversed()
        norm = Normalize(vmin=gdf['entropy'].min(), vmax=gdf['entropy'].max())
        gdf['color'] = gdf['entropy'].apply(lambda x: cmap(norm(x)))
        gdf.plot(ax=ax, color=gdf['color'], edgecolor='black', linewidth=0.01)
        ax.set_axis_off()
        cax = inset_axes(ax, width="5%", height="50%", loc='upper left', borderpad=2)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cax)
        cbar.set_ticks([gdf['entropy'].min(), gdf['entropy'].max()])
        cbar.set_ticklabels(['Low', 'High'], fontsize=30)
        ax.set_frame_on(True)  # 显示边框
        # title
        ax.set_title('result - Certainty Index', fontsize=20)

    gdf = gpd.read_file('./data/output/{}/U{}_{}.geojson'.format(dataset, Flows_suffix, str(K)))
    
    fig, axs = plt.subplots(2, 2, figsize=(20, 20), dpi=100)
    plot_single_gdf(gdf, axs[0, 0])
    plot_trip_volume_matrix(dataset, Flows_suffix, K, axs[0, 1])
    plot_confidence_index(dataset, Flows_suffix, K, axs[1, 0])
    plot_entropy_map(dataset, Flows_suffix, K, axs[1, 1])
    plt.tight_layout()
    plt.show()
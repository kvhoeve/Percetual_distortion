# kvhoeve
# 31/10/2023
# LISA processing

# add packages:
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd # Spatial data manipulation
from pysal.explore import esda  # Exploratory Spatial analytics
from pysal.lib import weights, cg  # Spatial weights
from scipy.stats import kendalltau # can import levene
from joblib import Parallel, delayed

# LISA pipeline
path = 'your root path'
img_dir = 'image directory path'
os.mkdir(os.path.join(img_dir + 'correlations'))
os.mkdir(os.path.join(path + 'memory'))

# data loading
data = pd.read_csv(os.path.join(path + 'son_clip_data_V2.csv'))
s_data = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.longitude, data.latitude), crs=4326)
s_data['geom_tuple'] = list((zip(s_data.longitude, s_data.latitude)))

# step 1: kendalls tau correlations between the target attributes and landscape attributes
# per target attribute
target_dict = {}
for col in range(3,len(data.columns)-2):
    tau, p_val = kendalltau(data.Scenicness, data.iloc[:,col])
    target_dict[data.columns[col]] = (tau, p_val)

# construct datatframe
target_df = pd.DataFrame.from_dict(target_dict, orient='index', columns=['Kendalls Tau', 'P-value'])
themes_list = ['weather'] * 30 + ['aesthetics'] * 30 + ['land use'] * 30 + ['structures'] * 30 + ['buildings'] * 30 + ['relief'] * 30 + ['water bodies'] * 30 + ['vegetation'] * 30 + ['transportation'] * 30 + ['infrastructure'] * 30
target_df['Theme'] = themes_list
target_df.to_csv(os.path.join(path + 'son_correlation.csv'), index=False)

# create visuals
for i, t in enumerate(['weather', 'aesthetics', 'land use', 'structures', 'buildings', 'relief', 'water bodies', 'vegetation', 'transportation', 'infrastructure']):
    fig, ax = plt.subplots(figsize=(10,6))
    target_df['Kendalls Tau'].iloc[(i*30):((i+1)*30),].plot.bar(ylabel='Kendalls Tau', fontsize=10)
    ax.set_title('Kendalls Tau correlation with Scenicness per attribute for theme: ' + t)
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray')
    plt.tight_layout()
    fig.savefig(os.path.join(img_dir + '\\correlations\\' + t + '.png')) 

# step 2: LISA values voor alle attributen.
# post-hoc testing on the set that has the lowest Delta LISA
# step 2: LISA values voor alle attributen.
# check which number of nieghbours is sufficient
DistTree = cg.KDTree(s_data.geom_tuple, distance_metric="Arc", radius=cg.RADIUS_EARTH_KM)    

def dist_check(df, tree, neighbours):
    '''This function calculates the spatial autocorrelation for Scenicness per number of neighbours'''
    ws = weights.distance.KNN(tree, k=neighbours)
    lisa = esda.Moran_Local(df.Scenicness, ws, transformation='r')
    return [lisa.Is, lisa.p_sim]

# This step takes a while
auto_corr = Parallel(n_jobs=-1)(delayed(dist_check)(s_data, DistTree, i) for i in range(5,50,5))
# unpack the LISA values 
dist_df = pd.DataFrame({'fake':[]})
means_list = []
for i in range(len(auto_corr)):
    lisa_col = 'LISA_' + str((i*5)+5)
    p_col = 'P-val_' + str((i*5)+5)
    dist_df[lisa_col] = auto_corr[i][0]
    dist_df[p_col] = auto_corr[i][1]
    dist_df[lisa_col].loc[dist_df[p_col] > 0.05] = 0 # p-values lower than 0.05 are valid.
    means_list.append(dist_df[lisa_col].mean())

# visualize results
fig, ax = plt.subplots(figsize=(6,3))
plt.bar(x=[5,10,15,20,25,30,35,40,45], height=means_list)
plt.ylim(bottom=0.31, top=0.36)
plt.title('Mean spatial autocorrelation per number of neighbours')
plt.ylabel('Mean LISA')
plt.tight_layout()
plt.savefig(os.path.join(img_dir + '\\mean_spat_corr.png'))
# from this process, we find that a number of 25 neighbours is optimal. 
# This number of neighbours will be used for all following LISA calculations

# this step takes a while
KDTree = cg.KDTree(s_data.geom_tuple, distance_metric="Arc", radius=cg.RADIUS_EARTH_KM)
w = weights.distance.KNN(KDTree, k=25)
# this steps takes hours. Be careful running this part and make sure the right columns are selected
def cal_LISA(df, weight_matrix):
    lisa = esda.Moran_Local(df, weight_matrix, transformation='r')
    return [lisa.Is, lisa.p_sim]

results = Parallel(n_jobs=-1, verbose=2, temp_folder=os.path.join(path + 'memory'))(delayed(cal_LISA)(data.iloc[:,i], w) for i in range(3,303,1))

LISA_df = pd.DataFrame({'fake': []})
# save all values in a dataframe
for i in range(len(results)):
    lisa_col = 'LISA_' + data.columns[i+3]
    p_col = 'P-val_' + data.columns[i+3]
    LISA_df[lisa_col] = results[i][0]
    LISA_df[p_col] = results[i][1]

LISA_df.drop(columns=['fake'], inplace=True)

LISA_df.to_csv(os.path.join(path + 'LISA_clip_300.csv'), index=False)

lisa_copy = LISA_df.copy(deep=False)

lisa_means = {}
for ci in range(0,len(lisa_copy.columns), 2):
    cn = lisa_copy.columns[ci]
    lisa_copy[cn].loc[lisa_copy.iloc[:,ci+1] > 0.05] = 0
    # grabbing the attribute name from the column name by splitting,
    # and saving the mean of the corrected LISA in a dict.
    lisa_means[cn.split('_')[-1]] = lisa_copy[cn].mean()

# create a dataframe for all means, and include the themes
means_df = pd.DataFrame.from_dict(lisa_means, orient='index', columns=['mean'])
means_df['Theme'] = themes_list

# sort them from high to low
sorted_df = means_df.sort_values(by='mean', axis=0, ascending=False)
# save sorted values
sorted_df.to_csv(os.path.join(path + 'sorted_means.csv'))

highest_100 = sorted_df.iloc[0:100]
middle_100 = sorted_df.iloc[100:200]
lowest_100 = sorted_df.iloc[200:]

highest_100.plot.bar(figsize=(14,4), ylim=(0,0.43), ylabel='Mean LISA', legend=False, fontsize=10, title='LISA means for the 100 highest attributes')
plt.tight_layout()
plt.savefig(os.path.join(img_dir +'highest_100.png'))

middle_100.plot.bar(figsize=(14,4), ylabel='Mean LISA', ylim=(0,0.43), legend=False, fontsize=10, title='LISA means for the 100 middle attributes')
plt.tight_layout()
plt.savefig(os.path.join(img_dir +'middle_100.png'))

lowest_100.plot.bar(figsize=(14,4), ylabel='Mean LISA', ylim=(0,0.43), legend=False, fontsize=10, title='LISA means for the 100 lowest attributes')
plt.tight_layout()
plt.savefig(os.path.join(img_dir +'lowest_100.png'))


# create a violin plot to show the spread of the means
fig, ax = plt.subplots(figsize=(12,6))
sns.violinplot(x=sorted_df['Theme'], y=sorted_df['mean'], inner='quartile').set(title="Mean LISA spread per theme", ylabel="mean LISA")
plt.axhline(y=sorted_df['mean'].iloc[0:100].min(), linestyle='--', color='blue')
plt.axhline(y=sorted_df['mean'].iloc[200:300].max(), linestyle='--', color='blue')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(img_dir + 'violin_plot_LISA_spread.png'))

# kvhoeve
# 23/11/2023
# linear probing

# load packages
import os
import sklearn.svm as svm
from sklearn.metrics import mean_squared_error, accuracy_score, ConfusionMatrixDisplay
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import SGDClassifier
import collections
import ast

# define paths
root_path = 'your root'
son_path = 'path to SoN data'

# Data preparation
# read all data
son_clip = pd.read_csv(os.path.join(root_path + 'son_clip_data_V2.csv'))
votes = pd.read_csv(os.path.join(son_path + 'votes.tsv'), sep='\t')
corine = gpd.read_file(os.path.join(son_path + 'son_pts_aux_info.geojson'))

# votes contains infromation for images not included in the dataset
no_exist_list = os.listdir(os.path.join(son_path + 'images\\no_exist'))
# remove 
for si, sn in enumerate(no_exist_list):
    no_exist_list[si] = int(sn.split('.')[0])

# removes those entries from vtoes for which no image exists
votes = votes.query("ID not in @no_exist_list")

# include the ID in son_clip
son_clip['ID'] = votes['ID']

# aggregate all data
all_data = son_clip.merge(corine, on='ID', how='left')
all_data.drop(columns=['index', 'ID', 'Lat', 'Lon', 'Average', 'Variance', 'Votes',
       'Geograph U', 'split', 'folder_num', 'bin', 'pop_values',
       'geometry'], axis=1, inplace=True)
# dropping rows, where no LC values were present for
all_data.dropna(axis=0, how='any', inplace=True)

all_data.to_csv(os.path.join(root_path + 'lin_pro_data.csv'), index=False)

sorted_means = pd.read_csv(os.path.join(root_path + 'sorted_means.csv'), index_col=0)
highest_names = sorted_means.index[0:100].to_list()
lowest_names = sorted_means.index[200:].to_list()

# put all data in required structures
all_scen_labels = all_data.Scenicness.to_numpy()
all_lc_labels = all_data.lc.to_numpy()
img_features = all_data.img_feature.apply(lambda row: ast.literal_eval(row)).to_list()
all_features = all_data.iloc[:,3:303].apply(func=lambda row: row.to_list(), axis=1).to_list()
highest_features = all_data[highest_names].apply(lambda row: row.to_list(), axis=1).to_list()
lowest_features = all_data[lowest_names].apply(lambda row: row.to_list(), axis=1).to_list()

# reshaping
img_features = np.array(img_features)
all_features = np.array(all_features)
highest_features = np.array(highest_features)
lowest_features = np.array(lowest_features)

# split data into 80-20 train-test split
# image features
train_img = img_features[:166600,]
test_img = img_features[166600:,]
# all attributes
train_features = all_features[:166600,]
test_features = all_features[166600:,]
# 100 most spatially autocorrelated
train_highest = highest_features[:166600,]
test_highest = highest_features[166600:,]
# 100 least spatially autcorrelated
train_lowest = lowest_features[:166600,]
test_lowest = lowest_features[166600:,]
# labels for regressor
train_scen_labels = all_scen_labels[:166600,]
test_scen_labels = all_scen_labels[166600:,]
# labels for classifier
train_lc_labels = all_lc_labels[:166600,]
test_lc_labels = all_lc_labels[166600:,]

# === Linear Probe ==
# scenicness
baseline_scen = BaggingRegressor(svm.LinearSVR(random_state=213, epsilon=0.01, C=0.1, max_iter=1000, dual='auto'), n_estimators=30, max_samples = 1.0/30, bootstrap=True, max_features=512, random_state=5087, n_jobs=-3)
baseline_scen.fit(train_img, train_scen_labels)
base_pred_scen = baseline_scen.predict(test_img)
base_mse = mean_squared_error(test_scen_labels, base_pred_scen) # 0.7564584253341149

# with classes
regressor = BaggingRegressor(svm.LinearSVR(random_state=213, epsilon=0.01, C=0.1, max_iter=1000, dual='auto'), n_estimators=30, max_samples = 1.0/30, bootstrap=True, max_features=300, random_state=5087, n_jobs=-3)
regressor.fit(train_features, train_scen_labels)
scen_pred = regressor.predict(test_features)
mse = mean_squared_error(test_scen_labels, scen_pred) # 0.9075453889340301

highest_100 = BaggingRegressor(svm.LinearSVR(random_state=213, epsilon=0.01, C=0.1, max_iter=1000, dual='auto'), n_estimators=10, max_samples = 1.0/10, bootstrap=True, max_features=100, random_state=5087, n_jobs=-3)
highest_100.fit(train_highest, train_scen_labels)
high_pred = highest_100.predict(test_highest)
mse_h = mean_squared_error(test_scen_labels, high_pred) # 0.8362122036828518

lowest_100 = BaggingRegressor(svm.LinearSVR(random_state=213, epsilon=0.01, C=0.1, max_iter=1000, dual='auto'), n_estimators=10, max_samples = 1.0/10, bootstrap=True, max_features=100, random_state=5087, n_jobs=-3)
lowest_100.fit(train_lowest, train_scen_labels)
low_pred = lowest_100.predict(test_lowest)
mse_l = mean_squared_error(test_scen_labels, low_pred) # 0.940803413680694

# CORINE land classes

# create list of class names
# solution of making a sorted list of values by NPE at https://stackoverflow.com/questions/9001509/how-do-i-sort-a-dictionary-by-key
# accessed November 2023
lc_dict = {212: 'Permanently irrigated land', 241: 'Annual crops associated with permanent crops', 121: 'Industrial or commercial units', 323: 'Sclerophylous vegetation', 421: 'Salt marshes',
            322: 'Moors and heathland', 313: 'Mixed forest', 321: 'Natural grassland', 211: 'Non-irrigated arable land', 312: 'Coniferous forest', 122: 'Road and rail networks and associated land',
            331: 'Beaches, dunes, and sand plains', 334: 'Burnt areas', 511: 'Water courses', 521: 'Coastal lagoons', 224: 'Agro-forestry areas', 142: 'Sport and leisure facilities', 132: 'Dump sites',
            523: 'Sea and ocean', 112: 'Discontinuous urban fabric', 422: 'Salines', 131: 'Mineral extraction sites', 412: 'Peatbogs', 333: 'Sparsely vegetated areas', 123: 'Port areas', 332: 'Bare rock',
            124: 'Airports', 512: 'Water bodies', 243: 'Agriculture and natural vegeation', 223: 'Olive groves', 141: 'Green urban areas', 133: 'Construction sites',
            522: 'Estuaries', 335: 'Glaciers and perpetual snow'}
od = collections.OrderedDict(sorted(lc_dict.items()))
lc_list = list(od.values())

# baseline
baseline_lc = SGDClassifier(loss="log_loss", random_state=5087, alpha=0.1, early_stopping=True, class_weight='balanced', penalty="l2", n_iter_no_change=15, max_iter=1000)
baseline_lc.fit(train_img, train_lc_labels)
baseline_train = baseline_lc.predict(train_img)
print(str(accuracy_score(train_lc_labels, baseline_train))) # 0.21630852340936374
base_pred_lc = baseline_lc.predict(test_img)
base_acc = accuracy_score(test_lc_labels, base_pred_lc) # 0.13939030244839173

# all classes
clf = SGDClassifier(loss="log_loss", random_state=5087, alpha=0.001, early_stopping=True, class_weight='balanced', penalty="l2", n_iter_no_change=10, max_iter=1000)
clf.fit(train_features, train_lc_labels)
clf_train = clf.predict(train_features)
print(str(accuracy_score(train_lc_labels, clf_train))) # 0.20231692677070828
clf_pred = clf.predict(test_features)
clf_acc = accuracy_score(test_lc_labels, clf_pred) # 0.17822851656265

clf_high = SGDClassifier(loss="log_loss", random_state=5087, alpha=0.001, early_stopping=True, class_weight='balanced', penalty="l2", n_iter_no_change=10, max_iter=1000)
clf_high.fit(train_highest, train_lc_labels)
clfh_train = clf_high.predict(train_highest)
print(accuracy_score(train_lc_labels, clfh_train)) # 0.2615606242496999
clf_high_pred = clf_high.predict(test_highest)
clf_high_acc = accuracy_score(test_lc_labels, clf_high_pred) # 0.18967834853576573

clf_low = SGDClassifier(loss="log_loss", random_state=5087, alpha=0.001, early_stopping=True, class_weight='balanced', penalty="l2", n_iter_no_change=10, max_iter=1000)
clf_low.fit(train_lowest, train_lc_labels)
clfl_train = clf_low.predict(train_lowest)
print(accuracy_score(train_lc_labels, clfl_train)) # 0.17821128451380552
clf_low_pred = clf_low.predict(test_lowest)
clf_low_acc = accuracy_score(test_lc_labels, clf_low_pred) # 0.1185069611137782

# This code still needs some work, but the bare bones can be used tocreate confusion matrices. 
# The for loop aspect does not work yet.

# title_list = ['CORINE Baseline', 'all 300 classes', 'the 100 most spatially autocorrelated classes', 'the 100 least spatially autocorrelated classes']
# img_n = ['baseline', '300_classes', 'highest_100', 'low_100']
# clf_list = [base_pred_lc, clf_pred, clf_high_pred, clf_low_pred]
# for p in clf_list:
#     cimg = ConfusionMatrixDisplay.from_predictions(
#     test_lc_labels, 
#     p, 
#     display_labels=lc_list, 
#     cmap=plt.cm.Blues,
#     colorbar=False,
#     xticks_rotation='vertical'
#     )
#     cimg.figure_.set_figheight(16)
#     cimg.figure_.set_figwidth(16)
#     cimg.ax_.set_title('Confusion matrix for ' + title_list[clf_list.index(p)])
#     plt.tight_layout()




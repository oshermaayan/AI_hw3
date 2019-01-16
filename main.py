import hw3_utils
import our_utils
import random
#from classifier import *
import pickle
from sklearn.decomposition import PCA
import sklearn.preprocessing

data, labels, test = hw3_utils.load_data()

#Shuffle
combined = list(zip(data, labels))
random.shuffle(combined)
data[:], labels[:] = zip(*combined)

#Scale
scaler = sklearn.preprocessing.StandardScaler().fit(data)
data = scaler.transform(data)
test = scaler.transform(test)

# Use PCA
pca=PCA(n_components=5)
pca.fit(data)
data = pca.fit_transform(data)
test = pca.fit_transform(test)

new_dataset = (data, labels, test)

with open("Shuffled_scaled_PCA_data.data","wb") as f:
    pickle.dump(new_dataset, f)

'''
pca=PCA(n_components=5)
pca.fit(data)
data_after_pca = pca.fit_transform(data)
test_after_PCA = pca.fit_transform(test)

new_dataset_after_PCA = (data_after_pca, labels, test_after_PCA)


with open("Shuffled_scaled_PCA_data.data","wb") as f:
    pickle.dump(new_dataset_after_PCA, f)

print("")
'''
'''
combined = list(zip(data, labels))
random.shuffle(combined)

data[:], labels[:] = zip(*combined)
new_dataset = (data, labels, set)

with open("Shuffled_data.data","wb") as f:
    pickle.dump(new_dataset, f)


scaler = sklearn.preprocessing.StandardScaler().fit(data)
scaled_data = scaler.transform(data)

new_dataset_scaled = (scaled_data, labels, test)

with open("Shuffled_scaled_data.data","wb") as f:
    pickle.dump(new_dataset_scaled, f)

'''


'''
#Shuffle original data

combined = list(zip(data, labels))
random.shuffle(combined)

data[:], labels[:] = zip(*combined)

new_dataset = (data, labels, test)

our_utils.split_crosscheck_groups(new_dataset,2)
'''
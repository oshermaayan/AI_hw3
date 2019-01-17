import hw3_utils as utils
from sklearn import tree
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
import sklearn.preprocessing
import our_utils
import pickle
import numpy as np
from sklearn.svm import SVC
import graphviz


class knn_classifier(utils.abstract_classifier):
    def __init__(self, data, labels, k):
        '''
        :param k: k parameter for KNN
        '''
        self.k = k
        self.dataset = data
        self.labels = labels

    def classify(self, feature_vector):
        distances = []

        for sample, tag in zip(self.dataset, self.labels):
            distance = our_utils.euclidean_distance(feature_vector, sample)
            distances.append((tag, distance))

        # Sort list by distances
        sorted_distances = sorted(distances, key=lambda dist: dist[1])

        negative_count=0
        positive_count =0
        for i in range(self.k):
            dataset_tag = sorted_distances[i][0]
            if dataset_tag==True:
                positive_count += 1
            else:
                negative_count += 1

        if positive_count>=negative_count:
            return True
        else:
            return False

class TreeClassifier():
    def __init__(self, DecisionTree):
        self.dtc = DecisionTree

    def classify(self, feature_vector):
        return self.dtc.predict([feature_vector])

class PerceptronClassifier():
    def __init__(self, percpetron_classifier):
        self.clf = percpetron_classifier

    def classify(self, feature_vector):
        return self.clf.predict([feature_vector])[0]

class SvmClassifier():
    def __init__(self, svm_classifier):
        self.clf = svm_classifier

    def classify(self, feature_vector):
        return self.clf.predict([feature_vector])[0]

class knn_factory(utils.abstract_classifier_factory):
    def __init__(self, k):
        self.k = k

    def train(self, data, labels):
        return knn_classifier(data, labels, self.k)


class DecisionTree_factory(utils.abstract_classifier_factory):
    def train(self, data, labels):
        dtc = tree.DecisionTreeClassifier()
        dtc = dtc.fit(data, labels)
        #dot_data = tree.export_graphviz(dtc, out_file=None)
        return TreeClassifier(dtc)

class Perceptron_factory(utils.abstract_classifier_factory):
    def train(self, data, labels):
        clf = Perceptron()
        clf.fit(data, labels)
        return PerceptronClassifier(clf)


class Svm_factory(utils.abstract_classifier_factory):
    def train(self, data, labels):
        clf = SVC(gamma='auto')
        clf = clf.fit(data, labels)
        return SvmClassifier(clf)

def run_section6():
    csv_path = "experiments6.csv"
    k_list = [1, 3, 5, 7, 13]

    with open(csv_path, "w") as file:
        for k in k_list:
            classifier_factory = knn_factory(k)
            accuracy, error = our_utils.evaluate(classifier_factory, 2)
            output = "{k},{acc},{error}\n".format(k=k, acc=accuracy, error=error)
            file.write(output)

def run_section7():
    csv_path = "experiments12.csv"
    with open(csv_path, "w") as file:
        classifier_factory = Perceptron_factory()
        accuracy, error = our_utils.evaluate(classifier_factory, 2)
        output = "Perceptron,{acc},{error}\n".format(acc=accuracy, error=error)
        file.write(output)

        classifier_factory = DecisionTree_factory()
        accuracy, error = our_utils.evaluate(classifier_factory, 2)
        output = "Tree,{acc},{error}\n".format(acc=accuracy, error=error)
        file.write(output)

def scale_data(data):
    scaled_data = sklearn.preprocessing.scale(data)
    return scaled_data

def normalize_data(data):
    normalized_data = sklearn.preprocessing.normalize(data)
    return normalized_data

def try_tree_improvements():
    tf = DecisionTree_factory()
    accuracy, error = our_utils.evaluate_choose_folds(tf, 2, prefix, scale_data=True)
    print("Accuracy on original set:{acc}".format(acc=accuracy))

    prefix_shuffled = "ecg_fold_"

    accuracy, error = our_utils.evaluate_choose_folds(tf, 2, prefix, scale_data=True)
    print("Accuracy on normalized set:{acc}".format(acc=accuracy))

def run_knn():
    # preprocess_set("ecg_fold_0.data",0, scale_data, "ecg_fold_scaled_")
    # preprocess_set("ecg_fold_1.data",1, scale_data, "ecg_fold_scaled_")

    shuffled_prefix = "ecg_fold_shuffled_"
    scaled_prefix = "ecg_fold_scaled_"

    for k in [1, 3, 5, 7, 13]:
        clf = knn_factory(k)
        accuracy, error = our_utils.evaluate_choose_folds(clf, 2, scaled_prefix)
        print("K={k}: Accuracy on original set:{acc}".format(k=k, acc=accuracy))

        accuracy, error = our_utils.evaluate_choose_folds(clf, 2, shuffled_prefix, scale_data=True)
        print("K={k}: Accuracy on new/transformed set:{acc}".format(k=k, acc=accuracy))

def run_svm():
    shuffled_prefix = "ecg_fold_shuffled_"
    scaled_prefix = "ecg_fold_scaled_"

    clf = Svm_factory()
    accuracy, error = our_utils.evaluate_choose_folds(clf, 2, scaled_prefix)
    print("Accuracy on original set:{acc}".format(acc=accuracy))

    accuracy, error = our_utils.evaluate_choose_folds(clf, 2, "ecg_fold_", use_pca=True, scale_data=True,
                                                      pca_comp_num=3)
    print("Accuracy on new/transformed set:{acc}".format(acc=accuracy))

def preprocess_set(fold_path, fold_num, preproc_func, prefix):
    data, labels = our_utils.load_data_fold(fold_path)
    normalized_fold = preproc_func(data)
    fold_name = prefix+str(fold_num)+".data"
    with open(fold_name, 'wb') as fold_file:
        pickle.dump((normalized_fold, labels), fold_file)


#Main
training_set, labels, test_set = utils.load_data(r'Shuffled_scaled_PCA_data.data')

#Create classifier and train them
svm = Svm_factory()
svm = svm.train(training_set, labels)

tree_classifier = DecisionTree_factory()
tree_classifier = tree_classifier.train(training_set, labels)

knn_7 = knn_factory(7)
knn_7 = knn_7.train(training_set, labels)

knn_9 = knn_factory(9)
knn_9 = knn_9.train(training_set, labels)

knn_11 = knn_factory(11)
knn_11 = knn_11.train(training_set, labels)

#Predictions for test set
predictions = []
for sample in test_set:
    counter = 0
    counter += 1 if svm.classify(sample) else 0
    counter += 1 if tree_classifier.classify(sample) else 0
    counter += 1 if knn_7.classify(sample) else 0
    counter += 1 if knn_9.classify(sample) else 0
    counter += 1 if knn_11.classify(sample) else 0
    if counter > 2:
        predictions.append(True)
    else:
        predictions.append(False)


print(np.where(np.array(predictions)==False)[0].shape)
utils.write_prediction(predictions)


#preprocess_set("ecg_fold_0.data",0, scale_data, "ecg_fold_scaled_")
#preprocess_set("ecg_fold_1.data",1, scale_data, "ecg_fold_scaled_")
'''
shuffled_prefix = "ecg_fold_shuffled_"
clf = knn_factory(9)
accuracy, error = our_utils.evaluate_choose_folds(clf, 2, shuffled_prefix, use_pca=False, scale_data=True,
                                                      pca_comp_num=3)

print("Accuracy on new/transformed set:{acc}".format(acc=accuracy))



print("Done")

'''

#run_section6()
#run_section7()
#try_tree_improvements()
'''
#normalize_set("ecg_fold_0_sub.data",0)
#normalize_set("ecg_fold_1_sub.data",1)

'''





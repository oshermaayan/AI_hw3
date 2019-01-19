import math
import pickle
import numpy as np
import sklearn.preprocessing
from sklearn.decomposition import PCA


def load_data_fold(path):
    with open(path,'rb') as f:
        train_features, train_labels = pickle.load(f)
    return train_features, train_labels

def euclidean_distance(x, y):
    '''
    :param x: first vector of features
    :param y: second vector of features
    :return: the euclidean distance between the two vectors
    '''
    assert len(x)==len(y)

    result = 0.0
    for f1, f2 in zip(x, y):
        result += (f1-f2)**2

    return math.sqrt(result) #was: result**0.5

def split_crosscheck_groups(dataset, num_folds):
    '''

    :param dataset:
    :param num_folds: number of folds to divide the dataset into
    :return: creates and saves num_folds "smaller" datasets
    '''
    #if len(dataset)%num_folds != 0:
    features = dataset[0]
    tags = dataset[1]
    positive_samples_ind = [i for i in range(len(tags)) if tags[i]==True]
    negative_samples_ind = [i for i in range(len(tags)) if tags[i]==False]

    num_positive_samples = len(positive_samples_ind)
    num_negative_samples = len(negative_samples_ind)

    for f in range(num_folds):
        current_features = []
        current_tags = []
        # Add positive samples
        start_ind = math.floor(f*num_positive_samples / num_folds)
        end_ind = math.floor((f+1)* num_positive_samples / num_folds)
        for i in range(start_ind, end_ind):
            current_features.append(features[positive_samples_ind[i]])
            current_tags.append(tags[positive_samples_ind[i]])
            #TODO : consider just adding True to current_tags

        start_ind = f*math.floor(num_negative_samples / num_folds)
        end_ind = (f + 1)*math.floor(num_negative_samples / num_folds)

        for i in range(start_ind, end_ind):
            current_features.append(features[negative_samples_ind[i]])
            current_tags.append(tags[negative_samples_ind[i]])
            #TODO : consider just adding False to current_tags

        #TODO: ask if number of samples is divides by num_folds; if not - how should we proceed?

        #Save fold
        current_fold = (current_features, current_tags)
        fold_name = "ecg_fold_final_pca_{fold}.data".format(fold=f)
        with open(fold_name, 'wb') as fold_file:
            pickle.dump(current_fold, fold_file)


def evaluate(classifier_factory, k):
    '''

    :param classifier_factory: creates classifiers
    :param k: number of folds
    :return:
    '''
    accuracies = []
    errors = []

    for fold in range(k):
        valid_fold_file = "ecg_fold_{foldNum}_sub.data".format(foldNum=str(fold))
        validation_data, validation_labels = load_data_fold(valid_fold_file)
        N = len(validation_data)

        for other_fold in range(k):
            if other_fold!=fold:
                train_fold_file = "ecg_fold_{foldNum}_sub.data".format(foldNum=str(other_fold))
                train_data, train_labels = load_data_fold(train_fold_file)
                classifier = classifier_factory.train(train_data, train_labels)
                #TODO: see if we need to support more than 1 fold as training (concat lists)

                true_classifications = 0
                for sample,true_label in zip(validation_data, validation_labels):
                    classification = classifier.classify(sample)
                    if classification==true_label:
                        true_classifications += 1

                accuracy = true_classifications / N
                accuracies.append(accuracy)
                errors.append(1 - accuracy)

    classifier_accuracy = sum(accuracies)/len(accuracies)
    classifier_error = sum(errors)/len(errors)

    return classifier_accuracy, classifier_error

def evaluate_choose_folds(classifier_factory, k, fold_prefix,
                          scale_data=False, normalize_data=False,
                          use_pca=False,pca_comp_num=30):
    '''

    :param classifier_factory: creates classifiers
    :param k: number of folds
    :param scale_data: should data be scaled (zero mean, unit std)
    :return:
    '''
    accuracies = []
    errors = []

    for fold in range(k):
        valid_fold_file = fold_prefix+str(fold)+".data"
        validation_data, validation_labels = load_data_fold(valid_fold_file)
        N = len(validation_data)

        for other_fold in range(k):
            if other_fold != fold:
                train_fold_file = fold_prefix+str(other_fold)+".data"
                train_data, train_labels = load_data_fold(train_fold_file)

                if scale_data:
                    scaler = sklearn.preprocessing.StandardScaler().fit(train_data)
                    train_data = scaler.transform(train_data)
                    validation_data = scaler.transform(validation_data)

                if normalize_data:
                    normalizer = sklearn.preprocessing.Normalizer().fit(train_data)
                    train_data = normalizer.transform(train_data)
                    validation_data = normalizer.transform(validation_data)

                if use_pca:
                    pca=PCA(n_components=pca_comp_num)
                    pca.fit(train_data)
                    train_data = pca.fit_transform(train_data)
                    validation_data = pca.fit_transform(validation_data)

                classifier = classifier_factory.train(train_data, train_labels)
                #TODO: see if we need to support more than 1 fold as training (concat lists)

                true_classifications = 0
                for sample,true_label in zip(validation_data, validation_labels):
                    classification = classifier.classify(sample)
                    if classification==true_label:
                        true_classifications += 1

                accuracy = true_classifications / N
                accuracies.append(accuracy)
                errors.append(1 - accuracy)

    classifier_accuracy = sum(accuracies)/len(accuracies)
    classifier_error = sum(errors)/len(errors)

    return classifier_accuracy, classifier_error


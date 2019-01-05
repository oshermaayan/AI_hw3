data_path = r'data/Data.pickle'
import pickle
import numpy as np
import math

def load_data(path=r'data/Data.pickle'):
    '''
    return the dataset that will be used in HW 3
    prameters:
    :param path: the path of the csv data file (default value is data/ecg_examples.data)

    :returns: a tuple train_features, train_labels ,test_features
    features - a numpy matrix where  the ith raw is the feature vector of patient i.
    '''
    with open(path,'rb') as f:
        train_features, train_labels, test_features = pickle.load(f)
    return train_features, train_labels ,test_features


# TODO: probably remove this method later
def load_data_fold(path):
    with open(path,'rb') as f:
        train_features, train_labels = pickle.load(f)
    return train_features, train_labels

def write_prediction(pred, path='results.data'):
    '''
    write the prediction of the test set into a file for submission
    prameters:
    :param pred: - a list of result the ith entry represent the ith subject (as integers of 1 or 0, where 1 is a healthy patient and 0 otherwise)
    :param path: - the path of the csv data file will be saved to(default value is res.data)
    :return: None
    '''
    output = []
    for l in pred:
        output.append(l)
    with open(path, 'w') as f:
        f.write(', '.join([str(x) for x in output]) + '\n')

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

    return result**0.5



class abstract_classifier_factory:
    '''
    an abstruct class for classifier factory
    '''
    def train(self, data, labels):
        '''
        train a classifier
        :param data: a list of lists that represents the features that the classifier will be trained with
        :param labels: a list that represents  the labels that the classifier will be trained with
        :return: abstruct_classifier object
        '''
        raise Exception('Not implemented')


class abstract_classifier:
    '''
        an abstruct class for classifier
    '''

    def classify(self, features):
        '''
        classify a new set of features
        :param features: the list of feature to classify
        :return: a tagging of the given features (1 or 0)
        '''
        raise Exception('Not implemented')


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
        fold_name = "ecg_fold_{fold}.data".format(fold=f)
        with open(fold_name, 'wb') as fold_file:
            pickle.dump(current_fold, fold_file)

    #TODO : remove later
    print ("Done creating folds")

def evaluate(classifier_factory, k):
    '''

    :param classifier_factory: creates classifiers
    :param k:
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

    classifer_accuracy = sum(accuracies)/len(accuracies)
    classifier_error = sum(errors)/len(errors)

    return classifer_accuracy, classifier_error

'''
dataset = load_data()
split_crosscheck_groups(dataset,2)
'''


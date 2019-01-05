import hw3_utils as utils
from sklearn import tree

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
            distance = utils.euclidean_distance(feature_vector, sample)
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

class knn_factory(utils.abstract_classifier_factory):
    def __init__(self, k):
        self.k = k

    def train(self, data, labels):
        return knn_classifier(data, labels, k)


class DecisionTree_factory(utils.abstract_classifier_factory):
    def train(self, data, labels):
        dtc = tree.DecisionTreeClassifier()
        dtc = dtc.fit(data, labels)
        return TreeClassifier(dtc)


# Section 6
csv_path = "experiments6.csv"
k_list = [1, 3, 5, 7, 13]

with open(csv_path,"w") as file:
    for k in k_list:
        classifier_factory = knn_factory(k)
        accuracy, error = utils.evaluate(classifier_factory, 2)
        output = "{k},{acc},{error}\n".format(k=k,acc=accuracy,error=error)
        file.write(output)
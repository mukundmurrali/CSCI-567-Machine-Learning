import numpy as np
import utils as Util


class DecisionTree():
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    def train(self, features, labels):
        # features: List[List[float]], labels: List[int]
        # init
        assert (len(features) > 0)
        num_cls = np.unique(labels).size

        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()

        return

    def predict(self, features):
        # features: List[List[any]]
        # return List[int]
        y_pred = []
        for idx, feature in enumerate(features):
            pred = self.root_node.predict(feature)
            y_pred.append(pred)
        return y_pred


class TreeNode(object):
    def __init__(self, features, labels, num_cls):
        # features: List[List[any]], labels: List[int], num_cls: int
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls
        # find the most common labels in current node
        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label
                # splitable is false when all features belongs to one class
        if len(np.unique(labels)) < 2:
            self.splittable = False
        else:
            self.splittable = True

        self.dim_split = None  # the index of the feature to be split

        self.feature_uniq_split = None  # the possible unique values of the feature to be split

    # TODO: try to split current node
    def split(self):
        if len(self.features) == 0 or len(self.labels) == 0 or self.num_cls == 0 \
                or len(self.features) == 0 or len(self.features[0]) == 0 or self.splittable == False:
            self.splittable = False
            return

        classVsCount = {x: self.labels.count(x) for x in self.labels}
        counts = classVsCount.values()

        # sum of the counts
        sum_counts = sum(counts)
        S = 0

        if sum_counts != 0:
            for class_count in counts:
                p = class_count / sum_counts
                S += -(p * np.log2(p))

        max_information_gain = 0
        attribute_length_max = 0
        # for each attribute try to find the maximum information gain
        for i in range(0, len(self.features[0])):
            attribute_length_curr = len(set([row[i] for row in self.features]))

            branches = []
            # attribute to class counts
            attribute_classes_counts = {}
            for j in range(len(self.features)):
                attribute = self.features[j][i]
                classes_count = attribute_classes_counts.get(attribute, {})
                count = classes_count.get(self.labels[j], 0) + 1
                classes_count[self.labels[j]] = count
                attribute_classes_counts[attribute] = classes_count

            for attribute, classes_count in attribute_classes_counts.items():
                counts = list(classes_count.values())
                branches.append(counts)

            feature_information_gain = Util.Information_Gain(S, branches)
            if feature_information_gain > 0:
                if feature_information_gain > max_information_gain:
                    max_information_gain = feature_information_gain
                    self.dim_split = i
                    attribute_length_max = attribute_length_curr
                # tie-break
                elif feature_information_gain == max_information_gain:
                    # len of the attribute
                    if attribute_length_max < attribute_length_curr:
                        max_information_gain = feature_information_gain
                        self.dim_split = i
                        attribute_length_max = attribute_length_curr
                    elif attribute_length_max == attribute_length_curr:
                        if i < self.dim_split:
                            max_information_gain = feature_information_gain
                            self.dim_split = i
                            attribute_length_max = attribute_length_curr

        if self.dim_split is None:
            self.splittable = False
            return
        attributeVsFeatureAndLabel = {}

        for i in range(0, len(self.features)):
            attribute = self.features[i][self.dim_split]
            feature = self.features[i]
            label = self.labels[i]

            # remove the attribute from that row
            feature = np.delete(feature, self.dim_split)
            features, labels = attributeVsFeatureAndLabel.get(attribute, ([], []))
            features.append(feature)
            labels.append(label)
            attributeVsFeatureAndLabel[attribute] = (features, labels)

        self.feature_uniq_split = sorted(attributeVsFeatureAndLabel.keys())

        for attribute in self.feature_uniq_split:
            (features, labels) = attributeVsFeatureAndLabel[attribute]
            child = TreeNode(features, labels, np.unique(labels).size)
            if child.splittable:
                child.split()
            self.children.append(child)

    # TODO: predict the branch or the class
    def predict(self, feature):
        # feature: List[any]
        # return: int
        if self.splittable:
            if self.dim_split is not None and len(self.features) > 0 and feature[self.dim_split] in self.feature_uniq_split:
                attribute = feature[self.dim_split]
                childIndex = self.feature_uniq_split.index(attribute)
                child = self.children[childIndex]
                feature = np.delete(feature, self.dim_split)
                return child.predict(feature)
        return self.cls_max

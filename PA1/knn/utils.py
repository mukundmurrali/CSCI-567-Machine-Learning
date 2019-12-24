import numpy as np
from knn import KNN

############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################


# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    assert len(real_labels) == len(predicted_labels)
    yy1Sum = 0
    ySum = 0
    y1Sum=0

    for y,y1 in zip(real_labels,predicted_labels):
        yy1Sum += y * y1
        ySum += y
        y1Sum += y1

    if y1Sum + ySum != 0 :
        return 2 * (yy1Sum/float(ySum + y1Sum))
    return 0


class Distances:
    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        distance = sum([abs((a - b)) ** 3 for a, b in zip(point1, point2)])
        return distance ** (1. / 3)

    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        distance = [(a - b) ** 2 for a, b in zip(point1, point2)]
        return np.sqrt(sum(distance))

    @staticmethod
    # TODO
    def inner_product_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        distance = [(a * b) for a, b in zip(point1, point2)]
        return np.sum(distance)

    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        if np.all(point1==0):
            point1 += 0.000001
        if np.all(point1==0):
            point1 += 0.000002
        distance = np.sum([(a * b) for a, b in zip(point1, point2)])
        a1 = np.sqrt(np.sum(np.square(point1)))
        b1 = np.sqrt(np.sum(np.square(point2)))
        if (a1 * b1) != 0:
            return 1 - distance / (a1 * b1)
        return -1

    @staticmethod
    # TODO
    def gaussian_kernel_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        distance = [(a - b) ** 2 for a, b in zip(point1, point2)]
        sumDistance = sum(distance)
        return -np.exp(-0.5 * sumDistance)


class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you should try different distance function you implemented in part 1.1, and find the best k.
        Use k range from 1 to 30 and increment by 2. Use f1-score to compare different models.

        :param distance_funcs: dictionary of distance functions you must use to calculate the distance.
            Make sure you loop over all distance functions for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val:  List[List[int]] Validation data set will be used on your KNN predict function to produce
            predicted labels and tune k and distance function.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_function and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function and self.best_model respectively.
        NOTE: self.best_scaler will be None

        NOTE: When there is a tie, choose model based on the following priorities:
        Then check distance function  [euclidean > minkowski > gaussian > inner_prod > cosine_dist]
        If they have same distance fuction, choose model which has a less k.
        """
        
        # You need to assign the final values to these variables
        self.best_k = None
        self.best_distance_function = None
        self.best_model = None

        limit = 30 if len(x_train) > 30 else len(x_train) - 1

        distance_funcs_list_order = ["euclidean", "minkowski" , "gaussian" ,"inner_prod" , "cosine_dist"]

        max_score = 0

        for distance, function in distance_funcs.items():
            for k in range(1, limit, 2):
                knn= KNN(k, function)
                knn.train(x_train,y_train)
                f1 = f1_score(y_val, knn.predict(x_val))
                print("f1 = " ,f1 , max_score)
                if f1 > max_score:
                    self.best_k = k
                    self.best_distance_function = distance
                    self.best_model = knn
                    max_score = f1
                elif f1 == max_score:
                    if self.best_distance_function is None or \
                            distance_funcs_list_order.index(self.best_distance_function) > distance_funcs_list_order.index(distance):
                        self.best_k = k
                        self.best_distance_function = distance
                        self.best_model = knn
                    elif distance_funcs_list_order.index(self.best_distance_function) == distance_funcs_list_order.index(distance) and self.best_k > k:
                        self.best_k = k
                        self.best_distance_function = distance
                        self.best_model = knn


    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is similar to Part 1.3 except that before passing your training and validation data to KNN model to
        tune k and disrance function, you need to create the normalized data using these two scalers to transform your
        data, both training and validation. Again, we will use f1-score to compare different models.
        Here we have 3 hyperparameters i.e. k, distance_function and scaler.

        :param distance_funcs: dictionary of distance funtions you use to calculate the distance. Make sure you
            loop over all distance function for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param scaling_classes: dictionary of scalers you will use to normalized your data.
        Refer to test.py file to check the format.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data set you will use on your KNN predict function to produce predicted
            labels and tune your k, distance function and scaler.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_funtion, scaler and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function, self.best_scaler and self.best_model respectively

        NOTE: When there is a tie, choose model based on the following priorities:
        For normalization, [min_max_scale > normalize];
        Then check distance function  [euclidean > minkowski > gaussian > inner_prod > cosine_dist]
        If they have same distance function, choose model which has a less k.
        """
        
        # You need to assign the final values to these variables
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None
        limit = 30 if len(x_train) > 30 else len(x_train)
        scaler_order_list = ["min_max_scale", "normalize"]
        distance_funcs_list_order = ["euclidean", "minkowski", "gaussian", "inner_prod", "cosine_dist"]
        max_score = 0
        for scaler, function_scaler in scaling_classes.items():
            for distance, function in distance_funcs.items():
                for k in range(1, limit, 2):
                    scaler_method = function_scaler()
                    x_train_scaled = scaler_method(x_train)
                    x_val_scaled = scaler_method(x_val)
                    knn = KNN(k, function)
                    knn.train(x_train_scaled, y_train)
                    f1 = f1_score(y_val, knn.predict(x_val_scaled))
                    if f1 > max_score:
                        self.best_k = k
                        self.best_distance_function = distance
                        self.best_model = knn
                        self.best_scaler = scaler
                        max_score = f1
                    elif f1 == max_score:
                        if self.best_scaler is None or  scaler_order_list.index(
                                self.best_scaler) > scaler_order_list.index(scaler):
                            self.best_k = k
                            self.best_distance_function = distance
                            self.best_model = knn
                            self.best_scaler = scaler
                        elif  scaler_order_list.index(
                                self.best_scaler) == scaler_order_list.index(scaler) and distance_funcs_list_order.index(
                                self.best_distance_function) > distance_funcs_list_order.index(distance):
                            self.best_k = k
                            self.best_distance_function = distance
                            self.best_model = knn
                            self.best_scaler = scaler
                        elif distance_funcs_list_order.index(
                                self.best_distance_function) == distance_funcs_list_order.index(
                                distance) and self.best_k > k:
                            self.best_k = k
                            self.best_distance_function = distance
                            self.best_model = knn
                            self.best_scaler = scaler

class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        normalizedFeatures = []
        for feature in features:
            normalizer = np.sqrt(np.sum(np.square(feature)))
            if normalizer == 0:
                normalizedFeatures.append(feature)
            else:
                normalizedFeatures.append(np.true_divide(feature,normalizer))
        return normalizedFeatures

class MinMaxScaler:
    """
    Please follow this link to know more about min max scaling
    https://en.wikipedia.org/wiki/Feature_scaling
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
    will be the training set.

    Hints:
        1. Use a variable to check for first __call__ and only compute
            and store min/max in that case.

    Note:
        1. You may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler1 = MinMaxScale()
        train_features_scaled = scaler1(train_features)
        # train_features_scaled should be equal to [[0, 1], [1, 0]]

        test_features_scaled = scaler1(test_features)
        # test_features_scaled should be equal to [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """

    def __init__(self):
        self.min = None
        self.max = None
        self.minmax = None

    def __call__(self, features):
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        dNumpy = np.array(features)
        if self.min is None or self.max is None:
            self.min = np.amin(dNumpy, axis=0)
            self.max = np.amax(dNumpy,axis=0)
            self.minmax = self.max - self.min
            self.minmax[self.minmax == 0] = 1
        minmaxNormalized = (dNumpy - self.min) / self.minmax
        return np.array(minmaxNormalized.tolist())

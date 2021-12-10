import sys
import numpy as np

# <editor-fold desc="Global Variables">

ETA = 0.1
EPOCHS = 200
LEARNING_RATE = 0.1

# </editor-fold>


# <editor-fold desc="Global Functions">

def euclideanDistance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))


def getTauValue(Y_weightVec, Y_hat_weightVec, x):
    """
    Function role: calculate the value of TAU in th Passive Aggressive algorithm.
    """
    norm = np.linalg.norm(x)
    if norm == 0:
        norm = 1
    loss = np.maximum(0, 1 - np.dot(Y_weightVec, x) + np.dot(Y_hat_weightVec, x))
    return loss / (2 * (norm ** 2))


def getLabels(Y_train):
    labels = set()
    for y in Y_train:
        labels.add(int(y))
    return sorted(labels)


def zScoreNormalizationTrain(data):
    """
    Function role: normalize the given data set (for train).
    """
    mean = []
    std = []
    features = data.shape[1]
    for i in range(features):
        std.append(np.std(data[:, i]))
        mean.append(np.mean(data[:, i]))
        if std[i] == 0 or std[i] == 0.0:
            continue
        data[:, i] = (data[:, i] - mean[i]) / std[i]
    return data, std, mean


def zScoreNormalizationTest(data, stdArray, meanArray):
    """
    Function role: normalize the given data set (for test).
    """
    features = data.shape[1]
    for i in range(features):
        if stdArray[i] == 0 or stdArray[i] == 0.0:
            continue
        data[:, i] = (data[:, i] - meanArray[i]) / stdArray[i]
    return data


def setBias(data):
    """
    Function role: add a "bias column" for each x in the given data set (column of 1's).
    """
    temp = [[1] for i in range(len(data))]
    data = np.append(data, temp, axis=1)
    return data


def shuffle(X_train, Y_train):
    """
    Function role: shuffle the given data.
    """
    data = list(zip(X_train, Y_train))
    np.random.shuffle(data)
    X_train, Y_train = zip(*data)
    X_train = np.asarray(X_train)
    Y_train = np.asarray(Y_train)
    return X_train, Y_train


def executeAlgorithms(X_test, X_train, Y_train, output):
    """
    Function role: execute all the implemented algorithms (after shuffling the data).
    First, the KNN algorithm is executed (before the process of normalization & adding the bias column).
    Second, we normalize the given data sets (train_X & test_X).
    Third, we add a bias column to the given data sets (train_X & test_X).
    Fourth, the PERCEPTRON, PA & SVM algorithms are executed.
    Finally, we create on output file with all the algorithms outputs.
    """

    X_train, Y_train = shuffle(X_train, Y_train)

    _knn = MultiClassKNN(X_test, X_train, Y_train)
    _knnLabels = _knn.train()

    X_test, std, mean = zScoreNormalizationTrain(X_test)
    X_train = zScoreNormalizationTest(X_train, std, mean)

    X_test = setBias(X_test)
    X_train = setBias(X_train)

    _perceptron = MultiClassPerceptron(X_test, X_train, Y_train)
    _perceptronLabels = _perceptron.train()

    _svm = MultiClassSVM(X_test, X_train, Y_train)
    _svmLabels = _svm.train()

    _pa = MultiClassPA(X_test, X_train, Y_train)
    _paLabels = _pa.train()

    f = open(output, "a")
    for i in range(0, len(_knnLabels)):
        f.write(f"knn: {_knnLabels[i]}, perceptron: {_perceptronLabels[i]}, svm: {_svmLabels[i]}, pa: {_paLabels[i]}\n")


# </editor-fold>


# <editor-fold desc="Algorithms">

class MultiClassKNN:
    def __init__(self, X_test, X_train, Y_train):
        self.X_test = X_test
        self.X_train = X_train
        self.Y_train = Y_train

        self.N_samples, self.N_features = self.X_train.shape
        self.k = int(np.ceil(np.sqrt(self.N_samples)))

    def train(self, X_test=None, X=None, Y=None, k=None):
        """
        Function role: train the KNN algorithm.

        First, if we are not passing any arguments to the function - we will use the "default arguments"
        (I did it because we use this function with different sizes of X & Y - to predict the best k value).
        Second, in each iteration we pick one test sample and predict its the y value (label).
        Third, we return the predicted labels.
        """
        if X is None:
            X_test = self.X_test
            X = self.X_train
            Y = self.Y_train
            k = self.k

        y_hat_result = []
        for x_i in X_test:
            y_hat_result.append(self.predict(X, Y, x_i, k))
        return y_hat_result

    def predict(self, X, Y, x, k):
        """
        Function role: predict the label of a given example.

        First, we calculate the distance of the current sample from all other samples and add them to distances array.
        Second, we sort the distance array and return the first k neighbors (after the sort).
        Third, we create an array that holds the labels of the first k neighbors - that we found in step two.
        """
        distances = []
        for x_i in X:
            distances.append(euclideanDistance(x, x_i))

        kNeighborsIndex = np.argsort(distances)[: k]
        kNeighborsClasses = [Y[i] for i in kNeighborsIndex]

        return int(max(list(kNeighborsClasses), key=kNeighborsClasses.count))

    def predictK(self):
        """
        Function role: predict the most successful k value for the KNN algorithm.

        First, we create new train & test samples:
        The first 70% of the X_train & Y_train values - will be our new train examples and labels.
        The other 30% of the X_train & Y_train values - will be our new test examples (and labels (*)).
        Second, we iterate all values from 1 to the root of the number of examples, and in every iteration we "train"
        the new values that we creates in previous step. Because we know the labels of the new "test examples" (*),
        at the end of each iteration we will check for which K we got the highest percentage of accuracy - and take it.
        """
        K_X_train, K_Y_train, K_X_test, K_Y_test = [], [], [], []
        trainValue = int(np.ceil(self.N_samples * 0.7))
        testValue = int(self.N_samples - trainValue)

        for raw in range(trainValue):
            K_X_train.append(self.X_train[raw])
            K_Y_train.append(self.Y_train[raw])
        for raw in range(testValue):
            K_X_test.append(self.X_train[trainValue + raw])
            K_Y_test.append(int(self.Y_train[trainValue + raw]))

        k, matchIndex = 0, 0
        for kVal in range(1, int(np.ceil(np.sqrt(self.N_samples)))):
            K_Y_hat = self.train(K_X_test, K_X_train, K_Y_train, kVal)
            temp = np.sum(np.array(K_Y_hat) == np.array(K_Y_test))
            if temp > matchIndex:
                k = kVal
                matchIndex = temp
        return k


class MultiClassPerceptron:
    def __init__(self, X_test, X_train, Y_train):
        self.eta = ETA

        self.X_test = X_test
        self.X_train = X_train
        self.Y_train = Y_train

        self.N_samples, self.N_features = self.X_train.shape

        self.labels = getLabels(self.Y_train)
        bias = [[1] for i in range(len(self.labels))]

        self.W_vectors = np.zeros((len(self.labels), self.N_features - 1))
        self.W_vectors = np.append(self.W_vectors, bias, axis=1)


    def train(self):
        """
        Function role is to train the Multi-Class Perceptron algorithm.

        First, in every new iteration we shuffle the X,Y train data .
        Second, we pick in each iteration a x_i & y_i corresponding values:
        x_i - we are using the predict method that returns us the predicted y_hat value.
        Third, if the original y (right label) and the y_hat (predicted label) are not equal, then we need to update:
        1) The vector that we didnt anticipate his Y: we will add to it the x_i feature vector (including eta)
        2) The vector that we anticipate his Y: we will subtract from it the x_i feature vector (including eta)
        [In each iteration we update the value of the ETA parameter to be lower]
        """
        for i in range(EPOCHS):
            for x_i, y_i in zip(self.X_train, self.Y_train):
                y_hat = np.argmax(np.dot(self.W_vectors, x_i))

                if not (int(y_i) == y_hat):
                    self.W_vectors[int(y_i)] = self.W_vectors[int(y_i)] + (self.eta * x_i)
                    self.W_vectors[y_hat] = self.W_vectors[y_hat] - (self.eta * x_i)
                self.eta *= 0.9999

        y_hat_result = []
        for x in self.X_test:
            y_hat_result.append(np.argmax(np.dot(self.W_vectors, x)))

        return y_hat_result


class MultiClassPA:
    def __init__(self, X_test, X_train, Y_train):
        self.X_test = X_test
        self.X_train = X_train
        self.Y_train = Y_train

        self.N_samples, self.N_features = self.X_train.shape

        self.labels = getLabels(self.Y_train)
        bias = [[1] for i in range(len(self.labels))]

        self.W_vectors = np.zeros((len(self.labels), self.N_features - 1))
        self.W_vectors = np.append(self.W_vectors, bias, axis=1)

    def train(self):
        """
        Function role is to train the Multi-Class Passive Aggressive algorithm.

        First, in every new iteration we shuffle the X,Y train data .
        Second, we pick in each iteration a x_i & y_i corresponding values:
        x_i - we are using the predict method that returns us the predicted y_hat value.
        Third, if the original y (right label) and the y_hat (predicted label) are not equal, then we need to update:
        1) The vector that we didnt anticipate his Y: we will add to it the x_i feature vector (including eta)
        2) The vector that we anticipate his Y: we will subtract from it the x_i feature vector (including eta)
        """
        for i in range(EPOCHS):
            for x_i, y_i in zip(self.X_train, self.Y_train):
                y_hat = np.argmax(np.dot(self.W_vectors, x_i))

                if not (int(y_i) == y_hat):
                    tau = getTauValue(self.W_vectors[int(y_i), :], self.W_vectors[int(y_hat), :], x_i)
                    learningRate = tau * x_i * 0.4 * (1 - i / EPOCHS)
                    self.W_vectors[int(y_i)] = self.W_vectors[int(y_i)] + learningRate
                    self.W_vectors[y_hat] = self.W_vectors[y_hat] - learningRate

        y_hat_result = []
        for x in self.X_test:
            y_hat_result.append(np.argmax(np.dot(self.W_vectors, x)))

        return y_hat_result


class MultiClassSVM:
    def __init__(self, X_test, X_train, Y_train):
        self.eta = ETA

        self.X_test = X_test
        self.X_train = X_train
        self.Y_train = Y_train

        self.N_samples, self.N_features = self.X_train.shape

        self.labels = getLabels(self.Y_train)
        bias = [[1] for i in range(len(self.labels))]

        self.W_vectors = np.zeros((len(self.labels), self.N_features - 1))
        self.W_vectors = np.append(self.W_vectors, bias, axis=1)

    def train(self):
        for i in range(EPOCHS):
            for x_i, y_i in zip(self.X_train, self.Y_train):
                y_hat = np.argmax(np.dot(self.W_vectors, x_i))

                if not (int(y_i) == y_hat):
                    learningRate = 1 - (self.eta * LEARNING_RATE)
                    self.W_vectors[int(y_i), :] = (self.W_vectors[int(y_i)] * learningRate) + (self.eta * x_i)
                    self.W_vectors[y_hat, :] = (self.W_vectors[y_hat, :] * learningRate) - (self.eta * x_i)
                self.eta *= 0.9999

        y_hat_result = []
        for x in self.X_test:
            y_hat_result.append(np.argmax(np.dot(self.W_vectors, x)))

        return y_hat_result


# </editor-fold>


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Error: Insufficient arguments!")
        sys.exit()
    else:
        _trainingExample = np.loadtxt(sys.argv[1], delimiter=",")
        _trainingLabels = np.loadtxt(sys.argv[2], delimiter=",")
        _testExamples = np.loadtxt(sys.argv[3], delimiter=",")
        _outputFile = sys.argv[4]

        executeAlgorithms(_testExamples, _trainingExample, _trainingLabels, _outputFile)

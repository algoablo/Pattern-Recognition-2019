import numpy as np
import sys
import csv
from sklearn.neural_network import MLPClassifier
from datetime import datetime

def read_data(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    matrix = np.array(data, dtype=int)
    # separate labels from samples
    samples = matrix[:, 1:]
    labels = matrix[:, 0]
    return labels, samples


def main():
    train_filename = 'mnist_train.csv'
    test_filename = 'mnist_test.csv'

    train_labels, train_samples = read_data(train_filename)
    classifier = MLPClassifier(activation='logistic', solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100,), random_state=1)
    start_total = datetime.now()
    classifier.fit(train_samples, train_labels)
    print('Done with training the MLP.')
    print('Now, predicting the test dataset')

    test_labels, test_samples = read_data(test_filename)
    predicted = classifier.predict(test_samples)
    success = 0
    for i in range(len(predicted)):
        if (test_labels[i] == predicted[i]):
            success+=1

    end = datetime.now()
    print('Total runtime: {duration}'.format(duration=end - start_total))
    print('Success rate: {success_rate}'.format(success_rate = success/len(test_samples)))

if __name__ == '__main__':
    main()

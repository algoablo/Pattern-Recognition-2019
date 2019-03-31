import numpy as np
import csv
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
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
    start_total = datetime.now()
    best_classifier = None
    max_accuracy = 0
    for iter in range(5):
        iteration = (iter + 1) * 200
        for neu in range(5):
            neurons = (neu + 1) * 20
            for lr in range(5):
                learning_rate = (lr + 1) * 0.02
                classifier = MLPClassifier(activation='logistic', solver='sgd', hidden_layer_sizes=(neurons,),
                                           max_iter=iteration, learning_rate='constant',
                                           learning_rate_init=learning_rate)
                k_folds = KFold(n_splits=4)
                scores = []
                for train, test in k_folds.split(train_samples):
                    train_samples_cv = train_samples[train]
                    train_labels_cv = train_labels[train]

                    """Train the MLP"""
                    trained_mlp = classifier.fit(train_samples_cv, train_labels_cv)

                    """Test the classifier"""
                    test_samples_cv = train_samples[test]
                    test_labels_cs = train_labels[test]
                    score = trained_mlp.score(test_samples_cv, test_labels_cs)
                    scores.append(score)
                print('For iterations = {iter}'.format(iter=iteration))
                print('For number of neurons on the hidden layer = {neurons}'.format(neurons=neurons))
                print('For learning rate = {learning_rate}'.format(learning_rate=learning_rate))
                print('The following accuracies are obtained for four-fold CV: {acc}'.format(acc=scores))
                accuracy = np.mean(scores)
                print('Average accuracy = {acc:.4f}'.format(acc=accuracy))
                print('---------------------------------------------------')
                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    best_classifier = classifier
    print('Best accuracy found with the MLP with the following parameters: {params}'.format(params=best_classifier))

    end = datetime.now()
    print('Total runtime: {duration}'.format(duration=end - start_total))


if __name__ == '__main__':
    main()

import csv
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from multiprocessing import Pool
from functools import partial
import time


def read_features(test=False):
    path = "features/" + ("valid" if test else "train") + "/features.csv"
    with open(path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    matrix = np.array(data)
    data = {}
    for row in matrix:
        key = row[0]
        data[key] = []
        i = 1
        while i < len(row):
            size = row[i]
            lower_contour = row[i + 1]
            upper_contour = row[i + 2]
            b_w_transitions = row[i + 3]
            black_pixels = row[i + 4]
            data[key].append([size, lower_contour, upper_contour, b_w_transitions, black_pixels])
            i += 5
    return data


def read_keywords():
    keywords = []
    path = "task/keywords.txt"
    with open(path, "r") as train_or_valid_txt:
        file_content = train_or_valid_txt.readlines()
    for word in file_content:
        keywords.append(word.replace("\n", ""))
    return keywords


def read_transcription():
    transcriptions = {}
    path = "ground-truth/transcription.txt"
    with open(path, "r") as train_or_valid_txt:
        file_content = train_or_valid_txt.readlines()
    for line in file_content:
        words = line.split()
        image_id = words[0]
        transcription = words[1]
        transcriptions[image_id] = transcription
    return transcriptions


def calculate_dwt(test_feature, training_features):
    distance, path = fastdtw(test_feature, training_features, dist=euclidean)
    return distance


def write_res(result):
    path = "result.txt"
    file = open(path, "w+")
    file_content = ""
    for word, res in result.items():
        file_content = file_content + word + ", "
        for t, k in res:
            file_content = file_content + str(t) + ", " + str(k) + ", "
        file_content = file_content[:-2]
        file_content = file_content + "\n"
    file.write(file_content)
    file.close()


def calculate_avg_precision(word, transcriptions, test_features, result):
    precisions = []
    count = 0
    # Initially count how many time the word is in validation data / test_features.
    for key in test_features.keys():
        if transcriptions[key] == word:
            count += 1
    print(count)
    true_positives = 0
    false_positives = 0
    for image_id, distance in result:
        if true_positives == count:
            break
        if word == transcriptions[image_id]:
            true_positives += 1
        else:
            false_positives += 1
        precisions.append(true_positives / (true_positives + false_positives))
    mean = np.mean(precisions) if len(precisions) > 0 else 0
    return mean


def is_word_in_test_features(word, test_features, transcriptions):
    for key in test_features.keys():
        if transcriptions[key] == word:
            return True


def main():
    start_time = time.time()
    training_features = read_features()
    test_features = read_features(test=True)
    keywords = read_keywords()
    transcriptions = read_transcription()
    result = {}
    precisions = []
    for word in keywords:
        training_feature_sequence_vectors = []
        result[word] = {}
        for image_id, w in transcriptions.items():
            if word == w:
                if image_id in training_features.keys():
                    training_feature_sequence_vectors.append(training_features[image_id])
        if len(training_feature_sequence_vectors) > 0:
            threads = Pool(len(training_feature_sequence_vectors))
            for test_image_id, test_feature_sequence_vectors in test_features.items():
                func = partial(calculate_dwt, test_feature_sequence_vectors)
                dwt_dst = np.min(threads.map(func, training_feature_sequence_vectors))
                result[word][test_image_id] = dwt_dst
            threads.close()
            # Sort the result in ascending order
            result[word] = sorted(result[word].items(), key=lambda kv: (kv[1], kv[0]))
            # Calculate the precision for this predicted word
            # precision = calculate_avg_precision(word, transcriptions, test_features, result[word])
            # precisions.append(precision)
            print('Finished processing word = {w}'.format(w=word))
    end_time = time.time()
    print(end_time - start_time)
    print('Finished recognition.')
    # print('Average mean precision = {prcs}'.format(prcs=np.mean(precisions)))
    print('Writing result to file result.txt')
    write_res(result)


if __name__ == '__main__':
    main()

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
        file_content = file_content + word + ":"
        for t, k in res:
            file_content = file_content + "(" + str(t) + ", " + str(k) + "), "
        file_content = file_content + "\n"
    file.write(file_content)
    file.close()


def main():
    start_time = time.time()
    training_features = read_features()
    test_features = read_features(test=True)
    keywords = read_keywords()
    transcriptions = read_transcription()
    result = {}
    for word in keywords:
        training_feature_sequence_vectors = []
        result[word] = {}
        for image_id, w in transcriptions.items():
            if word == w:
                if image_id in training_features.keys():
                    training_feature_sequence_vectors.append(training_features[image_id])
        threads = Pool(len(training_feature_sequence_vectors))
        for test_image_id, test_feature_sequence_vectors in test_features.items():
            func = partial(calculate_dwt, test_feature_sequence_vectors)
            dwt_dst = np.mean(threads.map(func, training_feature_sequence_vectors))
            result[word][test_image_id] = dwt_dst
        threads.close()
        result[word] = sorted(result[word].items(), key=lambda kv: (kv[1], kv[0]))
        print('Finished processing word = {w}'.format(w=word))
    end_time = time.time()
    print(result)
    print(end_time - start_time)
    print('Finished recognition.')
    print('Writing result to file result.txt')
    write_res(result)


if __name__ == '__main__':
    main()

import os
import cv2
import csv
import numpy as np


def read_processed_image(test=False):
    processed_images = {}
    path = 'preprocessed_images/' + ('valid' if test else 'train')
    image_names = os.listdir(path)
    for image in image_names:
        image_path = path + "/" + image
        processed_images[image.replace(".jpg", "")] = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return processed_images


def is_black_pixel(pixel):
    return pixel < 100


def extract_features(current_window):
    features = []
    lower_contour = 0
    upper_contour = 0
    b_w_transitions = 0
    black_pixels = 0
    found_upper_contour = False
    i = 0
    while (i + 1) < len(current_window):
        current_pixel = current_window[i]
        next_pixel = current_window[i + 1]

        current_pixel_black = is_black_pixel(current_pixel)
        # If it is a black pixel, count it.
        black_pixels = (black_pixels + 1) if current_pixel_black else black_pixels

        next_pixel_black = is_black_pixel(next_pixel)

        # If any of the pixels is not black -> black/white transition!
        if current_pixel_black != next_pixel_black:
            b_w_transitions += 1
            # If this is the first time finding a new b/w transition -> this is the upper contour
            if not found_upper_contour:
                found_upper_contour = True
                upper_contour = i
            # Keep the last b/w transition as the lower contour
            lower_contour = i
        i += 1
    black_pixels = (black_pixels + 1) if is_black_pixel(current_window[len(current_window) - 1]) else black_pixels
    features.append(lower_contour)
    features.append(upper_contour)
    features.append(b_w_transitions)
    features.append(black_pixels)
    return features


def write_features(features, test=False):
    path = "features/" + ("valid" if test else "train")
    if not os.path.exists(path):
        os.makedirs(path)
    path = path + "/features.csv"
    with open(path, "w+", newline='') as my_csv:
        csv_writer = csv.writer(my_csv, delimiter=',')
        csv_writer.writerows(features)


def process_features(test=False):
    images = read_processed_image(test)
    feature_set = list([])
    for name, image in images.items():
        features = list([])
        features.append(name)
        # shape[0] = colons
        # shape[1] = rows
        scaled_image = cv2.resize(image, (100, 100), interpolation=cv2.INTER_NEAREST)
        for i in range(scaled_image.shape[0]):
            # Get all the rows of the i-th column (column-wise) -> Sliding window approach
            current_window = scaled_image[:, i]
            # Get the features from this column
            lower_contour, upper_contour, b_w_transitions, black_pixels = extract_features(current_window)
            features.append(image.shape[1] / image.shape[0])
            features.append(lower_contour)
            features.append(upper_contour)
            features.append(b_w_transitions)
            features.append(black_pixels)
        feature_set.append(features)
    return feature_set


def normalization_parameters(data):
    data = np.array(((np.array(data))[:, 1:]), dtype=float)
    means = []
    stds = []
    for i in range(0, len(data)):
        temp = []
        for j in range(0, len(data[i]), 5):
            temp_array = []
            temp_array.append(data[i][j])
            temp_array.append(data[i][j + 1])
            temp_array.append(data[i][j + 2])
            temp_array.append(data[i][j + 3])
            temp_array.append(data[i][j + 4])
            temp.append(temp_array)
        means.append(np.mean(temp, axis=0))
        stds.append((np.mean(temp, axis=0)))
    mean = np.mean(means, axis=0)
    std = np.std(stds, axis=0)  # np.std([np.mean(data[n], axis=0) for n in data], axis=0)
    return mean, std


def normalize(data, mean, std):
    data = np.array(data)
    numbers = np.array((data[:, 1:]), dtype=float)
    for i in range(0, len(numbers)):
        for j in range(0, len(numbers[i]), 5):
            x = numbers[i][j]
            numbers[i][j] = "{0:.5f}".format((numbers[i][j] - mean[j % 5]) / std[j % 5])
            numbers[i][j + 1] = "{0:.5f}".format((numbers[i][j + 1] - mean[(j + 1) % 5]) / std[(j + 1) % 5])
            numbers[i][j + 2] = "{0:.5f}".format((numbers[i][j + 2] - mean[(j + 2) % 5]) / std[(j + 2) % 5])
            numbers[i][j + 3] = "{0:.5f}".format((numbers[i][j + 3] - mean[(j + 3) % 5]) / std[(j + 3) % 5])
            numbers[i][j + 4] = "{0:.5f}".format((numbers[i][j + 4] - mean[(j + 4) % 5]) / std[(j + 4) % 5])
    numbers = numbers.tolist()
    data[:, 1:] = numbers
    return data


def main():
    np.set_printoptions(suppress=True, formatter={'float_kind': '{:0.5f}'.format})
    print('Handling training data.')
    train_features = process_features(test=False)
    mean, std = normalization_parameters(train_features)
    train_features = normalize(train_features, mean, std)
    write_features(train_features, test=False)
    print('--------Finished--------')

    print('Handling validation data')
    test_features = process_features(test=True)
    test_features = normalize(test_features, mean, std)
    write_features(test_features, test=True)
    print('--------Finished--------')


if __name__ == '__main__':
    main()

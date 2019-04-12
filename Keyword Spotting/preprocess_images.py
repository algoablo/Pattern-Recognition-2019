import numpy as np
import cv2
import xml.etree.ElementTree as ET


def read_images(test=True):
    path = "task/" + ("valid.txt" if test else "train.txt")
    with open(path, "r") as train_or_valid_txt:
        numbers = train_or_valid_txt.readlines()
    images = {}
    for number in numbers:
        path = "images/" + number.replace("\n", "") + ".jpg"
        images[number.replace("\n", "")] = cv2.imread(path, 0)
    return images


def read_svg_file(file_number):
    file_path = 'ground-truth/locations/' + file_number + '.svg'
    with open(file_path, "r") as svg_file:
        file_content = svg_file.read()
    return file_content


def is_number(element):
    try:
        float(element)
        return True
    except ValueError:
        return False


def crop_images(test=False):
    # Handle training images images
    train_images = read_images(test)
    for image_number, image in train_images.items():
        xml_content = read_svg_file(image_number)
        root = ET.fromstring(xml_content)
        height = image.shape[0]
        width = image.shape[1]
        for child in root:
            attributes = child.attrib
            # Here will be stored as list the content of the attribute 'd'
            # from the element path of the respective svg
            # e.g. ['M', '243.43', '241.43', 'L', '250.00', '242.00', 'L', '250.19', '248.19', 'L', ...]
            entries = attributes['d'].split()
            polygon = []
            i = 0
            # Create tuples of length two from every two numbers from entries and avoid non numeric values
            # e.g. from the example above, the polygon array would be:
            # [[243.43, 241.43], [250.00, 242.00], [250.19, 248.19], ...]
            while i < len(entries):
                point = []
                if is_number(entries[i]) and (i + 1) < len(entries) and is_number(entries[i + 1]):
                    point.append(float(entries[i]))
                    point.append(float(entries[i + 1]))
                    polygon.append(point)
                i += 1
            # points_umat = cv2.UMat(polygon)
            # x, y, w, h = cv2.boundingRect(points_umat)
            x = []
            y = []
            for point in polygon:
                x.append(point[0])
                y.append(point[1])
            x_min = int(min(x))
            x_max = int(max(x))
            y_min = int(min(y))
            y_max = int(max(y))
            img = image[y_min:y_max, x_min:x_max]
            image_id = attributes['id']
            path = "preprocessed_images/" + ("valid/" if test else "train/") + image_id + ".jpg"
            cv2.imwrite(path, img)
            # Remove comments from the following lines
            # if you want to show the cropped images.
            # cv2.imshow('', img)
            # cv2.waitKey(0)


def main():
    crop_images()
    crop_images(test=True)


if __name__ == '__main__':
    main()

# Keyword spotting

### 1 Image preprocessing (preprocess_images.py)

The documents/images are preprocessed based on the data on the svg files. The words are extracted based on the **d** property of the **path** tag. 
Based on these data a foursquare is created and that part is croped and a "word image" is created. These images are stored in the folder
preprocessed_images/train and preprocessed_images/valid for training and validation data.

### 2 Features (feature_handling.py)
The features are extracted from the preprocessed images. Features are extracted on sliding-windows where one window is one pixel wide and the height is 
as many pixels as the photo is high. The following features are extracted:
 - Lower contour
 - Upper contour
 - Black / white pixel transitions
 - Ratio of the width / height of the preprocessed image
 
 The features are normalized: mean and standard deviation are found on a per-feature basis and then the data are normalized.
 
 ### 3 Word recognition (word_recognition.py)
 
For every **key word** in the **keywords.txt** file:
 - All the same words in the training data are found.
 - For every word in the validation data dtw distance is calculated. Then the minimum of these calculations is considered as the distance
 of the **key word** and the respective word from validation data.
 - The result is sorted based on the distance.
 - Mean average precision is calculated.
 - The result is stored in **result.txt** file.
 

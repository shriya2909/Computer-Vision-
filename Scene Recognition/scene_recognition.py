
# coding: utf-8

# In[38]:

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from scipy import stats
from pathlib import Path, PureWindowsPath


def extract_dataset_info(data_path):
    # extract information from train.txt
    f = open(os.path.join(data_path, "train.txt"), "r")
    contents_train = f.readlines()
    label_classes, label_train_list, img_train_list = [], [], []
    for sample in contents_train:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        if label not in label_classes:
            label_classes.append(label)
        label_train_list.append(sample[0])
        img_train_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))
    print('Classes: {}'.format(label_classes))

    # extract information from test.txt
    f = open(os.path.join(data_path, "test.txt"), "r")
    contents_test = f.readlines()
    label_test_list, img_test_list = [], []
    for sample in contents_test:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        label_test_list.append(label)
        img_test_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))  # you can directly use img_path if you run in Windows

    return label_classes, label_train_list, img_train_list, label_test_list, img_test_list


def compute_dsift(img, stride, size):
    sift = cv2.xfeatures2d.SIFT_create()
    kp = [cv2.KeyPoint(x, y, size) for x in range(0, img.shape[0], stride) 
                                    for y in range(0, img.shape[1], stride)]

    kp, dense_feature = sift.compute(img, kp)
    return dense_feature


def get_tiny_image(img, output_size):
    img = cv2.resize(img,output_size)
    feature = np.reshape(img,(output_size[0]*output_size[1],1))
    feature_nm = (np.double(feature) - np.mean(feature))
    unit_length = np.linalg.norm(feature_nm)
    feature_nm = feature_nm / unit_length 
    feature = np.reshape(feature_nm,output_size)
    return feature


def predict_knn(feature_train, label_train, feature_test, k):
    label_test_pred = []
    knn_classifier = NearestNeighbors(n_neighbors = k, algorithm='auto').fit(feature_train)
    distnace, indices = knn_classifier.kneighbors(feature_test)
    
    for i in range(0, indices.shape[0]):
        voting_scheme = {
        "Kitchen": 0,
        "Store": 0,
        "Bedroom": 0,
        "LivingRoom": 0,
        "Office": 0,
        "Industrial": 0,
        "Suburb": 0,
        "InsideCity": 0,
        "TallBuilding": 0,
        "Street": 0,
        "Highway": 0,
        "OpenCountry": 0,
        "Coast": 0,
        "Mountain": 0,
        "Forest": 0
        }
        for j in range(0, indices.shape[1]):
            voting_scheme[label_train[indices[i][j]]] = voting_scheme[label_train[indices[i][j]]] + 1
        label_test_pred.append(max(voting_scheme, key=lambda key: voting_scheme[key]))
    return label_test_pred


def classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
        
    feature_train = []
    feature_test = []
    output_size = (16,16)
    
    for i in img_train_list:
        t = cv2.imread(i, 0)
        f = get_tiny_image(t,output_size)
        f = f.flatten()
        feature_train.append(f)
    
    for i in img_test_list:
        t = cv2.imread(i, 0)
        f = get_tiny_image(t,output_size)
        f = f.flatten()
        feature_test.append(f)
        
    feature_train = np.asarray(feature_train)
    feature_test = np.asarray(feature_test)
    
    label_train_list = np.asarray(label_train_list).flatten()
    
    k = 3
    label_test_pred = predict_knn(feature_train, label_train_list, feature_test, k)
    confusion = build_confusion_matrix(label_test_list, label_test_pred)
    diagonals = confusion.diagonal()
    accuracy = (np.sum(diagonals)/len(label_test_list))*100
    
    visualize_confusion_matrix(confusion, accuracy, label_classes)
    
    return confusion, accuracy


def build_visual_dictionary(dense_feature_list, dic_size):
    dense_feature_list = np.concatenate(dense_feature_list, axis=0).astype('float32')      
    kmeans = KMeans(n_clusters = dic_size, max_iter = 1000).fit(dense_feature_list)
    vocab = kmeans.cluster_centers_
    return vocab


def compute_bow(feature, vocab):
    nbrs = NearestNeighbors(n_neighbors = 1, algorithm = 'auto').fit(vocab)
    distances, indices = nbrs.kneighbors(feature)
    bow_feature, bin_edges = np.histogram(indices, bins = len(vocab))
    bow_feature = bow_feature/np.linalg.norm(bow_feature)
    return bow_feature


def classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    # To do
    dic_size = 50
    dense_feature_list = []
    sift_feature_list_train = []
    for img in img_train_list:
        image = cv2.imread(img, 0)
        feature = compute_dsift(image, 10, 10)
        sift_feature_list_train.append(feature)
        dense_feature_list.append(feature)
    #vocab = np.loadtxt("vocab1010200.txt")
    vocab = build_visual_dictionary(dense_feature_list, dic_size)
    
    bow_feature_train = []
    bow_feature_test = []

    for i in range(len(sift_feature_list_train)):
        feature = compute_bow(sift_feature_list_train[i], vocab)
        bow_feature_train.append(feature)
    

    for img in img_test_list:
        image = cv2.imread(img, 0)
        feature = compute_dsift(image, 10, 10)
        test_feature = compute_bow(feature, vocab)
        bow_feature_test.append(test_feature)

    
    bow_feature_train = np.asarray(bow_feature_train)
    bow_feature_test = np.asarray(bow_feature_test)
    
    k = 7
    label_test_pred = predict_knn(bow_feature_train, label_train_list, bow_feature_test, k)
    
    confusion = build_confusion_matrix(label_test_list, label_test_pred)
    diagonals = confusion.diagonal()
    accuracy = (np.sum(diagonals)/len(label_test_list))*100
    
    visualize_confusion_matrix(confusion, accuracy, label_classes)
    return confusion, accuracy


def predict_svm(feature_train, label_train, feature_test, n_classes):
    #get all the unique label values 
    uni = list(set(label_train))
    print(uni)
    #score list for test for each model
    scores = []
    #build one vs all svc for each class 
    for i in uni : 
        label_curr = []
        #build label 0 vs 1 for curr label value 
        for j in label_train : 
            if j == i : 
                label_curr.append(1)
            else :
                label_curr.append(-1)
        #build svc for curr label value 
        curr_model = LinearSVC(C=0.5, max_iter=20000)
        curr_model.fit(feature_train, label_curr) 
        score_curr = curr_model.decision_function(feature_test)
        scores.append(np.asarray(score_curr).transpose())
        
    scores = np.asarray(scores).transpose()
    label_test_pred = []
    
    for i in range(0,scores.shape[0]):
        max_score = np.amax(scores[i])
       
        max_index = np.where(scores[i] == max_score)
        max_index = max_index[0][0]
        
        pred_class = uni[max_index]
        label_test_pred.append(pred_class)
        
    return label_test_pred


def classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    
    dense_feature_list = []
    for i in img_train_list:
        img = cv2.imread(i, 0)
        dense_features = compute_dsift(img, 10, 10)
        dense_feature_list.append(dense_features)
    
    dic_size = 200
    vocab = build_visual_dictionary(dense_feature_list, dic_size)
    #np.savetxt("vocab1616_50.txt", vocab)
    bow_feature_train = []
    bow_feature_test = []
    #vocab = np.loadtxt("vocab1010200.txt")

    for i in range(len(dense_feature_list)):
        feature = compute_bow(dense_feature_list[i], vocab)
        bow_feature_train.append(feature)
    
    for img in img_test_list:
        image = cv2.imread(img, 0)
        feature = compute_dsift(image, 10, 10)
        test_feature = compute_bow(feature, vocab)
        bow_feature_test.append(test_feature)

    bow_feature_train = np.asarray(bow_feature_train)
    bow_feature_test = np.asarray(bow_feature_test)

    label_test_pred = predict_svm(bow_feature_train, label_train_list, bow_feature_test, 15)
    
    confusion = build_confusion_matrix(label_test_list, label_test_pred)
    diagonals = confusion.diagonal()
    accuracy = (np.sum(diagonals)/len(label_test_list))*100
    print(accuracy)
    
    visualize_confusion_matrix(confusion, accuracy, label_classes)
    return confusion, accuracy


def visualize_confusion_matrix(confusion, accuracy, label_classes):
    plt.title("accuracy = {:.3f}".format(accuracy))
    plt.imshow(confusion)
    ax, fig = plt.gca(), plt.gcf()
    plt.xticks(np.arange(len(label_classes)), label_classes)
    plt.yticks(np.arange(len(label_classes)), label_classes)
    # set horizontal alignment mode (left, right or center) and rotation mode(anchor or default)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="center", rotation_mode="default")
    # avoid top and bottom part of heatmap been cut
    ax.set_xticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    plt.show()
    
def build_confusion_matrix(label_test_list, label_test_pred):
    confusion_matrix = np.zeros((15,15))
    for i in range(len(label_test_pred)):
        xIndex = getIndex(label_test_list[i])
        yIndex = getIndex(label_test_pred[i])
        confusion_matrix[xIndex, yIndex] = confusion_matrix[xIndex, yIndex] + 1

    return confusion_matrix

def getIndex(label):
    if(label == "Kitchen"):
        return 0
    if(label == "Store"):
        return 1
    if(label == "Bedroom"):
        return 2
    if(label == "LivingRoom"):
        return 3
    if(label == "Office"):
        return 4
    if(label == "Industrial"):
        return 5
    if(label == "Suburb"):
        return 6
    if(label == "InsideCity"):
        return 7
    if(label == "TallBuilding"):
        return 8
    if(label == "Street"):
        return 9
    if(label == "Highway"):
        return 10
    if(label == "OpenCountry"):
        return 11
    if(label == "Coast"):
        return 12
    if(label == "Mountain"):
        return 13
    if(label == "Forest"):
        return 14


if __name__ == '__main__':
    # To do: replace with your dataset path
    label_classes, label_train_list, img_train_list, label_test_list, img_test_list = extract_dataset_info("./scene_classification_data")
    
    classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)

    classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)
    
    classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)






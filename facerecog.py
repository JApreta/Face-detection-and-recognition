import face_recognition
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import pandas as pd
from tabulate import tabulate


def encode_known_faces(known_faces_dir):
    known_face_encodings = []
    known_face_names = []

    for person_name in os.listdir(known_faces_dir):
        person_dir = os.path.join(known_faces_dir, person_name)
        if os.path.isdir(person_dir):
            for image_file in os.listdir(person_dir):
                image = face_recognition.load_image_file(os.path.join(person_dir, image_file))
                face_encodings = face_recognition.face_encodings(image)

                # Check if a face encoding is found
                if len(face_encodings) > 0:
                    face_encoding = face_encodings[0]
                    known_face_encodings.append(face_encoding)
                    known_face_names.append(person_name)
                    print(f"Encoded face: {person_name}")  # Debug line to print the encoded face name
                else:
                    print(f"Warning: No face detected in {image_file}")  # Debug line to print a warning message

    return known_face_encodings, known_face_names

def save_known_faces(known_face_encodings, known_face_names, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump((known_face_encodings, known_face_names), f)

def load_known_faces(file_path):
    with open(file_path, 'rb') as f:
        known_face_encodings, known_face_names = pickle.load(f)
    return known_face_encodings, known_face_names


def preprocess_image(image):
    target_size = (600, 600)
    resized_image = cv2.resize(image, target_size)
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    equalized_image = cv2.equalizeHist(gray_image)

    return equalized_image

def recognize_faces(image_path, known_face_encodings, known_face_names):
    # Load the image
    unknown_image = face_recognition.load_image_file(image_path)
    original_height, original_width, _ = unknown_image.shape

    # Apply preprocessing
    preprocessed_image = preprocess_image(unknown_image)

    # Convert the preprocessed image back to RGB for face recognition
    rgb_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_GRAY2RGB)

    # Get face encodings and locations
    unknown_face_encodings = face_recognition.face_encodings(rgb_image)
    face_locations = face_recognition.face_locations(rgb_image)

    face_names = []

    for face_encoding, face_location in zip(unknown_face_encodings, face_locations):
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)
        print(f"Recognized: {name}")

    # Scale the face locations back to the original image dimensions
    scaled_face_locations = []
    for (top, right, bottom, left) in face_locations:
        top = int(top * original_height / 600)
        right = int(right * original_width / 600)
        bottom = int(bottom * original_height / 600)
        left = int(left * original_width / 600)
        scaled_face_locations.append((top, right, bottom, left))

    return scaled_face_locations, face_names


def display_image(image_path, face_locations, face_names):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1)
    ax.imshow(image_rgb)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        rect = patches.Rectangle((left, top), right - left, bottom - top, linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(rect)

        textbox = plt.Rectangle((left, top - 20), right - left, 20, facecolor='g', edgecolor='g')
        ax.add_patch(textbox)
        plt.text(left + 3, top - 5, name, fontsize=12, color='white')

    plt.show()


# the code below is meant to run only once to train the model
def training():
    known_faces_dir = "dataset/known_faces/"
    encodings_file_path = "./known_face_encodings.pkl"
    known_face_encodings, known_face_names = encode_known_faces(known_faces_dir)
    save_known_faces(known_face_encodings, known_face_names, encodings_file_path)


def run_test_set(test_folder):
    # Get a list of test images
    test_images = sorted([f for f in os.listdir(test_folder) if os.path.isfile(os.path.join(test_folder, f))])

    # Initialize a list to store the predicted labels
    predicted_labels = []
    encodings_file_path = "./known_face_encodings.pkl"
    known_face_encodings, known_face_names = load_known_faces(encodings_file_path)
    # Loop through the test images
    for image_name in test_images:
        # Read the image
        image_path = os.path.join(test_folder, image_name)
        image = cv2.imread(image_path)

        # Perform face recognition
        face_locations, predicted_label = recognize_faces(image_path, known_face_encodings, known_face_names)
        display_image(image_path, face_locations, predicted_label)
        # Add the predicted label to the list
        predicted_labels.append(predicted_label)

    return predicted_labels

def run_single_test(image_path):
    encodings_file_path = "./known_face_encodings.pkl"
    known_face_encodings, known_face_names = load_known_faces(encodings_file_path)
    face_locations, face_names = recognize_faces(image_path, known_face_encodings, known_face_names)
    display_image(image_path, face_locations, face_names)

def compute_metrics(true_labels, predicted_labels):
    # Get unique individuals in the dataset
    individuals = sorted(list(set(true_labels)))

    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=individuals)

    # Compute classification report (precision, recall, F1-score)
    report = classification_report(true_labels, predicted_labels, target_names=individuals, output_dict=True)

    # Calculate TPR, FPR, and FNR
    tpr = []
    fpr = []
    fnr = []
    for i, individual in enumerate(individuals):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        tn = np.sum(cm) - tp - fp - fn

        tpr.append(tp / (tp + fn))
        fpr.append(fp / (fp + tn))
        fnr.append(fn / (fn + tp))

    metrics = {
        "tpr": np.mean(tpr),
        "fpr": np.mean(fpr),
        "fnr": np.mean(fnr),
        "classification_report": report
    }
    # Print the confusion matrix with labels
    classes = sorted(set(true_labels + predicted_labels))
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    print("Confusion Matrix:")
    print(tabulate(cm_df, headers='keys', tablefmt='psql'))
    return metrics

def performance_metrics():
    test_folder = "dataset/test/"
    predicted_labels = run_test_set(test_folder)

    true_labels=[['Unknown'], ['judith'], ['cleidia'], ['judith'], ['judith'],['cleidia', 'Unknown', 'judith'], ['cleidia', 'Unknown', 'judith', 'Unknown'], ['Unknown', 'cleidia', 'judith'], ['cleidia', 'Unknown', 'Unknown', 'judith'], ['judith', 'cleidia'], ['cleidia'], ['cleidia'], ['cleidia'], ['cleidia']]
    flat_true_labels = [label for sublist in true_labels for label in (sublist if sublist else ['Unknown'])]
    flat_predicted_labels= [label for sublist in predicted_labels for label in (sublist if sublist else ['Unknown'])]
    #print(flat_predicted_labels)
    
    metrics = compute_metrics(flat_true_labels, flat_predicted_labels)
    #print(metrics)
    # Extract the classification report dictionary
    classification_report_dict = metrics['classification_report']

    # Create a DataFrame from the classification report dictionary
    report_df = pd.DataFrame(classification_report_dict).transpose()

    # Print the metrics
    print(f"True Positive Rate: {metrics['tpr']}")
    print(f"False Positive Rate: {metrics['fpr']}")
    print(f"False Negative Rate: {metrics['fnr']}")

    # Print the classification report in a tabular format
    print("\nClassification Report:")
    print(tabulate(report_df, headers='keys', tablefmt='psql'))
    #print(metrics)

    # Convert the labels to binary format
    classes = sorted(set(flat_true_labels + flat_predicted_labels))

    true_labels_binary = label_binarize(flat_true_labels, classes=classes)
    predicted_labels_binary = label_binarize(flat_predicted_labels, classes=classes)

    n_classes = len(classes)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(true_labels_binary[:, i], predicted_labels_binary[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve for each class
    plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f"ROC curve of class {classes[i]} (AUC = {roc_auc[i]:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.show()


#training() # this line of code should be run only once traing the model in case the known_faces set is updated

#to run one test image at once
#image_path="dataset/test/test_0.jpg"
#run_single_test(image_path)


#to run all test set and print the performance metrics
performance_metrics()

#to run the test set without metrics
#test_folder = "dataset/test/"
#run_test_set(test_folder)

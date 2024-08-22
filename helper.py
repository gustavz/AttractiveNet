import os
import zipfile
import shutil
import gdown
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt


DATA_DIR = 'data/'
DOCS_DIR = 'docs/'
LABELS_FILE = 'All_labels.txt'
DATA_URL = 'https://drive.google.com/uc?id=1w0TorBfTIqbquQVd6k3h_77ypnrvfGwf'
ZFILE = 'SCUT-FBP5500_v2.1.zip'

def extract_zipfile():
    with zipfile.ZipFile(ZFILE) as zip_file:
        for member in zip_file.namelist():
            filename = os.path.basename(member)
            # skip directories, all non-jpgs, except labels
            if filename.endswith(".jpg") or filename == LABELS_FILE:
                # copy file (taken from zipfile's extract)
                source = zip_file.open(member)
                target = open(os.path.join(DATA_DIR, filename), "wb")
                with source, target:
                    shutil.copyfileobj(source, target)

def download_data():
    # Download Dataset
    if os.path.isfile(ZFILE) or os.path.isfile(DATA_DIR+LABELS_FILE):
        print('data already downloaded')
    else:
        print ("data does not exist. downloading it.")
        gdown.download(DATA_URL, ZFILE, quiet=False)
    # Extract ZipFile
    if os.path.isfile(DATA_DIR+LABELS_FILE):
        print("data already extracted.")
    else:
        print("extracting data.")
        if not os.path.exists(DATA_DIR):
            os.mkdir(DATA_DIR)
        extract_zipfile()
        os.remove(ZFILE)
        
def resize_with_pad(image, new_shape, padding_color = (255, 255, 255)) -> np.array:
    original_shape = (image.shape[1], image.shape[0])
    ratio = float(max(new_shape)) / max(original_shape)
    new_size = tuple([int(x * ratio) for x in original_shape])
    image = cv2.resize(image, new_size)
    delta_w = new_shape[0] - new_size[0]
    delta_h = new_shape[1] - new_size[1]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)

    return image

def preprocess_image(image,target_size):
    return resize_with_pad(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),target_size) / .255

def create_dataset(target_size):
    X = []
    y = []
    labels_dict = get_labels_dict()
    img_files = glob.glob(DATA_DIR+'*.jpg')
    print(f'reading {len(img_files)} images into dataset')
    for f in img_files:
        img = preprocess_image(cv2.imread(f), target_size)
        X.append(img)
        y.append(labels_dict[os.path.split(f)[-1]])
    return np.array(X), np.array(y)

def get_labels_dict():
    labels_dict = {}
    with open(DATA_DIR + LABELS_FILE) as fp:
        for line in fp:
            img,label = line.split(' ', 1)
            labels_dict[img] = float(label)
    return labels_dict

def plot_metrics(history, model_name, stage):
    f,(ax1) = plt.subplots(1, 1, figsize=(15,7))
    f.suptitle(f'Stage {stage} Model "{model_name}" training Metrics')
    ax1.plot(history.history["loss"], color='darkblue', label="Train")
    ax1.plot(history.history["val_loss"], color='darkred', label="Test")
    ax1.set_title('Loss (Mean Squared Error) over epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.legend()
    plt.savefig(f'{DOCS_DIR}metrics_stage_{stage}_{model_name}')
    plt.show()


if __name__ == "__main__":
    download_data()
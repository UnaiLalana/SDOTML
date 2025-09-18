import pandas as pd
import sys
import os
import shutil
import zipfile
import cv2
import numpy as np


def load_data_from_zip(output_dir):
    with zipfile.ZipFile(output_dir+'.zip', 'r') as zipf:
            zipf.extractall(output_dir)

def read_dataset(dataset_dir):

    if not os.path.exists(dataset_dir):
        print("No existe el directorio con las imagenes:", dataset_dir)
        load_data_from_zip(dataset_dir)
    

    df = pd.read_csv(os.path.join(dataset_dir + ".csv"))
    X = []
    y = []
    for _, row in df.iterrows():
        img = os.path.join(dataset_dir, row['file_name'].replace('train_data/', ''))
        if img is not None:
            X.append(img)
            y.append(row['label'])
    X = np.array(X)
    y = np.array(y)
    return X, y


if __name__ == "__main__":

    output_dir = "data/interim/initial_data"
    if os.path.exists(output_dir + ".zip"):
        load_data_from_zip(output_dir)

    else:
        data = pd.read_csv("data/raw/train.csv")

        initial_data = data[:1000]
        initial_data.to_csv("data/interim/initial_data.csv", index=False)

        for picture in initial_data['file_name']:
            output_dir = "data/interim/initial_data"
            os.makedirs(output_dir, exist_ok=True)
            src_path = os.path.join("data/raw/", picture)
            dst_path = os.path.join(output_dir, picture.replace('train_data/', ''))
            shutil.copy2(src_path, dst_path)

        zip_path = output_dir + ".zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, output_dir)
                    zipf.write(file_path, arcname)





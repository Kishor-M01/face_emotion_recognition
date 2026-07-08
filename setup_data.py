"""
Downloads FER2013 from Kaggle and organises it into:
  data/fer2013/train/<emotion>/
  data/fer2013/test/<emotion>/

Requirements:
  1. pip install kaggle
  2. Place kaggle.json in ~/.kaggle/  (get it from kaggle.com > Account > API)
"""
import os, zipfile, shutil, csv

KAGGLE_DATASET = "msambare/fer2013"
ZIP_NAME       = "fer2013.zip"
DATA_DIR       = "data/fer2013"
EMOTIONS       = ['angry','disgust','fear','happy','neutral','sad','surprise']

def download():
    os.makedirs("data", exist_ok=True)
    os.system(f"kaggle datasets download -d {KAGGLE_DATASET} -p data --unzip")

def check_already_organised():
    return os.path.isdir(os.path.join(DATA_DIR, "train", "happy"))

if __name__ == "__main__":
    if check_already_organised():
        print("Dataset already organised.")
    else:
        print("Downloading FER2013 …")
        download()
        print("Done. Dataset ready at", DATA_DIR)

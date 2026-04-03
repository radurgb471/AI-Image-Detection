import cv2
import os
import csv
import numpy as np
from forensic_utils_v2 import extract_all_features

def process_folder_v2(folder_path, label, csv_writer):

    print(f"Processing folder: {folder_path}")

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]

    count = 0
    for filename in files:
        full_path = os.path.join(folder_path, filename)

        img = cv2.imread(full_path)
        if img is None:
            continue
        #Jumping over corrupted files
        try:

            features = extract_all_features(img)

            csv_writer.writerow(features + [label])

            count += 1
            if count % 100 == 0:
                print(f"Progress: {count} processed images...")

        except Exception as e:
            print(f"Error with file: {filename}: {e}")
            continue

    print(f"Finalised! Processed {count} images from this folder.\n")


def main():

    path_real = r'R:\archive\real'
    path_fake = r'R:\archive\fakeV2\fake-v2'
    output_csv = 'date_antrenare_v2.csv'

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(['var_x', 'var_y', 'cov_xy', 'anisotropy', 'fft_score', 'is_fake'])

        process_folder_v2(path_real, 0, writer)  # 0 = REAL
        process_folder_v2(path_fake, 1, writer)  # 1 = FAKE (AI)

    print(f"'{output_csv}' file is ready for training V2.")


if __name__ == "__main__":
    main()

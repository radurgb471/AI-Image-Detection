import cv2
import os
import csv
import numpy as np
from forensicUtils import calculate_luminance, get_statistical_features


def process_folder(folder_path, label, csv_writer):
    print("Processing folder {}".format(folder_path))
    #label: 0 for REAL, 1 for FAKE

    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    count = 0

    for filename in files:
        full_path = os.path.join(folder_path, filename)
        img = cv2.imread(full_path)
        if img is None:
            continue

        gray = calculate_luminance(img)
        gray_norm = (gray / 255.0).astype(np.float32)

        gx = cv2.Sobel(gray_norm, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_norm, cv2.CV_32F, 0, 1, ksize=3)

        C, anisotropy = get_statistical_features(gx, gy)

        csv_writer.writerow([C[0,0], C[1,1], C[0,1], anisotropy, label])

        count += 1
        if count % 100 == 0:
            print('Processed {} images'.format(count))

def main():
    path_real = r'R:\archive\real'
    path_fake = r'R:\archive\fakeV2\fake-v2'
    output_csv = 'training_data.csv'

    with open(output_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['var_x', 'var_y', 'cov_xy', 'anisotropy', 'is_fake'])
        process_folder(path_real, 0, writer)
        process_folder(path_fake, 1, writer)

    print(f"\nGATA! Tabelul a fost salvat în: {output_csv}")

if __name__ == "__main__":
    main()
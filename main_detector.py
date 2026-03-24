import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from Versiunea1_SobelManual.forensicUtils import calculate_luminance, get_statistical_features

def main():
    image_path = '../Versiunea3_SobelFFTPrimele200/V3_1_ELA/F35real.jpg'
    if not os.path.exists(image_path):
        print('Image not found')
        return

    img_bgr = cv2.imread(image_path)
    print('Se calc luminanta...')
    gray = calculate_luminance(img_bgr)
    gray_norm = (gray / 255.0).astype(np.float32)
    ##data type mismatch, gray / 255 made a float64 matrix

    # gx, gy = sobel_manual(gray_norm)

    gx = cv2.Sobel(gray_norm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_norm, cv2.CV_32F, 0, 1, ksize=3)

    C, anisotropy = get_statistical_features(gx, gy)

    print("\n--- RAPORT CRIMINALISTIC ---")
    print(f"Matricea de Covarianță (C):\n{C}")
    print(f"Scor Anizotropie: {anisotropy:.6f}")

    if anisotropy < 0.1:
        print("Sfat: Textura pare foarte uniformă (posibil indicator AI).")
    else:
        print("Sfat: Textura are o direcție clară (specific obiectelor reale).")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 3, 1)

    # OpenCV uses BGR, but Matplotlib wants RGB for display

    plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)

    grad_mag = (gx ** 2 + gy ** 2) ** 0.5
    plt.imshow(grad_mag, cmap='gray')
    plt.title("Gradient Map (Sobel)")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.text(0.1, 0.5, f"Covariance C:\n{C}\n\nAnisotropy:\n{anisotropy:.4f}",
             fontsize=12, fontweight='bold')
    plt.axis('off')
    plt.title("Mathematical Analysis")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
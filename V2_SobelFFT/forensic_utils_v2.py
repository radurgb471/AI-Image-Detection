import numpy as np
import cv2

def calculate_luminance(img_bgr):

    b, g, r = img_bgr[:, :, 0], img_bgr[:, :, 1], img_bgr[:, :, 2]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def get_statistical_features(gx, gy):
    gx_flat = gx.flatten()
    gy_flat = gy.flatten()
    M = np.stack((gx_flat, gy_flat), axis=1)
    C = (M.T @ M) / M.shape[0]

    eigenvalues = np.linalg.eigvals(C)
    L1, L2 = np.max(eigenvalues), np.min(eigenvalues)
    anisotropy = (L1 - L2) / (L1 + L2 + 1e-6)

    return [C[0,0], C[1,1], C[0,1], anisotropy]

def get_fft_feature(gray_img):
    f = np.fft.fft2(gray_img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-6)

    h, w = magnitude_spectrum.shape
    cy, cx = h // 2, w // 2
    r = 20
    magnitude_spectrum[cy - r:cy + r, cx - r:cx + r] = 0

    spectral_score = np.max(magnitude_spectrum)
    return spectral_score


def extract_all_features(img_bgr):
    gray = calculate_luminance(img_bgr)
    gray_norm = (gray / 255.0).astype(np.float32)

    # Gradienți (Sobel)
    gx = cv2.Sobel(gray_norm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_norm, cv2.CV_32F, 0, 1, ksize=3)

    stats = get_statistical_features(gx, gy)
    fft_score = get_fft_feature(gray_norm)

    return stats + [fft_score]
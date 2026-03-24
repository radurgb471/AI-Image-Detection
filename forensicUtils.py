import numpy as np

def calculate_luminance(img_bgr):
    b = img_bgr[:, :, 0]
    g = img_bgr[:, :, 1]
    r = img_bgr[:, :, 2]

    L = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return L

def sobel_manual(gray_img):
    h, w = gray_img.shape
    gx = np.zeros_like(gray_img)
    gy = np.zeros_like(gray_img)

    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])

    Ky = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]])

    for i in range (1, h - 1):
        for j in range (1, w - 1):
            regiune = gray_img[i - 1:i + 2, j - 1:j + 2]
            gx[i, j] = np.sum(Kx*regiune)
            gy[i, j] = np.sum(Ky*regiune)

    return gx, gy

def get_statistical_features(gx, gy):
    gx_flat = gx.flatten()
    gy_flat = gy.flatten()

    M = np.stack((gx_flat, gy_flat), axis=1)
    N = M.shape[0]

    C = (M.T @ M) / N

    eigenvalues = np.linalg.eigvals(C)
    L1 = np.max(eigenvalues)
    L2 = np.min(eigenvalues)

    anisotropy = (L1 - L2) / (L1 + L2 + 1e-6)

    return C, anisotropy

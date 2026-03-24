import cv2
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from V2_SobelFFT.forensic_utils_v2 import extract_all_features


def predict_with_dashboard(image_path, model_path='detector_ai_v2.pkl'):

    model = joblib.load(model_path)
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    #Features Extraction
    features = extract_all_features(img_bgr)

    feature_names = ['var_x', 'var_y', 'cov_xy', 'anisotropy', 'fft_score']
    df_single = pd.DataFrame([features], columns=feature_names)

    #Prediction

    prediction = model.predict(df_single)[0]
    probs = model.predict_proba(df_single)[0]
    confidence = probs[prediction] * 100
    label = "AI / FAKE" if prediction == 1 else "REALĂ"

    #Visualizing FFT Spectrum

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) / 255.0
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-6)

    plt.figure(figsize=(14, 6))

    #Original Image

    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    color = 'red' if prediction == 1 else 'green'
    plt.title(f"VERDICT: {label} ({confidence:.2f}%)", color=color, fontsize=15, fontweight='bold')
    plt.axis('off')

    # Frequency Spectrum
    plt.subplot(1, 2, 2)
    plt.imshow(magnitude_spectrum, cmap='magma')
    plt.title(f"Spectral (FFT Score: {features[4]:.2f})")
    plt.colorbar(label='Energie')

    print(f"Analysis complete for {image_path}. Opening dashboard...")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    predict_with_dashboard('../Versiunea3_SobelFFTPrimele200/V3_1_ELA/F35Fals.png')
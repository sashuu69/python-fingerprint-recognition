import cv2
import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from enhance import image_enhance
from skimage.morphology import skeletonize
import numpy as np

def removedot(invertThin):
    # Previous implementation remains the same
    temp0 = numpy.array(invertThin[:])
    temp1 = temp0 / 255
    temp2 = numpy.array(temp1)

    filtersize = 6
    W, H = temp0.shape[:2]

    for i in range(W - filtersize):
        for j in range(H - filtersize):
            filter0 = temp1[i:i + filtersize, j:j + filtersize]

            flag = 0
            if sum(filter0[:, 0]) == 0:
                flag += 1
            if sum(filter0[:, filtersize - 1]) == 0:
                flag += 1
            if sum(filter0[0, :]) == 0:
                flag += 1
            if sum(filter0[filtersize - 1, :]) == 0:
                flag += 1
            if flag > 3:
                temp2[i:i + filtersize, j:j + filtersize] = numpy.zeros((filtersize, filtersize))

    return temp2

def get_descriptors(img):
    # Previous implementation remains the same
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = image_enhance.image_enhance(img)
    img = numpy.array(img, dtype=numpy.uint8)

    # Threshold
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    img[img == 255] = 1

    # Thinning
    skeleton = skeletonize(img)
    skeleton = numpy.array(skeleton, dtype=numpy.uint8)
    skeleton = removedot(skeleton)

    # Harris corners
    harris_corners = cv2.cornerHarris(img, 3, 3, 0.04)
    harris_normalized = cv2.normalize(harris_corners, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
    threshold_harris = 125

    keypoints = []
    for x in range(harris_normalized.shape[0]):
        for y in range(harris_normalized.shape[1]):
            if harris_normalized[x, y] > threshold_harris:
                keypoints.append(cv2.KeyPoint(y, x, 1))

    orb = cv2.ORB_create()
    _, des = orb.compute(img, keypoints)
    return keypoints, des

def verify_fingerprint(img1_path, img2_path, score_threshold=33):
    # Previous implementation remains the same
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    kp1, des1 = get_descriptors(img1)
    kp2, des2 = get_descriptors(img2)

    if des1 is None or des2 is None:
        print("Descriptors could not be extracted. Ensure the images are of sufficient quality.")
        return False, None, None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda match: match.distance)

    score = sum(match.distance for match in matches) / len(matches)
    is_match = score < score_threshold
    return is_match, score, matches

def calculate_biometric_metrics(y_true, y_pred, scores):
    """
    Calculate biometric-specific performance metrics
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total = len(y_true)
    
    # Basic Metrics
    far = fp / (fp + tn)  # False Accept Rate
    frr = fn / (fn + tp)  # False Reject Rate
    gar = tp / (tp + fn)  # Genuine Accept Rate (True Positive Rate)
    fmr = fp / (fp + tn)  # False Match Rate
    fnmr = fn / (fn + tp) # False Non-Match Rate
    
    # Advanced Metrics
    eer = (far + frr) / 2  # Equal Error Rate (simplified approximation)
    fta = numpy.sum(numpy.isnan(scores)) / len(scores)  # Failure to Acquire
    
    # Additional Metrics
    tpr = tp / (tp + fn)  # True Positive Rate
    tnr = tn / (tn + fp)  # True Negative Rate
    
    return {
        'FAR': far,
        'FRR': frr,
        'GAR': gar,
        'FMR': fmr,
        'FNMR': fnmr,
        'EER': eer,
        'FTA': fta,
        'TPR': tpr,
        'TNR': tnr
    }

def plot_biometric_metrics(metrics):
    """
    Create visualizations for biometric metrics
    """
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Basic Metrics
    plt.subplot(2, 2, 1)
    basic_metrics = ['FAR', 'FRR', 'GAR']
    values = [metrics[m] for m in basic_metrics]
    plt.bar(basic_metrics, values)
    plt.title('Basic Biometric Metrics')
    plt.ylabel('Rate')
    
    # Plot 2: ROC-like visualization
    plt.subplot(2, 2, 2)
    plt.plot([0, 1], [0, 1], 'r--')  # Diagonal line
    plt.plot([0, metrics['FAR']], [metrics['GAR'], metrics['GAR']], 'b-', label='Operating Point')
    plt.scatter(metrics['FAR'], metrics['GAR'], color='blue', s=100)
    plt.title('ROC Operating Point')
    plt.xlabel('False Accept Rate (FAR)')
    plt.ylabel('Genuine Accept Rate (GAR)')
    plt.legend()
    
    # Plot 3: Advanced Metrics
    plt.subplot(2, 2, 3)
    adv_metrics = ['FMR', 'FNMR', 'EER', 'FTA']
    values = [metrics[m] for m in adv_metrics]
    plt.bar(adv_metrics, values)
    plt.title('Advanced Biometric Metrics')
    plt.ylabel('Rate')
    
    # Plot 4: System Performance
    plt.subplot(2, 2, 4)
    labels = ['Genuine Accept', 'False Reject', 'False Accept', 'True Reject']
    sizes = [metrics['GAR'], metrics['FRR'], metrics['FAR'], metrics['TNR']]
    plt.pie(sizes, labels=labels, autopct='%1.1f%%')
    plt.title('System Performance Distribution')
    
    plt.tight_layout()
    return plt.gcf()

def main():
    # Input paths for two images
    img1_path = r"C:\Users\lenovo\python-fingerprint-recognition\database\1cenhanced.jpg"
    img2_path = r"C:\Users\lenovo\python-fingerprint-recognition\database\final_enhanced.jpg"

    # Get verification results
    is_match, score, matches = verify_fingerprint(img1_path, img2_path)

    if score is not None:
        print(f"Matching Score: {score:.2f}")
        print(f"Result: The fingerprints {'match' if is_match else 'do not match'}.")

        # Display matches visualization
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        kp1, _ = get_descriptors(img1)
        kp2, _ = get_descriptors(img2)
        img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, flags=2, outImg=None)
        plt.figure(figsize=(12, 4))
        plt.imshow(img_matches)
        plt.title("Matching Keypoints")
        plt.show()

        # Simulate a batch of results for metrics calculation
        # In a real system, you would have actual test results
        y_true = [1, 1, 1, 0, 0, 0, 1, 0]  # Ground truth
        y_pred = [1, 1, 0, 0, 1, 0, 1, 0]  # Predictions
        scores = [score] * len(y_true)      # Simulated scores

        # Calculate and display biometric metrics
        metrics = calculate_biometric_metrics(y_true, y_pred, scores)
        
        print("\nBiometric Performance Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        # Plot metrics
        fig = plot_biometric_metrics(metrics)
        plt.show()
        
    else:
        print("Unable to compute matching score due to descriptor issues.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        raise
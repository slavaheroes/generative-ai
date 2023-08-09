import numpy as np

def get_calibration_bucket(corrects, confidences, n_bins=20):
    thresholds = np.linspace(0, 1, n_bins+1)
    buckets = [(thresholds[i], thresholds[i + 1]) for i in range(len(thresholds) - 1)]
    
    bucket_acc = []
    
    for bucket in buckets:
        total = 0
        correct = 0
        for i in range(len(confidences)):
            if confidences[i] > bucket[0] and confidences[i] < bucket[1]:
                correct += corrects[i]
                total += 1
        
        if total!=0:
            bucket_acc.append(correct/total)
        else:
            bucket_acc.append(0.0)
    
    return bucket_acc
    
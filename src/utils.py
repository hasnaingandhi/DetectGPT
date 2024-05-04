from sklearn.metrics import roc_curve, precision_recall_curve, auc


def trim_to_shortest(text1, text2):
    shorter = min(len(text1.split(' ')), len(text2.split(' ')))
    text1 = " ".join(text1.split(' ')[:shorter])
    text2 = " ".join(text2.split(' ')[:shorter])
    return text1, text2

def get_roc_metrics(real_preds, sample_preds):
    fpr, tpr, _ = roc_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
    roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), float(roc_auc)
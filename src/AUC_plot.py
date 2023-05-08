import pandas as pd
# import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Load the data from the xlsx file
df = pd.read_excel("/opt/project/dataset/result_predictions_classification_all.xlsx")
df2 = pd.read_excel("/opt/project/dataset/result_resnet50.xlsx")
df3 = pd.read_excel("/opt/project/dataset/result_resnet50scratch_all.xlsx")

# Compute the AUC score
auc_score = roc_auc_score(df["Label"], df["Prediction"])
auc_score2 = roc_auc_score(df2["Label"], df2["Prediction"])
auc_score3 = roc_auc_score(df3["Label"], df3["Prediction"])

# Compute the false positive rate and true positive rate
fpr, tpr, thresholds = roc_curve(df["Label"], df["Prediction"])
fpr2, tpr2, threshold2s = roc_curve(df2["Label"], df2["Prediction"])
fpr3, tpr3, threshold3s = roc_curve(df3["Label"], df3["Prediction"])

# Plot the ROC curve
plt.plot(fpr, tpr, label="AUC Our = %.2f" % auc_score)
plt.plot(fpr2, tpr2, label="AUC ResNet50 from fine tune = %.2f" % auc_score2)
plt.plot(fpr3, tpr3, label="AUC ResNet50 from scratch = %.2f" % auc_score3)
plt.plot([0, 1], [0, 1], "k--")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend(loc="lower right")
plt.savefig('/opt/project/tmp/AUC_plot.jpg')
plt.show()

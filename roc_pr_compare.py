from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import os
'''
2021-01-19 offline evaluation results
'''

#load result
output_path = './roc_pr_compare_results/'
gt_db_path ='./sg_test_output/04_gt_db.npy'
pred_db_path = './sg_test_output/04_DL_db.npy'
gt_db=np.load(gt_db_path)
pred_db=np.load(pred_db_path)


pred_db = np.array(pred_db)
gt_db = np.array(gt_db)
#####ROC
fpr, tpr, roc_thresholds = metrics.roc_curve(gt_db, pred_db)
roc_auc = metrics.auc(fpr, tpr)
# print("fpr: ", fpr)
# print("tpr: ", tpr)
print("thresholds: ", roc_thresholds)
print("roc_auc: ", roc_auc)


my_gt_db_path ='./cjf_test_output/04_gt_db.npy'
my_pred_db_path = './cjf_test_output/04_DL_db.npy'
my_gt_db=np.load(my_gt_db_path)
my_pred_db=np.load(my_pred_db_path)


my_pred_db = np.array(my_pred_db)
my_gt_db = np.array(my_gt_db)
#####ROC
my_fpr, my_tpr, my_roc_thresholds = metrics.roc_curve(my_gt_db, my_pred_db)
my_roc_auc = metrics.auc(my_fpr, my_tpr)
# print("fpr: ", fpr)
# print("tpr: ", tpr)
print("thresholds: ", my_roc_thresholds)
print("roc_auc: ", my_roc_auc)
#  plot&save ROC
plt.figure(0)
lw = 2
ln1,=plt.plot(fpr, tpr, color='red',
            lw=lw, label=' author ROC curve (area = %0.2f)' % roc_auc )
ln2,=plt.plot(my_fpr, my_tpr, color='darkorange',
            lw=lw, label='ours ROC curve (area = %0.2f)' % my_roc_auc )
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('KITTI 04 ROC Curve')
plt.legend(loc="lower right")
# plt.show()
roc_out = os.path.join(output_path + "04_DL_roc_curve.png")
print(roc_out)
plt.savefig(roc_out)

#### P-R
precision, recall, pr_thresholds = metrics.precision_recall_curve(gt_db, pred_db)
#pr_auc = metrics.auc(recall, precision)
my_precision, my_recall, my_pr_thresholds = metrics.precision_recall_curve(my_gt_db, my_pred_db)
#my_pr_auc = metrics.auc(my_recall, my_precision)
# plot p-r curve
plt.figure(1)
lw = 2
ln3,=plt.plot(recall, precision, color='red',
            lw=lw, label=' author PR curve')
ln4,=plt.plot(my_recall, my_precision, color='darkorange',
            lw=lw, label='ours PR curve')
plt.axis([0,1,0,1])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('KITTI 04 Precision-Recall Curve')
plt.legend(loc="lower right")
pr_out = os.path.join(output_path + "04_DL_pr_curve.png")
plt.savefig(pr_out)
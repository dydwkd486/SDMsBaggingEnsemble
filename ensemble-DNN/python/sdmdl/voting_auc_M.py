#-*- encoding: utf8 -*-
import os
import pandas as pd
import numpy as np
from sklearn.metrics.ranking import roc_auc_score, roc_curve
from sklearn.metrics import recall_score,multilabel_confusion_matrix
from sklearn.metrics import cohen_kappa_score

# Hynobius_leechii_total_env_dataframe_test
# Cyanopica_cyanus_total_env_dataframe_test
# Platalea_minor_total_env_dataframe_test
# Hypsipetes_amaurotis_total_env_dataframe_test
# Hyla_japonica_total_env_dataframe_test
path_dir='F:/github/bird/result/rsep/청개구리/DMLP'
folder= os.listdir(path_dir)
print(folder)
# [0.85259631]
# [0.99039201]
# [0.87871489]
# [0.84298832]
for j in range(0, 5, 1):
    Sorensen =[]
    OPR = []
    UPR = []
    Sensitivity = []
    specificity = []
    KAPPA = []
    TSS = []
    for count in range(0, 1, 1):
        csv_tests = []
        forder_list=os.listdir(path_dir+"/"+folder[j])
        # print(forder_list)
        for list in forder_list:
            try:
                # print("%s/%s/%s/results1/_DNN_performance/DNN_eval.txt" % (path_dir, folder, list))
                csv_tests.append(pd.read_csv("%s/%s/%s/results1/_DNN_performance/pred.txt" % (path_dir, folder[j], list), engine='python',sep='\t',names=['%s' % folder[j]]))
                true_data=pd.read_csv("%s/%s/data1/Hyla_japonica_total_env_dataframe_test.csv" % (path_dir, folder[j]), engine='python')['present/pseudo_absent']
            except FileNotFoundError:
                continue
        # total_preds = pd.concat([round(csv_tests[0]-(count/10)) ,round(csv_tests[1]-count) ,round(csv_tests[2]-count),round(csv_tests[3]-count),round(csv_tests[4]-count),round(csv_tests[5]-count),round(csv_tests[6]-count),round(csv_tests[7]-count),round(csv_tests[8]-count),round(csv_tests[9]-count)], axis=1)
        # print(total_preds)
        total_preds = pd.concat(
            [(csv_tests[0]), (csv_tests[1]), (csv_tests[2]), (csv_tests[3]), (csv_tests[4]),
             (csv_tests[5]), (csv_tests[6]), (csv_tests[7]), (csv_tests[8]), (csv_tests[9])],
            axis=1)

        total_preds = pd.DataFrame(np.transpose(total_preds))
        # print(true_data)
        # print(total_preds)
        total_preds=round(np.clip(total_preds - count / 10, 0, 1))

        total_preds = total_preds.sum(axis=0)/10
        # preds_result = pd.concat(total_preds[1].value_counts()[1.0], total_preds[2].value_counts()[1.0])
        # print(total_preds)
        preds_result=[]
        for i in range(total_preds.size):
            # preds_result.append(sum(total_preds[i]) / 10)
            if total_preds[i]>0.5:
                preds_result.append(1)
            else:
                preds_result.append(0)
        # print(preds_result)
        # print(true_data)

        mcm = multilabel_confusion_matrix(np.round(true_data, 0), np.round(preds_result, 0))
        tp = mcm[:, 1, 1][1]  # a
        fn = mcm[:, 1, 0][1]  # b
        fp = mcm[:, 0, 1][1]  # c
        tn = mcm[:, 0, 0][1]  # d

        # print("tp:", tp)
        # print("fn:", fn)
        # print("fp:", fp)
        # print("tn:", tn)

        print(roc_auc_score(true_data, preds_result))
        # print("Sensitivity :", (tp / (tp + fp)))  # Sensitivity
        # Sorensen.append((2 * tp) / (fn + (2 * tp) + fp))
        # OPR.append( fp / (tp + fp))
        # UPR.append(fn / (tp + fn))
        Sensitivity.append(tp / (tp + fp))
        # print("specificity :", (tn / (fn + tn)))  # specificity
        specificity.append(tn / (fn + tn))
        # print((tn / (fp + tn))[1]) #
        # print("KAPPA :", cohen_kappa_score(preds_result, true_data))
        KAPPA.append(cohen_kappa_score(preds_result, true_data))
        # print("TSS :", (tp / (tp + fp)) + (tn / (fn + tn)) - 1)
        TSS.append((tp / (tp + fp)) + (tn / (fn + tn)) - 1)
        # print("bias :", (((tp + fn) / (tp + fp))))
        # print("prevalence :", ((tp + fp) / (tp + fp + fn + tn)))

    Sensitivity=pd.DataFrame(Sensitivity).sum()
    specificity=pd.DataFrame(specificity).sum()
    KAPPA=pd.DataFrame(KAPPA).sum()
    TSS=pd.DataFrame(TSS).sum()
    # Sorensen =pd.DataFrame(Sorensen).sum()
    # OPR =pd.DataFrame(OPR).sum()
    # UPR =pd.DataFrame(UPR).sum()
    # print(np.array(Sensitivity))
    # print(np.array(specificity))
    # print(np.array(KAPPA))
    # print(np.array(TSS))
    # print(np.array(Sorensen))
    # print(np.array(OPR))
    # print(np.array(UPR))
    # print("-------")

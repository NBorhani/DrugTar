from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve

import numpy as np
from math import sqrt

def get_mse(actual, predicted):
    loss = ((actual - predicted) ** 2).mean(axis=0)
    return loss


def get_accuracy(actual, predicted, threshold):
    correct = 0
    predicted_classes = []
    for prediction in predicted :
      if prediction >= threshold :
        predicted_classes.append(1)
      else :
        predicted_classes.append(0)
    for i in range(len(actual)):
      if actual[i] == predicted_classes[i]:
        correct += 1
    return correct / float(len(actual))


def pred_to_classes(actual, predicted, threshold):
    predicted_classes = []
    for prediction in predicted :
      if prediction >= threshold :
        predicted_classes.append(1)
      else :
        predicted_classes.append(0)
    return predicted_classes
    
#precision
def get_tp(actual, predicted, threshold):
    predicted_classes = pred_to_classes(actual, predicted, threshold)
    tp = 0
    for i in range(len(predicted_classes)):
      if predicted_classes[i] == 1 and actual[i] == 1:
       tp += 1
    return tp
    
    
def get_fp(actual, predicted, threshold):
    predicted_classes = pred_to_classes(actual, predicted, threshold)
    fp = 0
    for i in range(len(predicted_classes)):
      if predicted_classes[i] == 1 and actual[i] == 0:
       fp += 1
    return fp


def get_tn(actual, predicted, threshold):
    predicted_classes = pred_to_classes(actual, predicted, threshold)
    tn = 0
    for i in range(len(predicted_classes)):
      if predicted_classes[i] == 0 and actual[i] == 0:
       tn += 1
    return tn


def get_fn(actual, predicted, threshold):
    predicted_classes = pred_to_classes(actual, predicted, threshold)
    fn = 0
    for i in range(len(predicted_classes)):
      if predicted_classes[i] == 0 and actual[i] == 1:
       fn += 1
    return fn
    
    
#precision = TP/ (TP + FP)    
def get_precision(actual, predicted, threshold):
    if get_tp(actual, predicted, threshold)==0:
        prec=0
    else:
        prec = get_tp(actual, predicted, threshold) / (get_tp(actual, predicted, threshold) + get_fp(actual, predicted, threshold))

    return prec
    
    
#recall = TP / (TP + FN)   
# sensitivity = recall 
def get_recall(actual, predicted, threshold):
    sens = get_tp(actual, predicted, threshold)/ (get_tp(actual, predicted, threshold) + get_fn(actual, predicted, threshold))
    return sens
    
    
#Specificity = TN/(TN+FP)    
def get_specificity(actual, predicted, threshold):     
   spec =  get_tn(actual, predicted, threshold)/ (get_tn(actual, predicted, threshold) + get_fp(actual, predicted, threshold))
   return spec

#NPV = TN/(TN+FN)    
def get_NPV(actual, predicted, threshold):  
   if get_tn(actual, predicted, threshold)==0:
       npv = 0
   else:
       npv =  get_tn(actual, predicted, threshold)/ (get_tn(actual, predicted, threshold) + get_fn(actual, predicted, threshold))
   return npv

#f1 score  = 2 / ((1/ precision) + (1/recall))   
def get_f_score(actual, predicted, threshold):
    if get_precision(actual, predicted, threshold)==0 or get_recall(actual, predicted, threshold)==0:
        f_sc = 0
    else:
        f_sc = 2 / ( (1 / get_precision(actual, predicted, threshold)) + (1/ get_recall(actual, predicted, threshold)))
    return f_sc
   
#mcc = (TP * TN - FP * FN) / sqrt((TN+FN) * (FP+TP) *(TN+FP) * (FN+TP)) 
def mcc(act, pred, thre):
   tp = get_tp(act, pred, thre) 
   tn = get_tn(act, pred, thre)
   fp = get_fp(act, pred, thre)
   fn = get_fn(act, pred, thre)
   mcc_met = (tp*tn - fp*fn) / (sqrt((tn+fn)*(fp+tp)*(tn+fp)*(fn+tp)))
   return mcc_met
   

def get_auroc(act, pred):
   return roc_auc_score(act, pred)
  

def get_auprc(act, pred):
   return average_precision_score(act, pred)



def get_optimal_cutoff(target, predicted):
    precision_, recall_, thresholds_ = precision_recall_curve(target, predicted)
    f1_scores_ = 2 * recall_ * precision_ / (recall_ + precision_)
    best_threshold_index = np.argmax(f1_scores_)
    best_threshold = thresholds_[best_threshold_index]
    #print(f'Best Threshold (F1-score): {best_threshold}')
    return best_threshold


def evaluate(y_true, y_pred):
    auc = get_auroc(y_true, y_pred)
    auprc = get_auprc(y_true, y_pred)
    
    threshold = get_optimal_cutoff(y_true, y_pred)
    precision = get_precision(y_true, y_pred, threshold)
    recall = get_recall(y_true, y_pred, threshold)
    accuracy = get_accuracy(y_true,y_pred, threshold)
    specificity = get_specificity(y_true, y_pred, threshold)
    NPV = get_NPV(y_true, y_pred, threshold)
    F1 = get_f_score(y_true, y_pred, threshold)            
    return auc, auprc, precision, recall, accuracy, specificity, NPV, F1
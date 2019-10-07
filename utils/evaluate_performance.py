# -*- coding: utf-8 -*-
from __future__ import print_function

"""
Created on Thu Jan 12 09:40:26 2017
# v.17 :from sklearn.cross_validation import StratifiedKFold
# v.18 sklearn.model_selection import StratifiedKFolds
@author: 310241758
"""
import os
import time
import itertools
from scipy import interp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from numpy import newaxis, arange, unique, linspace, array,mean

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import auc, f1_score, roc_curve, precision_recall_curve,average_precision_score




def class_info(classes):
    classes = classes.tolist()
    unique_labels = unique(classes)
    counts = []
    percentages = []
    for k in unique_labels:
        counts.append(classes.count(k))        
    total = len(classes)
    print("Initial class percentages:")
    for i, cls in enumerate(counts):
        print("%6s: % 7d  =  % 5.1f%%" % (i, counts[i], float(counts[i])/float(total)*100))
        percentages.append(float(counts[i])/float(total)*100)
    return percentages

def plot_confusion_matrix(cm, classes=False,
                          normalize=False,
                          title='Confusion matrix', model_name=None, pyramid_lvl=0):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    fig, ax1 = plt.subplots(figsize=(16, 10))    
    cmap = plt.cm.Blues
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    if classes:
        tick_marks = arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:,newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    timestr = time.strftime("%Y%m%d-%H%M%S")    
    fig_title = 'ConfusionMatrix_ ' + model_name + '' + classes[1] + '_'  + timestr 
    fig_path = fig_title + '.png'
    plt.savefig(fig_path,bbox_inches='tight')
    

def plot_ROC_curve(classifier, X, y, pos_label=1, n_folds=5,
                   neighbors=3, class_name=None, model_name=None):
    
    """
    This function prints and plots the ROC curve for Cross Validation
    n_folds can be changed, default n_folds = 5
    
    """
    # check the class percentages
    percentages = class_info(y)
    # define figure dimensions
    fig, ax1 = plt.subplots(figsize=(16, 10))
    mean_tpr = 0.0
    mean_fpr = linspace(0, 1, 100)

    # v.17: class sklearn.cross_validation.StratifiedKFold(y, n_folds=3, shuffle=False, random_state=None)
    skf = KFold(n_splits=n_folds, shuffle=True, random_state=1)

    for i, (train_index, test_index) in enumerate(skf.split(X,y)):
        # calculate the probability of each class assuming it to be positive
        probas_ = classifier.fit(X[train_index], y[train_index]).predict_proba(X[test_index])
        # Compute ROC curve and area under the curve
        fpr, tpr, thresholds = roc_curve(y[test_index], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
     
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i+1, roc_auc))
        
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random')
         
        
    mean_tpr /= n_folds
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
  
    plt.plot(mean_fpr, mean_tpr, 'k--',
         label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate (1- specificity)')
    plt.ylabel('True Positive Rate (sensitivity)')

    # plt.title('ROC Curve for %s. %s. Instances =  %d, Class 0 = % 5.1f%%, Class 1 = % 5.1f%%. Neighbors: %d. PyramidLvl: %d'
    # % (str(class_name), model_name, len(X), percentages[0],percentages[1], int(neighbors), int(pyramid_lvl)))
    plt.grid()
    plt.legend(loc="lower right")
    
    # save plot to file disk      
    timestr = time.strftime("%Y%m%d-%H%M%S")    
    fig_title = 'ROC_curve ' + class_name + str(neighbors) + '_' + timestr + model_name
    fig_path = fig_title + '.png'
    print(fig_path)
    plt.savefig(fig_path,bbox_inches=0, dpi=1200)
   
    plt.show()

#---------------------------------------------------------
def plot_PR_curve(classifier, X, y, n_folds=5, neighbors=None, 
                  class_name=None, model_name=None):
    """
    Plot a basic precision/recall curve.
    n_folds = 5 default value
    """
    fig, ax1 = plt.subplots(figsize=(16, 10))
    percentages = class_info(y)
    average_auc = []
    f1_val = []
    #mean_recall = 0.0 # recall=sensitivity
   # mean_precision= linspace(0, 1, 100)
    # v.18 sklearn uses n_splits instead of n_folds as parameter
    # e.g: class sklearn.cross_validation.StratifiedKFold(y, n_folds=3, shuffle=False, random_state=None)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=1)
    
    for i, (train_index, test_index) in enumerate(skf.split(X,y)):
        # compute the probabilities after prediction
        probas_ = classifier.fit(X[train_index], y[train_index]).predict_proba(X[test_index])
        class_ = classifier.fit(X[train_index], y[train_index]).predict(X[test_index])
        # Compute ROC curve and area the curve
        precision, recall, thresholds = precision_recall_curve(y[test_index], probas_[:, 1],
                                                               )
        #mean_recall += interp(mean_recall, recall, precision)
        average_precision = average_precision_score(y[test_index], probas_[:,1])  
        average_auc.append(average_precision)
        f1_val.append(f1_score(y[test_index], class_))                                              
        plt.plot(recall, precision, lw=1,
                 label='PR fold %d (area = %0.2f)' % (i+1,average_precision))
 
    plt.plot([0, 1], [0.5, 0.5], '--', color=(0.6, 0.6, 0.6), label='Random')
 #   mean_auc = mean(array(average_auc))
    mean_f1 = mean(array(f1_val))
    #extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    # F1 = 2 * (precision * recall) / (precision + recall)
    print('Mean F1 score = %0.2f ' % mean_f1)

#    plt.plot(mean_precision, mean_recall, 'k--',
#         label='Mean PR (area = %0.2f)' % mean_auc, lw=2)    
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # plt.title('Precision-Recall Curve for %s. %s. Instances =  %d, Class 0 = % 5.1f%%, Class 1 = % 5.1f%%. Neighbors: %d. PyramidLvl: %d'
    #         % (class_name, model_name, len(X), percentages[0], percentages[1], int(neighbors), int(pyramid_lvl)))
    plt.grid()        
    #ax1.legend([extra], ('Mean F1 score = ' + str(mean_f1)))
    plt.legend(loc="lower right")
    
    # save plot to file disk         
    timestr = time.strftime("%Y%m%d-%H%M%S")    
    fig_title = 'PR_curve ' + class_name + str(neighbors) + '_' + timestr + model_name
    fig_path = fig_title + '.png'
    plt.savefig(fig_path, bbox_inches=0, dpi=100)
    plt.show()
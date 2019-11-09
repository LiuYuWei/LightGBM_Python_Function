#輸入datasets
from sklearn import datasets
#pandas可以提供高效能、簡易使用的資料格式(DataFrame)，讓使用者可以快速操作及分析資料
import pandas as pd
#數學公式計算都靠它
import numpy as np
#畫圖都靠它
import matplotlib.pyplot as plt
#此套件可將資料自由切分成 訓練資料集 和 測試資料集
from sklearn.model_selection import train_test_split
#標準化資料集
from sklearn.preprocessing import minmax_scale
#計算accuracy,recall,precision測量指標
from sklearn.metrics import accuracy_score,recall_score,precision_score,confusion_matrix

import lightgbm as lgb

def LightGBM(train_X, train_y, eval_X, eval_y, boosting_type = 'gbdt', objective = 'binary', num_leaves = 100,
             learning_rate = 0.05, feature_fraction = 0.9, bagging_fraction = 0.8, bagging_freq = 10, verbose = 0,
             num_boost = 300, early_stopping = 20):

    lgb_train = lgb.Dataset(train_X, train_y)
    lgb_eval = lgb.Dataset(eval_X, eval_y, reference=lgb_train)

    params = {
    'task': 'train',
    'boosting_type': boosting_type,
    'objective': objective,
    'metric': {'l2', 'auc'},
    'num_leaves': num_leaves,
    'learning_rate': learning_rate,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 10,
    'verbose': 0
    }

    print('Start training...')
    # train
    gbm = lgb.train(params,
    lgb_train,
    num_boost_round = num_boost,
    valid_sets = lgb_eval,
    early_stopping_rounds = early_stopping)
    return gbm

def text_size_plot(SMALL_SIZE = 18,MEDIUM_SIZE = 20,BIGGER_SIZE = 22):
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def prediction(y_pred, y_dev, small_CM=22, median_CM=22, big_CM=22, small_FI=24, median_FI=28, big_FI=32,              W_size_CM=9, L_size_CM=6, W_size_FI=24, L_size_FI=18, dpi_CM=80, dpi_FI=80):
    print('prediction:\n')
    y_pred[y_pred>=0.5]=1
    y_pred[y_pred<0.5]=0

    accuracy = accuracy_score(y_dev, y_pred)
    precision = precision_score(y_dev, y_pred,pos_label=1)
    recall = recall_score(y_dev, y_pred,pos_label=1)
    tn, fp, fn, tp = confusion_matrix(y_dev, y_pred).ravel()
    cm = confusion_matrix(y_dev, y_pred)

    text_size_plot(median_CM, big_CM, small_FI)
    plt.figure(num=None, figsize=(W_size_CM, L_size_CM), dpi=dpi_CM)
    plot_confusion_matrix(cm, classes=[0, 1], title='Confusion matrix')
    plt.savefig(Result_file+'/Confusion_matrix_'+now_time+'_'+str(MIN_PAST_MONTH)+'_'+str(MAX_PAST_MONTH)+'_CV.png')

    print('Accuracy = ' + str(accuracy) +'\nPrecision = ' + str(precision) +'\nRecall = ' + str(recall) + '\n')
    print('tn, fp, fn, tp = ('+str(tn)+', '+str(fp)+', '+str(fn)+', '+str(tp)+')\n')
    print(classification_report(y_dev, y_pred,digits=4))
    print('The roc of prediction is:', roc_auc_score(y_dev, y_pred))

    text_size_plot(24,28,32)
    plt.figure(num=None, dpi=dpi_FI)
    lgb.plot_importance(gbm, max_num_features=20).figure.set_size_inches(W_size_FI, L_size_FI)
    plt.title("Feature Importance")
    plt.show()


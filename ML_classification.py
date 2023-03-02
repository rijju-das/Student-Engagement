class ML_classification():
  def stratified_split(self, df, val_percent=0.2):
    import numpy as np
    '''
    Function to split a dataframe into train and validation sets, while preserving the ratio of the labels in the target variable
    Inputs:
    - df, the dataframe
    - target, the target variable
    - val_percent, the percentage of validation samples, default 0.2
    Outputs:
    - train_idxs, the indices of the training dataset
    - val_idxs, the indices of the validation dataset
    '''
    classes=[0,1,2]
    train_idxs, val_idxs = [], []
    for c in classes:
        idx=list(df.loc[df['Label_y']==c].index)
        np.random.shuffle(idx)
        val_size=int(len(idx)*val_percent)
        val_idxs+=idx[:val_size]
        train_idxs+=idx[val_size:]
    return train_idxs, val_idxs


  def feature_sets(self, df):
    import pandas as pd
    
    train_idxs, val_idxs = self.stratified_split(df, val_percent=0.25)
    val_idxs, test_idxs = self.stratified_split(df[df.index.isin(val_idxs)], val_percent=0.5)

    train_df = df[df.index.isin(train_idxs)]
    df_mp_train = train_df.loc[:,'x0':'y467']
    X_s1_train = df_mp_train.values

    df_gaze_train = train_df.loc[:,'gaze_0_x':'gaze_1_z']
    df_hpose_train = train_df.loc[:,'pose_Tx':'pose_Rz']
    s2_train = pd.concat([df_gaze_train, df_hpose_train],axis=1)
    X_s2_train = s2_train.values

    df_au_train = train_df.loc[:,"AU01_r":"AU45_r"]
    X_s3_train = df_au_train.values

    s4_train = pd.concat([df_mp_train,df_au_train,df_gaze_train,df_hpose_train], axis=1)
    X_s4_train = s4_train.values

    val_df = df[df.index.isin(val_idxs)]
    df_mp_val = val_df.loc[:,'x0':'y467']
    X_s1_val = df_mp_val.values

    df_gaze_val = val_df.loc[:,'gaze_0_x':'gaze_1_z']
    df_hpose_val = val_df.loc[:,'pose_Tx':'pose_Rz']
    s2_val = pd.concat([df_gaze_val, df_hpose_val],axis=1)
    X_s2_val = s2_val.values

    df_au_val = val_df.loc[:,"AU01_r":"AU45_r"]
    X_s3_val = df_au_val.values

    s4_val = pd.concat([df_mp_val,df_au_val,df_gaze_val,df_hpose_val], axis=1)
    X_s4_val = s4_val.values

    test_df = df[df.index.isin(test_idxs)]
    df_mp_test = test_df.loc[:,'x0':'y467']
    X_s1_test = df_mp_test.values

    df_gaze_test = test_df.loc[:,'gaze_0_x':'gaze_1_z']
    df_hpose_test = test_df.loc[:,'pose_Tx':'pose_Rz']
    s2_test = pd.concat([df_gaze_test, df_hpose_test],axis=1)
    X_s2_test = s2_test.values

    df_au_test = test_df.loc[:,"AU01_r":"AU45_r"]
    X_s3_test = df_au_test.values

    s4_test = pd.concat([df_mp_test,df_au_test,df_gaze_test,df_hpose_test], axis=1)
    X_s4_test = s4_test.values

    Y_train = train_df[['Label_y']].values
    Y_val = val_df[['Label_y']].values
    Y_test = test_df[['Label_y']].values

    #store data, all in numpy arrays
    training_data=[]
    training_data.append({'X_train':X_s1_train,'Y_train':Y_train,
                    'X_val': X_s1_val,'Y_val':Y_val,
                    'X_test': X_s1_test,'Y_test':Y_test})

    training_data.append({'X_train':X_s2_train,'Y_train':Y_train,
                    'X_val': X_s2_val,'Y_val':Y_val,
                    'X_test': X_s2_test,'Y_test':Y_test})

    training_data.append({'X_train':X_s3_train,'Y_train':Y_train,
                    'X_val': X_s3_val,'Y_val':Y_val,
                    'X_test': X_s3_test,'Y_test':Y_test})

    training_data.append({'X_train':X_s4_train,'Y_train':Y_train,
                    'X_val': X_s4_val,'Y_val':Y_val,
                    'X_test': X_s4_test,'Y_test':Y_test})
    return training_data

  def classifier_result(self, training_data,i,path_o):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    import joblib
    import os
    
    temp=[]

    from sklearn.ensemble import RandomForestClassifier

    rf_clf = RandomForestClassifier(n_jobs=None,random_state=27, verbose=1)
    rf_clf.fit(training_data['X_train'], training_data['Y_train'].reshape(training_data['Y_train'].shape[0],))
    predicted_labels = rf_clf.predict(training_data['X_test'])
    train_pred = rf_clf.predict(training_data['X_train'])
    temp.append([accuracy_score(training_data['Y_test'], predicted_labels),
                 precision_score(training_data['Y_test'], predicted_labels, average='weighted'), 
                 recall_score(training_data['Y_test'], predicted_labels, average='weighted'),
                 f1_score(training_data['Y_test'], predicted_labels,average='micro')])  
    path_d = os.path.join(path_o,"model_rf_%d"%(i))   
    joblib.dump(rf_clf,path_d)
    from xgboost.sklearn import XGBClassifier
    #initial model
    xgb = XGBClassifier(learning_rate=0.1,
                        n_estimators=1000,
                        max_depth=9,
                        min_child_weight=1,
                        # gamma=0,
                        # subsample=0.8,
                        # colsample_bytree=0.8,
                        objective='multi:softmax',
                        # nthread=4,
                        num_class=9,
                        seed=27)
    xgb.fit(training_data['X_train'], training_data['Y_train'].reshape(training_data['Y_train'].shape[0],))
    predicted_labels = xgb.predict(training_data['X_test'])
    train_pred = xgb.predict(training_data['X_train'])
    temp.append([accuracy_score(training_data['Y_test'], predicted_labels),
                 precision_score(training_data['Y_test'], predicted_labels, average='weighted'), 
                 recall_score(training_data['Y_test'], predicted_labels, average='weighted'),
                 f1_score(training_data['Y_test'], predicted_labels,average='micro')])     
    path_d = os.path.join(path_o,"model_xgb_%d"%(i))   
    joblib.dump(xgb,path_d)
    from sklearn import tree
    dt_clf = tree.DecisionTreeClassifier(max_depth=39)
    dt_clf.fit(training_data['X_train'], training_data['Y_train'].reshape(training_data['Y_train'].shape[0],))
    predicted_labels = dt_clf.predict(training_data['X_test'])
    train_pred = dt_clf.predict(training_data['X_train'])
    temp.append([accuracy_score(training_data['Y_test'], predicted_labels),
                 precision_score(training_data['Y_test'], predicted_labels, average='weighted'), 
                 recall_score(training_data['Y_test'], predicted_labels, average='weighted'),
                 f1_score(training_data['Y_test'], predicted_labels,average='micro')])    
    path_d = os.path.join(path_o,"model_dt_%d"%(i))   
    joblib.dump(dt_clf,path_d)
    from sklearn.svm import SVC
    svc_clf = SVC(C=1, gamma=0.001,probability=True)
    svc_clf.fit(training_data['X_train'], training_data['Y_train'].reshape(training_data['Y_train'].shape[0],))
    predicted_labels = svc_clf.predict(training_data['X_test'])
    train_pred = svc_clf.predict(training_data['X_train'])
    temp.append([accuracy_score(training_data['Y_test'], predicted_labels),
                 precision_score(training_data['Y_test'], predicted_labels, average='weighted'), 
                 recall_score(training_data['Y_test'], predicted_labels, average='weighted'),
                 f1_score(training_data['Y_test'], predicted_labels,average='micro')])     
    path_d = os.path.join(path_o,"model_svc_%d"%(i))   
    joblib.dump(svc_clf,path_d)
    from sklearn.ensemble import GradientBoostingClassifier
    clf_gb=GradientBoostingClassifier(random_state = 0)
    clf_gb.fit(training_data['X_train'], training_data['Y_train'].reshape(training_data['Y_train'].shape[0],))
    predicted_labels = clf_gb.predict(training_data['X_test'])
    train_pred = clf_gb.predict(training_data['X_train'])
    temp.append([accuracy_score(training_data['Y_test'], predicted_labels),
                 precision_score(training_data['Y_test'], predicted_labels, average='weighted'), 
                 recall_score(training_data['Y_test'], predicted_labels, average='weighted'),
                 f1_score(training_data['Y_test'], predicted_labels,average='micro')])     
    path_d = os.path.join(path_o,"model_gb_%d"%(i))   
    joblib.dump(clf_gb,path_d)
    fig = self.plot_roc_curve_all([xgb,rf_clf,dt_clf,clf_gb,svc_clf],training_data['X_test'],training_data['Y_test'])
    return(temp, fig)

  def roc_values(self, model, y_score,y_test,n_classes):
    import numpy as np
    from sklearn.metrics import roc_curve, auc
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
      fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
      roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
      mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    return fpr,tpr,roc_auc
  

  def plot_roc_curve_all(self, model_list, X_test, Y_test):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    
    m = len(model_list)
    Y_score=[]
    for i in range(m):
      Y_score.append(model_list[i].predict_proba(X_test))

    n_classes = len(np.unique(Y_test))
    Y_test = label_binarize(Y_test, classes=np.arange(n_classes))
    
    fpr=[]
    tpr=[]
    roc_auc=[]
    for i in range(m):
      f, t, r = self.roc_values(model_list[i],Y_score[i],Y_test,n_classes)
      fpr.append(f)
      tpr.append(t)
      roc_auc.append(r)

    # Plot ROC curve
    fig = plt.figure(figsize=(5,5))
    plt.rc('axes', labelsize=18)
    plt.rc('legend', fontsize=12)
    plt.plot(fpr[0]["micro"], tpr[0]["micro"],color="red",
            label='XGBoost(area = {0:0.2f})'
                  ''.format(roc_auc[0]["micro"]))
    plt.plot(fpr[1]["micro"], tpr[1]["micro"],color="blue",
            label='Random forest(area = {0:0.2f})'
                  ''.format(roc_auc[1]["micro"]))
    plt.plot(fpr[2]["micro"], tpr[2]["micro"], color="purple",
            label='Decision tree(area = {0:0.2f})'
                  ''.format(roc_auc[2]["micro"]))
    plt.plot(fpr[3]["micro"], tpr[3]["micro"], color="green",
            label='Gradient boosting(area = {0:0.2f})'
                  ''.format(roc_auc[3]["micro"]))
    plt.plot(fpr[4]["micro"], tpr[4]["micro"],color="yellow",
            label='SVM(area = {0:0.2f})'
                  ''.format(roc_auc[4]["micro"]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()
    return fig

    

  def plot_roc_curve(self, model, X_test, Y_test):

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    
    y_score = model.predict_proba(X_test)

    n_classes = len(np.unique(Y_test))
    y_test = label_binarize(Y_test, classes=np.arange(n_classes))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
      fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
      roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
      mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot ROC curve
    fig = plt.figure(figsize=(4,4))
    # plt.plot(fpr["micro"], tpr["micro"],
    #         label='micro-average ROC curve (area = {0:0.2f})'
    #               ''.format(roc_auc["micro"]))
    # plt.plot(fpr["macro"], tpr["macro"],
    #         label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]))

    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='Class {0} (area = {1:0.2f})'
                                      ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()
    return fig
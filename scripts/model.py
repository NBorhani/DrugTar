import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from metrics import evaluate

import keras
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler,EarlyStopping
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest,f_regression


# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

def train(model, train_feature, train_label, test_feature, test_label,
          save_model = False, model_file='model/',
          feat_selected=4000, k_fold=10, init_lr=0.0002, num_epochs=110,
          batch_size=64):
  
    # Build the model
    model.build_model()
    print("Initialize DrugTar successfully")
    model = model.get_model()
    model.summary()
    
    
    # Data loading
    print("Loading ProTar-II...")
    
    features = pd.read_pickle(os.path.join(current_dir,'..', train_feature))
    features.head()    
    print("Shape of features DataFrame:", features.shape)
    
    with open(os.path.join(current_dir,'..', train_label), 'rb') as f:
        labels = pickle.load(f)
    labels = labels.values.ravel()
    print("Shape of labels DataFrame:", labels.shape)
       
    
    # Create a StratifiedKFold object
    skf = StratifiedKFold(n_splits=k_fold, shuffle=True)

    metrics_cv = {
        "AUC": [],
        "AUPRC": [],
        "Precision": [],
        "Recall": [],
        "Accuracy": [],
        "Specificity": [],
        "NPV": [],
        "F1": []
    }
    
    # Iterate over each fold
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(features, labels)):
        
        print(f"\nFold {fold_idx + 1}/10, Start training...")
    
        # Split the data into training and validation sets for this fold
        X_train_fold, X_val_fold = features.iloc[train_idx], features.iloc[val_idx]
        y_train_fold, y_val_fold = labels[train_idx], labels[val_idx]   
        print("number of tot features:", X_train_fold.shape[1])     
    
        # Create the feature selector 
        selector = SelectKBest(f_regression, k=feat_selected) 
        selector.fit(X_train_fold, y_train_fold) 
        
        X_train_selected = selector.transform(X_train_fold) 
        X_val_selected = selector.transform(X_val_fold) 
        print("number of selected features:", X_train_selected.shape[1])
    
        METRICS = [
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc'),
            keras.metrics.AUC(name='pr_auc', curve='PR'),
        ]
        
        optimizer = Adam()

        def scheduler(epoch):
            lr = init_lr
            if epoch < 5:
                return lr  # Initial learning rate
            elif epoch < 10:
                return lr * 0.5  # Learning rate after 5 epochs (multiplied by gamma=0.5)
            elif epoch < 15:
                return lr * 0.5 * 0.5  # Learning rate after 10 epochs (multiplied by gamma=0.5 again)
            else:
                return lr * 0.5 * 0.5 * 0.5  # Learning rate after 15 epochs (multiplied by gamma=0.5 again)
    
        # Create a learning rate scheduler callback
        lr_scheduler = LearningRateScheduler(scheduler)
        
        # Define EarlyStopping callback
        early_stopping = EarlyStopping(monitor='val_auc', patience=5, restore_best_weights=True)
    
        # Compile the model
        model.compile(optimizer=optimizer,
                      loss='binary_crossentropy',
                      metrics= METRICS)
    
        # Train the model
        history = model.fit(X_train_selected, y_train_fold, epochs=num_epochs, batch_size=batch_size,
                            validation_data=(X_val_selected, y_val_fold), callbacks=[early_stopping, lr_scheduler])
         
        # Plot training and testing loss functions
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Test Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
    
        plt.figure(figsize=(6,6))
        plt.subplot(2, 1, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Test Loss')
        plt.legend(loc='upper right')
        plt.title('Loss')
        
        plt.subplot(2, 1, 2)
        plt.plot(history.history['auc'], label='Train auc')
        plt.plot(history.history['val_auc'], label='Test auc')
        plt.ylim([0, 1])
        plt.title('auc')
        
        plt.show()
    
        # Validation
        y_pred = model.predict(X_val_selected)
        
        # Calculate AUC
        auc = roc_auc_score(y_val_fold, y_pred)        
        y_true = y_val_fold
        
        auc, auprc, precision, recall, accuracy, specificity, NPV, F1 = evaluate(y_true,y_pred)
        
        # Append metrics
        metrics_cv["AUC"].append(auc)
        metrics_cv["AUPRC"].append(auprc)
        metrics_cv["Precision"].append(precision)
        metrics_cv["Recall"].append(recall)
        metrics_cv["Accuracy"].append(accuracy)
        metrics_cv["Specificity"].append(specificity)
        metrics_cv["NPV"].append(NPV)
        metrics_cv["F1"].append(F1)
        
        # Print metrics
        print("\nEvaluating fold %d ..." % (fold_idx+1))
        
        print("--- valid AUC score:                %.4f" % auc)
        print("--- valid AUPRC score:              %.4f" % auprc)
        print("--- valid precision score:          %.4f" % precision)
        print("--- valid recall score:             %.4f" % recall)
        print("--- valid accuracy score:           %.4f" % accuracy)
        print("--- valid specificity score:        %.4f" % specificity)
        print("--- valid NPV score:                %.4f" % NPV)
        print("--- valid F1 score:                 %.4f" % F1)
        
        break
        
    # Print metrics
    print("\nNow 10 fold cross validation metrics is calculating ....")
    
    # Print mean and std for each metric
    for metric, values in metrics_cv.items():
        mean_value = np.mean(values)
        std_value = np.std(values)
        print(f"--- mean {metric.lower()} score: {mean_value:.4f} (±{std_value:.4f})")
        
    
    if  save_model:
        # Save the model
        model.save(os.path.join(current_dir,'..', model_file, 'model.h5'))
    
        with open(os.path.join(current_dir,'..', model_file, 'selector.pkl'), 'wb') as f:
            pickle.dump(selector, f)
            
        print("Model saved successfully...")


def test(model, test_feature, test_label, save_model, model_file): 
    
    try:
        # Load the saved model
        model.load_model(os.path.join(current_dir,'..', model_file, 'model.h5'))
        print("Initialize DrugTar successfully...")
        model = model.get_model()
        model.summary()

        with open(os.path.join(current_dir,'..', model_file, 'selector.pkl'), 'rb') as f:
            selector = pickle.load(f)
            print("\nLoaded trained DrugTar")
    except:
        print("No DrugTar found, use 'train' phase!")
        return None

    
    X_Independent = pd.read_pickle(os.path.join(current_dir,'..', test_feature))
    X_Independent.head()
    print("number of tot features:", X_Independent.shape[1]) 
    
    with open(os.path.join(current_dir,'..', test_label), 'rb') as f:
        Y_Independent = pickle.load(f)

    X_Independent_to_use = selector.transform(X_Independent)
    print("number of selected features:", X_Independent.shape[1])  
    # Validation
    y_pred_valid = model.predict(X_Independent_to_use)
    y_true_valid = Y_Independent
    y_true_valid = y_true_valid.values.ravel()
    auc_valid, auprc_valid, precision_valid, recall_valid, accuracy_valid, specificity_valid, NPV_valid, F1_valid = evaluate(y_true_valid,y_pred_valid)
    
    print("\nEvaluating Independent set...")
    
    print("--- Ind valid AUC score:                %.4f" % auc_valid)
    print("--- Ind valid AUPRC score:              %.4f" % auprc_valid)
    print("--- Ind valid precision score:          %.4f" % precision_valid)
    print("--- Ind valid recall score:             %.4f" % recall_valid)
    print("--- Ind valid accuracy score:           %.4f" % accuracy_valid)
    print("--- Ind valid specificity score:        %.4f" % specificity_valid)
    print("--- Ind valid NPV score:                %.4f" % NPV_valid)
    print("--- Ind valid F1 score:                 %.4f" % F1_valid)

    
def prediction(model, test_feature, save_pred_file, save_model, model_file):

    try:
        # Load the saved model
        model.load_model(os.path.join(current_dir,'..', model_file, 'model.h5'))
        print("Initialize DrugTar successfully...")
        model = model.get_model()
        model.summary()
            
            
        with open(os.path.join(current_dir,'..', model_file, 'selector.pkl'), 'rb') as f:
            selector = pickle.load(f)
            print("\nLoaded trained DrugTar")
    except:
        print("No DrugTar found, use 'train' phase!")
        return None
    
    # Data loading
    print("\nLoading prediction data...")
    
    test_pred = pd.read_pickle(os.path.join(current_dir,'..', test_feature))
    test_pred.head()
    print("number of tot features:", test_pred.shape[1]) 
    
    test_pred_selected = selector.transform(test_pred)
    print("number of selected features:", test_pred_selected.shape[1]) 
    # Validation
    druggability_pred = model.predict(test_pred_selected)
    print(druggability_pred.shape)
    
    # Save the druggability scores  
    # Convert the NumPy array to a pandas DataFrame
    df = pd.DataFrame(druggability_pred, columns=['druggability score'])
    
    # Save the DataFrame to a CSV file
    df.to_csv(os.path.join(current_dir,'..', save_pred_file), index=False)
        
    print("\nDruggability scores saved successfully...")
    print("file path: %s" % save_pred_file)


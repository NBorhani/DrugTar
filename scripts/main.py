import numpy as np
import argparse

from model import train,test,prediction
from DrugTar import DrugTar

def str2bool(value):
    return value.lower() == 'true'

parser = argparse.ArgumentParser(description='')
# Learning hyperparameters
parser.add_argument('--phase',          dest='phase',           default='train') # choices:'train' (10-fold CV), 'test', 'prediction'
parser.add_argument('--batch_size',     dest='batch_size',      type=int,       default=32)
parser.add_argument('--num_epochs',     dest='num_epochs',      type=int,       default=110)
parser.add_argument('--k_fold',         dest='k_fold',          type=int,       default=10)
parser.add_argument('--init_lr',        dest='init_lr',         type=float,     default=0.0002)
parser.add_argument('--save_model',     dest='save_model',      type=str2bool,  default=False)

# Architecture hyperparameters
parser.add_argument('--DNN_dims',       dest='DNN_dims',        type=str,       default='128_64_32')
parser.add_argument('--dropout',        dest='dropout',         type=float,     default=0.5)
parser.add_argument('--feat_selected',  dest='feat_selected',   type=int,       default=4000)

# Loading datasets
parser.add_argument('--train_feature',  dest='train_feature',   default='feature/train_feature.pkl') # ProTar-II
parser.add_argument('--train_label',    dest='train_label',     default='feature/train_label.pkl') # ProTar-II
parser.add_argument('--test_feature',   dest='test_feature',    default='feature/test_feature.pkl') # ProTar-II-Ind
parser.add_argument('--test_label',     dest='test_label',      default='feature/test_label.pkl') # ProTar-II-Ind

# Loading and saving files
parser.add_argument('--model_file',     dest='model_file',      default='models/')
parser.add_argument('--save_pred_file', dest='save_pred_file',  default='prediction/druggability_scores.csv') # path for save druggability prediction scores

args = parser.parse_args()


# Training phase
if args.phase == 'train':
    
    # Initialize the DrugTar model
    model = DrugTar(input_dim=args.feat_selected, 
                    hidden_dim=list(map(int, args.DNN_dims.split("_"))), 
                    dropout_rate=args.dropout)

    # Training and evaluating DrugTar model with k-fold cross validation
    train(model, train_feature=args.train_feature, train_label=args.train_label, 
          test_feature=args.test_feature, test_label=args.test_label,
          save_model = args.save_model, model_file=args.model_file,
          feat_selected=args.feat_selected, k_fold=args.k_fold, 
          init_lr=args.init_lr, num_epochs=args.num_epochs,
          batch_size=args.batch_size)
    
    
# Independent test phase
elif args.phase == 'test': 

    # Initialize the DrugTar model with same architecture as training
    model = DrugTar(input_dim=args.feat_selected, 
                    hidden_dim=list(map(int, args.DNN_dims.split("_"))), 
                    dropout_rate=args.dropout) 
    
    # Make predictions on the test set using the trained model
    test(model, test_feature=args.test_feature, test_label=args.test_label, 
         save_model = args.save_model, model_file=args.model_file)
    
    
# Prediction phase for unlabeled proteins
elif args.phase == 'prediction':  
    
    # Load the trained model and perform predictions on unseen data
    # Feature number should match the training phase
    model = DrugTar(input_dim=args.feat_selected, 
                    hidden_dim=list(map(int, args.DNN_dims.split("_"))), 
                    dropout_rate=args.dropout)  
    
    # Predict labels for the test features and save the prediction scores
    prediction(model, test_feature=args.test_feature, save_pred_file=args.save_pred_file,
               save_model = args.save_model, model_file=args.model_file)
 
    
# Handle unknown phases       
else:
    print("Unknown phase! Please select one of the following: 'train', 'test', or 'prediction'")
    


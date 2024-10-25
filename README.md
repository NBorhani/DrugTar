# DrugTar-Protein Druggability Prediction
![DrugTar wide](https://github.com/user-attachments/assets/919309a3-2664-408a-9c95-10c1ce74b650)


This repository contains the Python code and dataset for the paper titled **“DrugTar: A Deep Learning Framework Integrating ESM-2 Embeddings and GO Terms for Protein Druggability Prediction”** by Niloofar Borhani, Iman Izadi, Ali Motahharynia, Mahsa Sheikholeslami, and Yousof Gheisari, currently under review.

## Feature Generation
- **`esm2_embedder.py`**: Generates ESM-2 embeddings for each protein sequence.
- **`feats_extraction.py`**: Extracts features (ESM-2 embeddings and Gene Ontology (GO) terms) and corresponding labels, creating feature matrices as DataFrames.

## Training and Evaluation with k-Fold Cross Validation
To train the DrugTar model and validate it using k-fold cross-validation, run the following command:

```bash
python scripts/main.py --phase='train' \
--batch_size=32 --num_epochs=110 --k_fold=10 --init_lr=0.0002 \
--save_model='True' --DNN_dims='128_64_32' --dropout=0.5 \
--feat_selected=4000 \
--train_feature='feature/train_feature.pkl' \
--train_label='feature/train_label.pkl' \
--test_feature='feature/test_feature.pkl' \
--test_label='feature/test_label.pkl' \
--model_file='models/' \
--save_pred_file='prediction/druggability_scores.csv'
```



## Testing the Trained Model
To test the performance of the trained DrugTar model on independent test data:
```
python scripts/main.py --phase='test' \
--batch_size=32 --num_epochs=110 --init_lr=0.0002 \
--save_model='True' --DNN_dims='128_64_32' --dropout=0.5 \
--feat_selected=4000 \
--test_feature='feature/test_feature.pkl' \
--test_label='feature/test_label.pkl' \
--model_file='models/' \
--save_pred_file='prediction/druggability_scores.csv'
```

## Prediction for Unlabeled Proteins
To predict druggability scores for unlabeled proteins using the trained DrugTar model:
```
python scripts/main.py --phase='prediction' \
--batch_size=32 --num_epochs=110 --init_lr=0.0002 \
--save_model='True' --DNN_dims='128_64_32' --dropout=0.5 \
--feat_selected=4000 \
--test_feature='feature/test_feature.pkl' \
--model_file='models/' \
--save_pred_file='prediction/druggability_scores.csv'
```

## Contact Information
For further inquiries, please contact.

**Niloofar Borhani**  
Ph.D. Student, Control Engineering  
Isfahan University of Technology  
Email: [n.borhani@ec.iut.ac.ir](mailto:n.borhani@ec.iut.ac.ir)  
CV: [Google Scholar](https://scholar.google.com/citations?user=SSD_k8MAAAAJ&hl=en)

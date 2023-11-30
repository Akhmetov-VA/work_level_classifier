from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA

import torch
import pandas as pd
import numpy as np

from transformers import AutoTokenizer, AutoModel

from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device('cuda')


class BERTEmbExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, tokenizer=None, model=None, columns=None, batch_size=128):
        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("ai-forever/ruBert-base")
            
        if model:
            self.model = model
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = AutoModel.from_pretrained("ai-forever/ruBert-base").to(device)
            
        self.columns = columns
        self.batch_size = batch_size
        
    def bert_extract(self, X, col):
        out_emb = []
        total_samples = len(X)

        for start in tqdm(range(0, total_samples, self.batch_size)):
            end = min(start + self.batch_size, total_samples)
            
            batch_X = X[col].values[start:end]  # Extract a batch of data
            tokenized = self.tokenizer(batch_X.tolist(), padding = True, truncation = True, return_tensors="pt")
            column_tokens = {k: torch.tensor(v).to(device) for k, v in tokenized.items()}

            hidden_state = self.model(**column_tokens) #dim : [batch_size(nr_sentences), tokens, emb_dim]
            emb = hidden_state.last_hidden_state[:,0,:].to("cpu")
            del hidden_state
            
            df_emb = pd.DataFrame(emb, columns=[f'{col}_{i}' for i in range(emb.shape[1])])
            out_emb.append(df_emb)
            
        return pd.concat(out_emb)
    
    def fit(self, X, y=None):
        if not self.columns:
            self.columns = X.columns
        
        self.column_df_emb = {}
        self.column_pca = {}
        with torch.no_grad():
            for col in self.columns:
                df_emb = self.bert_extract(X, col)
                
                pca = PCA(0.99)
                pca.set_output(transform='pandas')
                
                df_emb = pca.fit_transform(df_emb)

                self.column_pca[col] = pca
            
        return self

    def transform(self, X, y=None):
        out = []
        with torch.no_grad():
            for col in self.columns:
                df_emb = self.bert_extract(X, col)
                df_emb = self.column_pca[col].transform(df_emb)
                df_emb.columns = [f'{col}_{pca_col}' for pca_col in df_emb.columns]
                out.append(df_emb)
        
        return pd.concat(out, axis=1)
    
    
def find_best_ccp_aplpha(X_train, y_train, X_val, y_val):
    print('Start compute cost_complexity_pruning_path')
    clf = DecisionTreeClassifier(random_state=0)
    path = clf.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    
    fig, ax = plt.subplots()
    ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
    ax.set_xlabel("effective alpha")
    ax.set_ylabel("total impurity of leaves")
    ax.set_title("Total Impurity vs effective alpha for training set")
    plt.show()
    
    print('Start searching best ccp alphas')
    ccp_alphas_select = ccp_alphas[np.linspace(0, len(ccp_alphas[:-15]), 30).astype(int)].tolist() + ccp_alphas[-15:].tolist()
    
    clfs = []
    for ccp_alpha in tqdm(ccp_alphas_select):
        clf = ExtraTreesClassifier(class_weight='balanced_subsample',
                                oob_score=True, 
                                bootstrap=True, 
                                n_jobs=5, 
                                random_state=0, 
                                ccp_alpha=ccp_alpha)
        clf.fit(X_train, y_train)
        clfs.append(clf)
    
    
    train_scores = [clf.score(X_train, y_train) for clf in clfs]
    test_scores = [clf.score(X_val, y_val) for clf in clfs]

    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs alpha for training and testing sets")
    ax.plot(ccp_alphas_select, train_scores, marker="o", label="train", drawstyle="steps-post")
    ax.plot(ccp_alphas_select, test_scores, marker="o", label="test", drawstyle="steps-post")
    ax.legend()
    plt.show()
    
    best_hyper = np.array(test_scores).argmax()
    print(f'best accuracy val: train {train_scores[best_hyper]} ,{test_scores[best_hyper]}') 
    best_ccp = ccp_alphas_select[best_hyper]
    
    print(f'Лучший коэф для прунинга {best_ccp}')
    
    return best_ccp

def metrics(pipe, X_val, y_val, LE):
    y_pred = pipe.predict(X_val)
    print(classification_report(y_val, y_pred,))

    display_labels = sorted(list(set(y_pred).union(y_val)))
    ConfusionMatrixDisplay.from_estimator(
        pipe, X_val, y_val, display_labels=LE.classes_[display_labels], xticks_rotation="vertical"
    )
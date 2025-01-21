import os
import re
import numpy as np
from Bio import SeqIO
import esm
import pickle
import torch
import torch.nn.functional
    

def read_fasta(file):
    all_sequence=[]
    fasta_sequences = SeqIO.parse(open(file),'fasta')
    for fasta in fasta_sequences:
        name, sequence = fasta.id, str(fasta.seq)
        all_sequence.append([name,re.sub('[^ACDEFGHIKLMNPQRSTVWY]', '', ''.join(sequence).upper())])
    return all_sequence


esm2b, esm2b_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
esm2b = esm2b.eval()
esm2b_batch_converter = esm2b_alphabet.get_batch_converter()


def protein_embedding(path):
    
    i = 1
    all_seq = read_fasta(path)
    esm2_embed = {}
    
    for pr_seq in all_seq:
        esm2b_data = [
        pr_seq,
        ]      
        
        if '|' in pr_seq[0]:
            pr_name = pr_seq[0].split('|')[1]
        else:
            pr_name = pr_seq[0].split('|')[0]
        
        file_path = os.path.join(current_dir,'..', 'dataset/ave_esm2_embedding/'+ pr_name +'.npy')
        
        if os.path.exists(file_path):
            print(i,'-',pr_name, 'ESM-2 Exist!')
            i = i+1
            
            ave_token_representations =  np.load(file_path)
            esm2_embed[pr_name] = ave_token_representations
        else:
            print(len(pr_seq[1]))
            esm2b_batch_labels, esm2b_batch_strs, esm2b_batch_tokens = esm2b_batch_converter(esm2b_data)
            b =esm2b_batch_tokens.shape[1:3][0]
            if b>1024:
                tokens=esm2b_batch_tokens[:,0:1024]
            elif b<=1024:
                tokens=esm2b_batch_tokens
            tokens = torch.tensor(tokens.numpy())
            with torch.no_grad():
                tokens=tokens
                results = esm2b(tokens, repr_layers=[33], return_contacts=True)
                token_representations = results["representations"][33].cpu()
            torch.cuda.empty_cache()
            print(i,'- Processing',pr_name)
            i = i + 1
            ave_token_representations = torch.mean(token_representations,dim=1)     
            
            esm2_embed[pr_name] = ave_token_representations
            
            np.save(file_path,ave_token_representations.numpy())
            torch.cuda.empty_cache()


    with open(os.path.join(current_dir,'..', 'dataset/ave_esm2_embedding.pkl'), 'wb') as f:
        pickle.dump(esm2_embed, f)   
    print('ESM_ave_Dict saved successfully...')         
    torch.cuda.empty_cache()
    
    
if __name__ == "__main__":   
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    protein_embedding(os.path.join(current_dir,'..', 'dataset/Training Set.fasta'))
    
        

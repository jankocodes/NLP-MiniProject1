import torch
import torch.nn as nn

class Skipgram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        
        self.vocab_size= vocab_size
        self.embedding_dim= embedding_dim
        
        self.center_embedding= nn.Embedding(vocab_size, embedding_dim)
        
        self.context_embedding= nn.Embedding(vocab_size, embedding_dim)
        
        self.init_embeddings()
    
    def init_embeddings(self):
        
        self.center_embedding.weight.data.uniform_(-1,1)
        self.context_embedding.weight.data.uniform_(-1,1)

    def forward(self, center, context, negatives):
        center_emb= self.center_embedding(center) #(batch_size x embedding_dim)
        
        context_emb= self.context_embedding(context) #(batch_size x embedding_dim)
        
        negatives_emb= self.context_embedding(negatives) #(batch_size x num_neg_samples x embedding_dim)
        
        positive_score= torch.sum(center_emb*context_emb, dim=1) #(batch_size)

        #(batch_size x num_neg_samples x embedding_dim) * (batch_size, embedding_dim, 1)
        negative_score= torch.bmm(negatives_emb, center_emb.unsqueeze(2)).squeeze(2) #(batch_size, num_neg_samples)
        
        return (positive_score, negative_score)
    
    
        
    
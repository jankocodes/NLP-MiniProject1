import torch
import torch.nn as nn

class Skipgram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        
        self.vocab_size= vocab_size
        self.embedding_dim= embedding_dim
        
        self.embedding= nn.Embedding(vocab_size, embedding_dim)
        
        self.output_embedding= nn.Embedding(vocab_size, embedding_dim)
        
        self.init_embeddings()
    
    def init_embeddings(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.output_embedding.weight)
    
    def forward(self, center, context, negatives):
        center_embedding= self.embedding(center) #(batch_size x embedding_dim)
        
        context_embedding= self.output_embedding(context) #(batch_size x embedding_dim)
        
        negatives_embedding= self.output_embedding(negatives) #(batch_size x num_neg_samples x embedding_dim)
        
        positive_score= torch.sum(center_embedding*context_embedding, dim=1) #(batch_size)

        #(batch_size x num_neg_samples x embedding_dim) * (batch_size, embedding_dim, 1)
        negative_score= torch.bmm(negatives_embedding, center_embedding.unsqueeze(2)).squeeze(2) #(batch_size, num_neg_samples)
        
        return (positive_score, negative_score)
    
    
        
    
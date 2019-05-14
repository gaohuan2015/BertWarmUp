import torch.nn as nn
import torch.nn.functional as F

class CBOW(nn.Module):
    def __init__(self, vocabulary_size, embedding_size, padding_idx=0):
        super(CBOW, self).__init__()
        self.embedding =  nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=embedding_size, padding_idx=padding_idx)
        self.fc1 = nn.Linear(in_features=embedding_size, out_features=vocabulary_size)

    def forward(self, x_in):
        x_embedded_sum = F.dropout(self.embedding(x_in).sum(dim=1), 0.3)
        y_out = self.fc1(x_embedded_sum)
        y_out = F.softmax(y_out, dim=1)
        return y_out

import math

import torch
import torch.nn as nn
import torch.nn.functional as F




class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        embedding_dim,
        block_size,
        heads,
        dropout
    ):

        super().__init__()

        

        assert embedding_dim % heads == 0

        self.heads = heads
        self.head_dim = embedding_dim // heads

        

        self.query = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.key = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.value = nn.Linear(
            embedding_dim,
            embedding_dim
        )

      

        self.proj = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        

        self.attn_dropout = nn.Dropout(
            dropout
        )

        self.proj_dropout = nn.Dropout(
            dropout
        )

        

        mask = torch.tril(
            torch.ones(
                block_size,
                block_size
            )
        )

        self.register_buffer(
            "mask",
            mask.view(
                1,
                1,
                block_size,
                block_size
            )
        )


    def forward(self, x):

        B, T, C = x.shape

        

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

      

        q = q.view(
            B,
            T,
            self.heads,
            self.head_dim
        ).transpose(1, 2)

        k = k.view(
            B,
            T,
            self.heads,
            self.head_dim
        ).transpose(1, 2)

        v = v.view(
            B,
            T,
            self.heads,
            self.head_dim
        ).transpose(1, 2)

        

        att = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(
            self.head_dim
        )

        

        att = att.masked_fill(
            self.mask[
                :, :, :T, :T
            ] == 0,
            float("-inf")
        )

        

        att = F.softmax(
            att,
            dim=-1
        )

       

        att = self.attn_dropout(att)

        

        out = att @ v

        

        out = out.transpose(
            1,
            2
        ).contiguous().view(
            B,
            T,
            C
        )

        

        out = self.proj(out)

        out = self.proj_dropout(out)

        return out




class FeedForward(nn.Module):

    def __init__(
        self,
        embedding_dim,
        dropout
    ):

        super().__init__()

        self.net = nn.Sequential(

            
            nn.Linear(
                embedding_dim,
                4 * embedding_dim
            ),

            
            nn.GELU(),

          
            nn.Linear(
                4 * embedding_dim,
                embedding_dim
            ),

            # Dropout
            nn.Dropout(
                dropout
            )
        )

    

    def forward(self, x):

        return self.net(x)




class Block(nn.Module):

    def __init__(
        self,
        embedding_dim,
        block_size,
        heads,
        dropout=0.1
    ):

        super().__init__()

        

        self.ln1 = nn.LayerNorm(
            embedding_dim
        )

       

        self.attn = MultiHeadAttention(
            embedding_dim,
            block_size,
            heads,
            dropout
        )

        

        self.ln2 = nn.LayerNorm(
            embedding_dim
        )

       

        self.ff = FeedForward(
            embedding_dim,
            dropout
        )

    

    def forward(self, x):

        

        x = x + self.attn(
            self.ln1(x)
        )

        

        x = x + self.ff(
            self.ln2(x)
        )

        return x


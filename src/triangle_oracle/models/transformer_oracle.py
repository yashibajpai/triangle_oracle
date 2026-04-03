import torch  
import torch.nn as nn  


class EdgeTransformerOracle(nn.Module):  
    """
    transformer encoder model for predicting edge heaviness
    """

    def __init__(  # initialize model
        self,
        vocab_size: int,  # total number of tokens
        d_model: int = 128,  # embedding dimension
        nhead: int = 4,  # number of attention heads
        num_layers: int = 3,  # number of encoder layers
        dim_feedforward: int = 256,  # hidden size in feedforward network
        dropout: float = 0.1,  # dropout rate
        max_len: int = 128,  # max sequence length
    ):
        super().__init__()  # call parent constructor

        self.token_embedding = nn.Embedding(vocab_size, d_model)  # map tokens to vectors

        self.position_embedding = nn.Embedding(max_len, d_model)  # positional encoding

        encoder_layer = nn.TransformerEncoderLayer(  # define one encoder layer
            d_model=d_model,  # embedding size
            nhead=nhead,  # number of heads
            dim_feedforward=dim_feedforward,  # feedforward size
            dropout=dropout,  # dropout
            batch_first=True,  # input format [batch, seq, dim]
            activation="gelu",  # activation function
        )

        self.encoder = nn.TransformerEncoder(  # stack encoder layers
            encoder_layer=encoder_layer,  # base layer
            num_layers=num_layers,  # number of layers
        )

        self.regressor = nn.Sequential(  # regression head
            nn.Linear(d_model, d_model),  # linear layer
            nn.ReLU(),  # activation
            nn.Dropout(dropout),  # dropout
            nn.Linear(d_model, 1),  # output scalar
        )

        self.max_len = max_len  # store max length

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:  # forward pass

        batch_size, seq_len = input_ids.shape  # get batch and sequence size

        if seq_len > self.max_len:  # check sequence length
            raise ValueError(f"sequence too long: {seq_len}")  # throw error if too long

        positions = torch.arange(seq_len, device=input_ids.device)  # create position indices
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)  # expand to batch size

        x = self.token_embedding(input_ids)  # get token embeddings
        x = x + self.position_embedding(positions)  # add positional embeddings

        src_key_padding_mask = attention_mask == 0  # create mask for padding tokens

        h = self.encoder(x, src_key_padding_mask=src_key_padding_mask)  # run transformer encoder

        mask = attention_mask.unsqueeze(-1)  # expand mask to match embedding dims
        h_masked = h * mask  # zero out padded positions

        pooled = h_masked.sum(dim=1)  # sum across sequence
        pooled = pooled / mask.sum(dim=1).clamp(min=1.0)  # divide by number of valid tokens

        pred_log = self.regressor(pooled)  # pass through regression head
        pred_log = pred_log.squeeze(-1)  # remove last dimension

        return pred_log  # return prediction
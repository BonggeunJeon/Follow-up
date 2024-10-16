import torch
import torch.nn as nn
import numpy as np

class Model_mlp_mse(nn.Module):
    ## NN with three relu hidden layers
    # quantile outputs are independent of each other
    def __init__(self, 
                 n_input, 
                 n_hidden, 
                 n_output, 
                 is_dropout=False, 
                 is_batch=False, 
                 activation="relu",
    ):
        super(Model_mlp_mse, self).__init__()
        self.layer1 = nn.Linear(n_input, n_hidden, bias=True)
        self.layer2 = nn.Linear(n_hidden, n_hidden, bias=True)
        self.layer3 = nn.Linear(n_hidden, n_hidden, bias=True)
        self.layer4 = nn.Linear(n_hidden, n_output, bias=True)
        self.drop1 = nn.Dropout(0.333)
        self.drop2 = nn.Dropout(0.333)
        self.drop3 = nn.Dropout(0.333)
        self.batch1 = nn.BatchNorm1d(n_hidden)
        self.batch2 = nn.BatchNorm1d(n_hidden)
        self.batch3 = nn.BatchNorm1d(n_hidden)
        self.is_dropout = is_dropout
        self.is_batch = is_batch
        self.activation = activation
        self.loss_fn = nn.MSELoss()
        
    def forward_net(self, x):
        # layer 1
        x = self.layer1(x)
        if self.activation == "relu":
            x = torch.relu(x)
        elif self.activation == "gelu":
            x = torch.nn.functional.gelu(x)
        else:
            raise Exception("bad activation passed in")
        if self.is_dropout:
            x = self.drop1(x)
        if self.is_batch:
            x = self.batch1(x)
        
        # layer 2    
        x = self.layer2(x)
        if self.activation =="relu":
            x = torch.relu(x)
        elif self.activation == "gelu":
            x = torch.nn.functional.gelu(x)
        if self.is_dropout:
            x = self.drop2(x)
        if self.is_batch:
            x = self.batch2(x)
            
        # uncomment for 3 hidden layer
        x = self.layer3(x)
        if self.activation == "relu":
            x = torch.relu(x)
        elif self.activation == "gelu":
            x = torch.nn.functional.gelu(x)
        if self.is_dropout:
            x = self.drop3(x)
        if self.is_batch:
            x = self.batch3(x)
            
        x = self.layer4(x)
        return x
    
    def forward(self, x):
        # we write this in this was so can reuse forward_net in Model_mlp_diff
        return self.forward_net(x)
    
    def loss_on_batch(self, x_batch, y_batch):
        # add this here so can sync w diffusion model
        y_pred_batch = self(x_batch)
        loss = self.loss_fn(y_pred_batch, y_batch)
        return loss
    
    def sample(self, x_batch):
        return self(x_batch)
    
# Model_mlp_diff <- Model_mlp_mse
class Model_mlp_diff(Model_mlp_mse):
    # This model just piggy backs onto the vanilla MLP
    # later on I'll use a fancier architecture, ie transformer
    # and also make it possible to condition on images
    def __init__(
        self,
        x_dim,
        n_hidden,
        y_dim,
        is_drouput=False,
        is_batch=False,
        activation="relu",
    ):
        n_input = x_dim + y_dim + 1
        n_output= y_dim
        super(Model_mlp_diff, self).__init__(n_input, n_hidden, n_output, is_drouput, is_batch, activation)
    
    def forward(self, y, x, t, context_mask):
        nn_input = torch.cat([y, x, t], dim=-1) # torch.cat이 뭐야?
        return self.forward_net(nn_input)
    
    def loss_on_batch(self, x_batch, y_batch):
        # overwrite these methods as won't use them w diffusion model
        raise NotImplementedError
    
    def sample(self, x_batch):
        raise NotImplementedError
    

class TimeSiren(nn.Module): # 이게 무슨 역할을 할까???
    def __init__(self, input_dim, emb_dim):
        super(TimeSiren, self).__init__()
        # just a fully connected NN with sin activations
        self.lin1 = nn.Linear(input_dim, emb_dim, bias=False)
        self.lin2 = nn.Linear(emb_dim, emb_dim)
        
    def forward(self, x):
        x = torch.sin(self.lin1(x))
        x = self.lin2(x)
        return x
    
class FCBlock(nn.Module): # FUlly-Connected Block(?)
    def __init__(self, in_feats, out_feats):
        super().__init__()
        # one layer of non_linearities (just a useful building block to use below
        self.model = nn.Sequential(
            nn.Linear(in_feats, out_feats),
            nn.BatchNorm1d(num_features=out_feats),
            nn.GELU(),
        )
        
    def forward(self, x):
        return self.model(x)
    
class TransformerEncoderBlock(nn.Module):
    def __init__(self, trans_emb_dim, transformer_dim, nheads):
        super(TransformerEncoderBlock, self).__init__()
        
        self.trans_emb_dim = trans_emb_dim
        self.transformer_dim = transformer_dim
        self.nheads = nheads
        
        self.input_to_qkv1 = nn.Linear(self.trans_emb_dim, self.transformer_dim * 3)
        self.multihead_attn1 = nn.MultiheadAttention(self.transformer_dim, num_heads=self.nheads) # 이게 뭐야?
        self.attn1_to_fcn = nn.Linear(self.transformer_dim, self.trans_emb_dim)
        self.attn1_fcn = nn.Sequential(
            nn.Linear(self.trans_emb_dim, self.trans_emb_dim * 4),
            nn.GELU,
            nn.Linear(self.trans_emb_dim * 4, self.trans_emb_dim),
        )
        self.norm1a = nn.BatchNorm1d(self.trans_emb_dim)
        self.norm1b = nn.BatchNorm1d(self.trans_emb_dim)
        
    def split_qkv(self, qkv):
        assert qkv.shape[-1] == self.transformer_dim * 3 # assert 는 if statement랑 비슷
        # [:, :, :] 이게 무슨 의미일까?
        q = qkv[:, :, :self.transformer_dim]
        k = qkv[:, :, self.transformer_dim: 2 * self.transformer_dim]
        v = qkv[:, :, 2 * self.transformer_dim:]
        return (q, k, v)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        
        qkvs1 = self.input_to_qkv1(inputs) # nn.Linear
        # shape out = [3, batchsize, transformer_dim * 3]
        
        qs1, ks1, vs1 = self.split_qkv(qkvs1)
        # shape out = [3, batchsize, transformer_dim]
        
        attn1_a = self.multihead_attn1(qs1, ks1, vs1, need_weights=False) #nn.MultiheadAttention
        attn1_a - attn1_a[0]
        # shape out = [3, batchsize, transformer_dim = trans_emb_dix X nheads]
        
        attn1_b = self.attn1_to_fcn(attn1_a)    # nn.Linear
        attn1_b = attn1_b / 1.414 + inputs / 1.414 # add residual
        # shape out = [3, batchsize, trans_emb_dim]
        
        # normalize
        attn1_b = self.norm1a(attn1_b.transpose(0, 2).transpose(0, 1)) # nn.BatchNorm1d
        attn1_b = attn1_b.transpose(0, 1).transpose(0, 2)
        # batchnorm likes shape = [batchsize, trans_emb_dim, 3]
        # so have to shape like this, then return
        
        # fully connected layer
        attn1_c = self.attn1_fcn(attn1_b) / 1.414 + attn1_b / 1.414 # nn.Sequential(Linear -> GELU -> Linear)
        # shape out = [3, batchsize, trans_emb_dim]
        
        # normalize
        # attn1_c = self.norm1b(attn1_c)
        attn1_c = self.norm1b(attn1_c.transpose(0, 2).transpose(0, 1)) # nn.BatchNorm1d
        attn1_c = attn1_c.transpose(0, 1).transpose(0, 2)
        return attn1_c
    

class Model_mlp_diff_embed(nn.Module):
    # this model embeds x, y, t, before input into a fc NN (w residuals)
    def __init__(
        self,
        x_dim,
        n_hidden,
        y_dim,
        embed_dim,
        output_dim=None,
        is_dropout=False,
        is_batch=False,
        activation="relu",
        net_type="fc",
        use_prev=False,
    ):
        super(Model_mlp_diff_embed, self).__init__()
        self.embed_dim = embed_dim # input embedding dimension
        self.n_hidden = n_hidden
        self.net_type = net_type
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.use_prev = use_prev # whether x contains previous timestep
        if output_dim is None:
            self.output_dim = y_dim # by default, just output size of action space
        else:
            self.output_dim = output_dim # sometimes overwrite, eg for discretised, mean/variance, mixture density models
            
        # embedding NNs
        if self.use_prev:
            self.x_embed_nn = nn.Sequential(
                nn.Linear(int(x_dim / 2), self.embed_dim),
                nn.LeakyReLU(),
                nn.Linear(self.embed_dim, self.embed_dim),
            )
        else:
            self.x_embed_nn = nn.Sequential(
                nn.Linear(x_dim, self.embed_dim),
                nn.LeakyReLU(),
                nn.Linear(self.embed_dim, self.embed_dim),
            )
        self.y_embed_nn = nn.Sequential(
            nn.Linear(y_dim, self.embed_dim),
            nn.LeakyReLU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )
        self.t_embed_nn = TimeSiren(1, self.embed_dim)
        
        # fc nn layers
        if self.net_type == "fc":
            if self.use_prev:
                self.fc1 = nn.Sequential(FCBlock(self.embed_dim * 4, n_hidden)) # concat x, x_prev,
            else:
                self.fc1 = nn.Sequential(FCBlock(self.embed_dim * 3, n_hidden)) # no prev hist
            self.fc2 = nn.Sequential(FCBlock(n_hidden + y_dim + 1, n_hidden)) # will concat y and t at each layer
            self.fc3 = nn.Sequential(FCBlock(n_hidden + y_dim + 1, n_hidden))
            self.fc4 = nn.Sequential(nn.Linear(n_hidden + y_dim + 1, self.output_dim))
            
        # transformer layers
        elif self.net_type == "transformer":
            self.nheads = 16
            self.trans_emb_dim = 64
            self.transformer_dim = self.trans_emb_dim * self.nheads # embedding dim for each of q, k and v (though only k and v have to same I think)
            
            self.t_to_input = nn.Linear(self.embed_dim, self.trans_emb_dim)
            self.y_to_input = nn.Linear(self.embed_dim, self.trans_emb_dim)
            self.x_to_input = nn.Linear(self.embed_dim, self.trans_emb_dim)
            
            self.pos_embed = TimeSiren(1, self.trans_emb_dim)
            
            self.transformer_block1 = TransformerEncoderBlock(self.trans_emb_dim, self.transformer_dim, self.nheads)
            self.transformer_block2 = TransformerEncoderBlock(self.trans_emb_dim, self.transformer_dim, self.nheads)
            self.transformer_block3 = TransformerEncoderBlock(self.trans_emb_dim, self.transformer_dim, self.nheads)
            self.transformer_block4 = TransformerEncoderBlock(self.trans_emb_dim, self.transformer_dim, self.nheads)
            
            if self.use_prev:
                self.final = nn.Linear(self.trans_emb_dim * 4, self.output_dim) # final layer params
            else:
                self.final = nn.Linear(self.trans_emb_dim * 3, self.output_dim)
        
        else:
            raise NotImplementedError
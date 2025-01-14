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
    
    def loss_on_batch(self, x_batch, y_batch): # x_batch = a mini-batch of input data | y_batch = ground truth for the mini-batch
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
        # [:, :, :] 이게 무슨 의미일까? => slicing 할 때, 사용[시작:끝:차원]
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
        elif self.net_type == "transformer": # Transformer 가 뭘까??
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
    
    def forward(self, y, x, t, context_mask):
        # embed y, x, t
        if self.use_prev:
            x_e = self.x_embed_nn(x[:, :int(self.x_dim / 2)])
            x_e_prev = self.x_embed_nn(x[:, int(self.x_dim / 2):])
        else:
            x_e = self.x_embed_nn(x) # no prev list
            x_e_prev = None
        y_e = self.y_embed_nn(y)
        t_e = self.t_embed_nn(t)
        
        # mask out context embedding, x_e, if context_mask == 1
        context_mask = context_mask.repeat(x_e.shape[1], 1).T
        x_e = x_e * (-1 * (1 - context_mask))
        if self.use_prev:
            x_e_prev = x_e_prev * (-1 * (1 - context_mask))
            
        # pass through fc nn
        if self.net_type == "fc":
            net_output = self.forward_fcnn(x_e, x_e_prev, y_e, t_e, x, y, t)
            
        # or pass through transformer encoder
        elif self.net_type == "transformer":
            net_output = self.forward_transformer(x_e, x_e_prev, y_e, t_e, x, y, t)
            
        return net_output
    
    def forward_fcnn(self, x_e, x_e_prev, y_e, t_e, x, y, t):
        if self.use_prev:
            net_input = torch.cat((x_e, x_e_prev, y_e, t_e), 1)
        else:
            net_input = torch.cat((x_e, y_e, t_e), 1)
            
        nn1 = self.fc1(net_input)
        nn2 = self.fc2(torch.cat((nn1 / 1.414, y, t), 1)) + nn1 / 1.414 # residual and concat inputs again
        nn3 = self.fc3(torch.cat((nn2 / 1.414, y, t), 1)) + nn2 / 1.414 
        net_output = self.fc4(torch.cat((nn3, y, t), 1))
        return net_output
    
    def forward_transformer(self, x_e, x_e_prev, y_e, t_e, x, y, t):
        # roughly follwing this: https://jalammar.github.io/illustrated-tranformer/  # Transformer가 뭘까???
        
        t_input = self.t_to_input(t_e)
        y_input = self.y_to_input(y_e)
        x_input = self.x_to_input(x_e)
        
        if self.use_prev:
            x_input_prev = self.x_to_input(x_e_prev)
        
        # shape out = [batchsize, trans_emb_dim]
        
        # add 'positional' encoding
        # note, here position refers to order tokens are fed into transformer
        t_input += self.pos_embed(torch.zeros(x.shape[0], 1).to(x.device) + 1.0)
        y_input += self.pos_embed(torch.zeros(x.shape[0], 1).to(x.device) + 2.0)
        x_input += self.pos_embed(torch.zeros(x.shape[0], 1).to(x.device) + 3.0)
        
        if self.use_prev:
            x_input_prev += self.pos_embed(torch.zeros(x.shape[0], 1).to(x.device) + 4.0)
            
        if self.use_prev:
            inputs1 = torch.cat(
                (
                    t_input[None, :, :],
                    y_input[None, :, :],
                    x_input[None, :, :],
                    x_input_prev[None, :, :],
                ),
                0,
            )
        else:
            inputs1 = torch.cat((t_input[None, :, :], y_input[None, :, :], x_input[None, :, :]), 0)
        # shape out = [3, batchsize, trans_emb_dim]
        
        block1 = self.transformer_block1(inputs1)
        block2 = self.transformer_block2(block1)
        block3 = self.transformer_block3(block2)
        block4 = self.transformer_block4(block3)
        
        # flatten and add final linear layer
        # transformer_out = block2
        transformer_out = block4
        transformer_out = transformer_out.transpose(0, 1) # roll batch to first dim
        # shape out = [batchsize, 3, trans_emb_dim]
        
        flat = torch.flatten(transformer_out, start_dim=1, end_dim=2)
        # shape out = [batchsize, 3 X trans_emb_dim]
        
        out = self.final(flat)
        # shape out = [batchsize, n_dim]
        return out
    
def ddpm_schedules(beta1, beta2, T, is_linear=True):
    """
        Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"
    
    # beta_t = (beta2 - beta1) * torch.arrange(0, T + 1, dtype=torch.float32) / T + beta1
    # beta_t = (beta2 - beta2) * torch.arrange(-1, T + 1, dtype=torch.float32) / T + beta1
    if is_linear:
        beta_t = (beta2 - beta1) * torch.arrange(-1, T, dtype=torch.float32) / (T - 1) + beta1
    
    else:
        beta_t = (beta2 - beta1) * torch.square(torch.arange(-1, T, dtype=torch.float32)) / torch.max(torch.square(torch.arange(-1, T, dtype=torch.float32))) + beta1
    beta_t[0] = beta1 # modifying this so that beta_1[1] = beta1, and beta_t[n_T]=beta2, while bata[0] is never used
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()
    
    
    oneover_sqrt = 1 / torch.sqrt(alpha_t)
    
    sqrtab = torch.sqrt(alphabar_t)
    sqrtmab = torch.sqrt(1 - alphabar_t) / sqrtmab
    
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab
    
    return {
        "alpha_t" : alpha_t, # alpha_t
        "oneover_sqrta" : oneover_sqrt, # 1 / sqrt{alpha_t}
        "sqrt_beta_t" : sqrt_beta_t, # sqrt{beta_t}
        "alphabar_t" : alphabar_t, # bar{alpha_t}
        "sqrtab" : sqrtab, # sqrt{bar{alpha_t}}
        "sqrtmab" : sqrtmab, # sqrt{1-bar{alpha_t}}
        "mab_over_sqrtmab_inv" : mab_over_sqrtmab_inv, # (1 - alpha_t) / sqrt{1 - bar{alpha_t}}
    }
        
class Model_Cond_Diffusion(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, x_dim, y_dim, drop_prob=0.1, guide_w=0.0):
        super(Model_Cond_Diffusion, self).__init__()
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)
            
        self.nn_model = nn_model
        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.guide_w = guide_w
        
        
    def loss_on_batch(self, x_batch, y_batch): # loss on batch가 무슨 역할을 할까? -> 각 batch 마다 loss 값을 구함
        _ts = torch.randint(1, self.n_T +1, (y_batch.shape[0], 1)).to(self.device)
        
        # dropout context with some probability
        context_mask = torch.bernoulli(torch.zeros(x_batch.shape[0])+ self.drop_prob).to(self.device)
        
        # randomly sample some noise, noise ~ N(0, 1)
        noise = torch.randn_like(y_batch).to(self.device)
        
        # add noise to clean target actions
        y_t = self.sqrtab[_ts] * y_batch + self.sqrtmab[1 - _ts] * noise
        
        # use nn model to predict nosie
        noise_pred_batch = self.nn_model(y_t, x_batch, _ts / self.n_T, context_mask)
        
        # return mse between predicted and true noise
        return self.loss_mse(noise, noise_pred_batch)
    
    def sample(self, x_batch, return_y_trace=False, extract_embedding=False):
        # also use this as a shortcut to avoid doubling batch when guide_w is zero
        is_zero = False
        if self.guide_w > -1e-3 and self.guide_w < 1e-3:
            is_zero = True
            
        # how many noisy actions to begin with
        n_sample = x_batch.shape[0]
        
        y_shape = (n_sample, self.y_dim)
        
        # sample initial noise, y_0 ~ N(0, 1),
        y_i = torch.randn(y_shape).to(self.device)
        
        if not is_zero:
            if len(x_batch.shape) > 2:
                # repeat x_batch twice, so can use guided diffusion
                x_batch = x_batch.repeat(2, 1, 1, 1)            # repeat(2, 1, 1, 1) 하고 repeat(2, 1) 차이가 뭐임?
            else:
                # repeat x_batch twice, so can use guided diffusion
                x_batch = x_batch.repeat(2, 1)
                
            # half of context will be zero
            context_mask = torch.zeros(x_batch.shape[0]).to(self.device)
            context_mask[n_sample:] = 1.0 # makes second half of batch context free
        else:
            context_mask = torch.zeros(x_batch.shape[0]).to(self.device)
            
        if extract_embedding:
            x_embed = self.nn_model.embed_context(x_batch)
            
        # run denoising chain
        y_i_store = [] # if want to trace how y_i evolved
        for i in range(self.n_T, 0, -1):
            t_is = torch.tensor([i / self.n_T]).to(self.device)
            t_is = t_is.repeat(n_sample, 1)
            
            if not is_zero:
                # double batch
                y_i = y_i.repeat(2, 1)
                t_is = t_is.repeat(2, 1)
                
            z = torch.randn(y_shape).to(self.device) if i > 1 else 0
            
            # split predictions and compute weighting
            if extract_embedding:
                eps = self.nn_model(y_i, x_batch, t_is, context_mask, x_embed)
            else:
                eps = self.nn_model(y_i, x_batch, t_is, context_mask)
            if not is_zero:
                eps1 = eps[:n_sample]
                eps2 = eps[n_sample:]
                eps = (1 + self.guide_w) * eps1 - self.guide_w * eps2
                y_i = y_i[:n_sample]
            y_i = self.oneover_sqrta[i] * (y_i - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z
            if return_y_trace and (i % 20 == 0 or i == self.n_T or i < 8):
                y_i_store.append(y_i.detach().cpu().numpy())
                
        if return_y_trace:
            return y_i, y_i_store
        else:
            return y_i
        
    def sample_update(self, x_batch, betas, n_T, return_y_trace=False):
        original_nT = self.n_T
        
        # set new schedule
        self.n_T = n_T
        for k, v in ddpm_schedules(betas[0], betas[1], self.n_T).items():
            self.register_buffer(k, v.to(self.device))
            
        # also use this as a shortcut to avoid doubling batch when guide_w is zero
        is_zero = False
        if self.guide_w > -1e-3 and self.guide_w < 1e-3:
            is_zero = True
            
        # how many nosiy actions to begin with
        n_sample = x_batch.shape[0]
        
        y_shape = (n_sample, self.y_dim)
        
        # sample initial noise, y_0 ~ N(0, 1),
        y_i = torch.randn(y_shape).to(self.device)
        
        if not is_zero:
            if len(x_batch.shape) > 2:
                # repeat x_batch twice, so can use guided diffusion
                x_batch = x_batch.repeat(2, 1, 1, 1)
            else:
                # repeat x_batch twice, so can use guided diffusion
                x_batch = x_batch.repeat(2, 1)
                
            # half of context will be zero
            context_mask = torch.zeros(x_batch.shape[0]).to(self.device)
            context_mask[n_sample:] = 1.0 # makes second half of batch context free
        else:
            context_mask = torch.zeros(x_batch.shape[0]).to(self.device)
            
        # run denoising chain
        y_i_store = [] # if want to trace how y_i evolved
        for i in range(self.n_T, 0, -1):
            t_is = torch.tensor([i / self.n_T]).to(self.device)
            t_is = t_is.repeat(n_sample, 1)
            
            if not is_zero:
                # double batch
                y_i = y_i.repeat(2, 1)
                t_is = t_is.repeat(2, 1)
                
            # I'm a bit confused why we are adding noise duing denoising? 
            z = torch.randn(y_shape).to(self.device) if i > 1 else 0
            
            # split predictions and compute weighting
            eps = self.nn_model(y_i, x_batch, t_is, context_mask)
            if not is_zero:
                eps1 = eps[:n_sample]
                eps2 = eps[n_sample:]
                eps = (1 + self.guide_w) * eps1 - self.guide_w * eps2
                y_i = y_i[:n_sample]
            y_i = self.oneover_sqrta[i] * (y_i - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z
            
            if return_y_trace and (i % 20 == 0 or i == self.n_T or i < 8):
                y_i_store.append(y_i.detach().cpu().numpy())
                
        # reset original schedule
        self.n_T = original_nT
        for k, v in ddpm_schedules(betas[0], betas[1], self.n_T).items():
            self.register_buffer(k, v.to(self.device))
            
        if return_y_trace:
            return y_i, y_i_store
        else:
            return y_i
        
    def sample_extra(self, x_batch, extra_steps=4, return_y_trace=False):
        # also use this as a shortcut to avoid doubling batch when guide_w is zero
        is_zero = False
        if self.guide_w > -1e-3 and self.guide_w < 1e-3:
            is_zero = True
            
        # how many noisy actions to begin with
        n_sample = x_batch.shape[0]
        
        y_shape = (n_sample, self.y_dim)
        
        # sample initial noise, y_0 ~ N(0, 1),
        y_i = torch.randn(y_shape).to(self.device)
        
        if not is_zero:
            if len(x_batch.shape) > 2:
                # repeat x_batch twice, so can use guided diffusion
                x_batch = x_batch.repeat(2, 1, 1, 1)
            else:
                # repeat x_batch twice, so can use guided diffusion
                x_batch = x_batch.repeat(2, 1)
            # half of context will be zero
            context_mask = torch.zeros(x_batch.shape[0]).to(self.device)
            context_mask[n_sample:] = 1.0 # makes second half of batch contexts free
            
        else:
            # context_mask = torch.zeros_like(x_batch[:,0]).to(self.device)
            context_mask = torch.zeros(x_batch.shape[0]).to(self.device)
            
        # run denoising chain
        y_i_store = [] # if want to trace how y_i evolved
        # fore i_dummy in range(self.n_T, 0, -1):
        for i_dummy in range(self.n_T, -extra_steps, -1):
            i = max(i_dummy, 1)
            t_is = torch.tensor([i / self.n_T]).to(self.device)
            t_is = t_is.repeat(n_sample, 1)
            
            if not is_zero:
                # double batch
                y_i = y_i.repeat(2, 1)
                t_is = t_is.repeat(2, 1)
                
            z = torch.randn(y_shape).to(self.device) if i > 1 else 0
            
            # split predictions and compute weighting
            eps = self.nn_model(y_i, x_batch, t_is, context_mask)
            if not is_zero:
                eps1 = eps[:n_sample]
                eps2 = eps[n_sample:]
                eps = (1 + self.guide_w) * eps1 - self.guide_w * eps2
                y_i = y_i[:n_sample]
            y_i = self.oneover_sqrta[i] * (y_i - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z
            
            if return_y_trace and (i % 20 == 0 or i == self.n_T or i < 8):
                y_i_store.append(y_i.detach().cpu().numpy())
                
        if return_y_trace:
            return y_i, y_i_store
        else:
            return y_i
        
class ResidualConvBlock(nn.Module): # 2개의 Convolution 레이어로 구성되어 있음 
    def __init__(self, in_channels, out_channels, is_res=False):
        super().__init__()
        self.same_channels = in_channels == out_channels
        self.is_res = is_res
        
        """ 
            Residual Block은 네트워크의 레이어 갯수가 점점 많아지면서 생기는 performance 저하를 막아주는 것이 주 역할
        """
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        
        def forward(self, x):
            if self.is_res:
                x1 = self.conv1(x)
                x2 = self.conv2(x1)
                # this adds on correct residual in case channels have increased
                if self.same_channels:
                    x = x + x2
                else:
                    x = x1 + x2
                return x / 1.414
            else:
                x = self.conv1(x)
                x = self.conv2(x)
                return x
        
class Model_cnn_mlp(nn.Module):
    def __init__(self, x_shape, n_hidden, y_dim, embed_dim, net_type, output_dim=None):
        super(Model_cnn_mlp, self).__init__()
        
        self.x_shape = x_shape
        self.n_hidden = n_hidden
        self.y_dim = y_dim
        self.embed_dim = embed_dim
        self.n_feat = 64
        self.net_type = net_type
        
        if output_dim is None:
            self.output_dim = y_dim # by default, just output size of action space
        else:
            self.output_dim = output_dim # sometimes overwrite, eg for discretised, mean/variance, mixture density models
            
        # set up CNN for image
        self.conv_down1 = nn.Sequential(
            ResidualConvBlock(self.x_shape[-1], self.n_feat, is_res=True),
            nn.MaxPool2d(2),
        )
        self.conv_down3 = nn.Sequential(
            ResidualConvBlock(self.n_feat, self.n_feat * 2, is_res=True),
            nn.MaxPool2d(2),
        )
        self.imageembed = nn.Sequential(nn.AvgPool2d(8))
        
        cnn_out_dim = self.n_feat * 2
        # how many features after flattening -- WARNING, will have to adjust this for diff size input resolution
        # it is the flattend size after CNN layers, and average pooling
        
        # then once have flattened vector out of CNN, just feed into previous Model_mlp_diff_embed
        self.nn_downstream = Model_mlp_diff_embed(
            cnn_out_dim,
            self.n_hidden,
            self.y_dim,
            self.embed_dim,
            self.output_dim,
            is_dropout=False,
            is_batch=False,
            activation="relu",
            net_type=self.net_type,
            use_prev=False,
        )
    
    def forward(self, y, x, t, context_mask, x_embed=None):
        # torch expects batch_size, channels, height, width
        # but we feed in batch_size, height, width, channels

        if x_embed is None:
            x_embed = self.embed_context(x)
        else:
            pass
        
        return self.nn_downstream(y, x_embed, t, context_mask)
    
    def embed_context(self, x):
        x = x.permute(0, 3, 2, 1)
        x1 = self.conv_down1(x)
        x3 = self.conv_down3(x1) # [batch_size, 128, 35, 18]
        # c3 is [batch_size, 128, 4, 4]
        x_embed = self.imageembed(x3)
        # c_embed is [batch_size, 128, 1, 1]
        x_embed = x_embed.view(x.shape[0], -1)
        # c_embed os now [batch_size, 128]
        return x_embed
    
    
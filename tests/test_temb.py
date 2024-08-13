import math
import numpy as np
import torch
torch.set_printoptions(linewidth=160)
import torch.nn as nn
from einops import rearrange

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        freqs = freqs.to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, dtype):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        if t_freq.dtype != dtype:
            t_freq = t_freq.to(dtype)
        t_freq_clone = t_freq.clone()

        t_emb = self.mlp(t_freq)
        print("TimestepEmbedder.forward")
        return t_emb,t_freq_clone


class TimestepEmbedderExpand(TimestepEmbedder):
    def forward(self,t, dtype):
        # t: (bsz,) or (bsz,f)

        if t.ndim == 1:
            # use original `TimestepEmbedder`
            return super().forward(t,dtype)
        
        assert t.ndim == 2
        b,f = t.shape
        t = rearrange(t, 'b f -> (b f)') # [t_{b1,f1},t_{b1,f2},t_{b1,f3},t_{b2,f1},t_{b2,f2},t_{b2,f3},...]
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        # t_freq = rearrange(t_freq, '(b f) c -> b f c', b=b,f=f)

        if t_freq.dtype != dtype:
            t_freq = t_freq.to(dtype)
        t_freq_clone = t_freq.clone()

        t_emb = self.mlp(t_freq) # (bsz,c); bsz = b*f

        t_emb = rearrange(t_emb, '(b f) c -> b f c', b=b,f=f)
        t_freq_clone = rearrange(t_freq_clone, '(b f) c -> b f c', b=b,f=f)

        print("TimestepEmbedderExpand.forward")
        return t_emb,t_freq_clone

DEVICE_DTYPE = dict(
    device = torch.device("cuda:0"),
    dtype = torch.float16
)

@torch.no_grad()
def main_wo_kv_cache():
    
    t_embedder = TimestepEmbedderExpand(1152)
    t_embedder.load_state_dict(load_ckpt(),strict=True)
    t_embedder = t_embedder.to(**DEVICE_DTYPE)
    timestep = torch.as_tensor([0]+[1000]*8).to(**DEVICE_DTYPE)[None,:] # (B,T), B=1
    print(timestep)
    temb,t_freq = t_embedder(timestep,dtype=timestep.dtype)
    print(t_freq,t_freq.shape) # (B,T,C)
    return temb,t_freq

@torch.no_grad()
def main_with_kv_cache():
    t_embedder = TimestepEmbedderExpand(1152)
    t_embedder.load_state_dict(load_ckpt(),strict=True)
    t_embedder = t_embedder.to(**DEVICE_DTYPE)

    timestep_clean = torch.as_tensor([0]).to(**DEVICE_DTYPE) # (B,), B=1
    timestep_denoise = torch.as_tensor([1000]).to(**DEVICE_DTYPE) # (B,), B=1

    temb_clean,t_freq_clean = t_embedder(timestep_clean,dtype=timestep_clean.dtype)
    temb_denoise,t_freq_denoise = t_embedder(timestep_denoise,dtype=timestep_denoise.dtype)

    print(t_freq_clean.shape,t_freq_denoise.shape) # (B,C)
    print(temb_clean.shape,temb_denoise.shape)
    # assert False
    return temb_clean,temb_denoise,t_freq_clean,t_freq_denoise

@torch.no_grad()
def main_with_kv_cache2():
    t_embedder = TimestepEmbedderExpand(1152)
    t_embedder.load_state_dict(load_ckpt(),strict=True)
    t_embedder = t_embedder.to(**DEVICE_DTYPE)

    timestep_clean = torch.as_tensor([0]).to(**DEVICE_DTYPE) # (B,), B=1
    timestep_denoise = torch.as_tensor([1000]).to(**DEVICE_DTYPE) # (B,), B=1

    timestep_denoise = timestep_denoise.repeat_interleave(8,dim=0)
    timestep = torch.cat([timestep_clean[None,:],timestep_denoise[None,:]],dim=1)
    print(timestep_clean.shape,timestep.shape)

    temb,t_freq = t_embedder(timestep,dtype=timestep.dtype)
    
    print(temb.shape,t_freq.shape)
    # assert False
    temb_clean,temb_denoise = temb[:,0,:], temb[:,1,:]
    t_freq_clean,t_freq_denoise =t_freq[:,0,:], t_freq[:,1,:] 
    

    print(t_freq_clean.shape,t_freq_denoise.shape) # (B,C)
    print(temb_clean.shape,temb_denoise.shape)
    # assert False
    return temb_clean,temb_denoise,t_freq_clean,t_freq_denoise

def allclose_abs(x,y,eps=1e-6):
    assert x.shape == y.shape

    num_eq = (torch.abs(x - y) < eps).sum().item()

    is_allclose = num_eq == x.numel()
    
    return num_eq,is_allclose


def compare():
    temb,t_freq = main_wo_kv_cache()
    temb_clean,temb_denoise,t_freq_clean,t_freq_denoise = main_with_kv_cache2()
    t_freq_clean0 = t_freq[:,0,:]
    t_freq_denoise0 = t_freq[:,1,:]

    num_eq,_ = allclose_abs(t_freq_clean,t_freq_clean0)
    num_all = t_freq_clean.numel()
    print(num_eq,num_all,num_eq/num_all,num_all-num_eq)

    num_eq,_ = allclose_abs(t_freq_denoise,t_freq_denoise0)
    num_all = t_freq_denoise.numel()
    print(num_eq,num_all,num_eq/num_all,num_all-num_eq)

    T_c = 1
    T=9
    T_n=8
    C=1152
    temb_clean0 = temb[:,:T_c,:][:,0,:]
    temb_denoise0 = temb[:,T_c:,:]
    assert all(torch.allclose(temb_denoise0[:,t,:],temb_denoise0[:,0,:]) for t in range(T_n))
    temb_denoise0 = temb_denoise0[:,0,:]
    print(temb_clean0.shape,temb_denoise0.shape)
    print(temb_clean.shape,temb_denoise.shape)

    num_eq,_ = allclose_abs(temb_clean,temb_clean0)
    num_all = temb_clean.numel()
    print(num_eq,num_all,num_eq/num_all,num_all-num_eq)
    for c in range(C):
        _, allclose = allclose_abs(temb_clean[:,c],temb_clean0[:,c])
        if not allclose:
            print(c,temb_clean[:,c],temb_clean0[:,c])


    num_eq,_ = allclose_abs(temb_denoise,temb_denoise0)
    num_all = temb_denoise.numel()
    print(num_eq,num_all,num_eq/num_all,num_all-num_eq)
    for c in range(C):
        _, allclose = allclose_abs(temb_denoise[:,c],temb_denoise0[:,c])
        if not allclose:
            print(c,temb_denoise[:,c],temb_denoise0[:,c])

def load_ckpt():
    state_dict = torch.load("/data/CausalSTDiT_working_dir/CausalSTDiT2-XL2_exp2_BaselineCausalAttnFixedTpe_33x256x256ArSize8pp3/epoch1-global_step13000/model_ckpt.pt",map_location=DEVICE_DTYPE["device"])

    # print(state_dict.keys())

    state_dict_t_embedder = {k:v for k,v in state_dict.items() if "t_embedder" in k}
    del state_dict

    state_dict_t_embedder = {k.replace("t_embedder.",""):v for k,v in state_dict_t_embedder.items()}

    return state_dict_t_embedder

def fp16tensor_to_hex_repr(tensor):
    
    assert tensor.dtype == torch.float16

    float16_numpy = tensor.cpu().numpy()

    # View the bytes as a uint16 (16-bit unsigned integer)
    uint16_representation = float16_numpy.view(np.uint16)

    # Get the binary representation
    binary_representation = bin(uint16_representation)[2:].zfill(16)

    # Get the hexadecimal representation
    hexadecimal_representation = hex(uint16_representation)

    # print("Float16 value:", tensor.item())
    # print("Binary representation:", binary_representation)
    # print("Hexadecimal representation:", hexadecimal_representation)
    assert isinstance(hexadecimal_representation,str)
    return hexadecimal_representation


@torch.no_grad()
def model_demo():
    bsz = 2
    channel = 4
    hidden_size, frequency_embedding_size=8,4
    generator = torch.Generator(DEVICE_DTYPE["device"])
    generator.manual_seed(111)

    t_embedder = TimestepEmbedderExpand(hidden_size, frequency_embedding_size)
    # t_embedder.load_state_dict(load_ckpt(),strict=True)
    t_embedder = t_embedder.to(**DEVICE_DTYPE)

    
    temb = torch.randn(size=(bsz,channel),generator=generator,**DEVICE_DTYPE)
    t_mlp = t_embedder.mlp(temb)
    print(t_mlp)

    # temb2 = torch.stack([temb[0,:].clone(),temb[1,:].clone()],dim=0)
    t_mlp0 = t_embedder.mlp(temb[0,None,:].clone())
    t_mlp1 = t_embedder.mlp(temb[1,None,:].clone())
    print(t_mlp0)
    print(t_mlp1)
    t_mlp2 = torch.cat([t_mlp0,t_mlp1],dim=0)

    print(torch.allclose(t_mlp,t_mlp2))
    for i in range(t_mlp.shape[0]):
        for j in range(t_mlp.shape[1]):
            allclose = torch.allclose(t_mlp[i,j],t_mlp2[i,j])
            abs_diff = torch.abs(t_mlp[i,j]-t_mlp2[i,j])
            hex1 = fp16tensor_to_hex_repr(t_mlp[i,j])
            hex2 = fp16tensor_to_hex_repr(t_mlp2[i,j])
            print((i,j),allclose,abs_diff.item(),hex1,hex2)
            # print()
    
    # print(t_mlp2.cpu().numpy().dtype)


if __name__ == "__main__":
    # main_wo_kv_cache()
    # main_with_kv_cache()
    # compare()
    # load_ckpt()
    model_demo()
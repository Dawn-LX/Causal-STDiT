import torch

generator1 = torch.Generator(device = torch.device("cuda:0"))
generator1.manual_seed(555)
B,C,T,H,W =1,4,8,32,32
x = torch.randn(size=(B,C,T,H,W),device = torch.device("cuda:0"),generator=generator1).type(torch.bfloat16)
generator = torch.Generator(device=x.device)
generator.manual_seed(99)
B,C,T,H,W = x.shape
noise = torch.randn(size=(B,C,T,H,W),device=x.device,dtype=x.dtype,generator=generator)
# print(f"<GaussianDiffusion.p_sample>: noise: {noise.shape}")
noise2print = noise.clone()
initial_seed = generator.initial_seed()
print(f"noise={noise2print[0,0,:,0,0]}, {initial_seed}, {noise2print.shape}")

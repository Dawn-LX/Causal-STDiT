import torch

torch.set_printoptions(linewidth=160,sci_mode=False,precision=2)

class CausalSTDiT:
    def __init__(self) -> None:
        self.max_tpe_len = 64
        self.pos_embed_temporal = torch.as_tensor(
            [i*0.01 for i in range(self.max_tpe_len)]
        ).float()

    def get_relative_tpe(self,num_temporal,training):
        seq_length = num_temporal
        max_length = self.max_tpe_len
        
        if training:
            assert seq_length <= max_length
            offset = torch.randint(0,max_length,size=())  # offset = offset % max_length
            rel_tpe = torch.cat(
                [self.pos_embed_temporal,self.pos_embed_temporal],
                dim=0
            )[offset:offset+seq_length]
        else:
            rel_tpe = torch.cat(
                [self.pos_embed_temporal] * (seq_length // max_length + 1),
                dim=0
            )[:seq_length]
            # NOTE: for very long seq_length, we should dequeue previous kv-cache to save memory
        
        return rel_tpe

    def forward(self,num_temporal):
        return self.get_relative_tpe(num_temporal,training=True)


    def forward_kv_cache(self,cached_len,denoise_len):
        num_temporal = denoise_len
        tpe = self.get_relative_tpe(cached_len+num_temporal,training=False)
        tpe = tpe[cached_len:cached_len+num_temporal]
        return tpe
    
    def write_kv_cache(self,cached_len,x_clean_len):
        num_temporal = x_clean_len
        tpe = self.get_relative_tpe(cached_len+num_temporal,training=False)
        tpe = tpe[cached_len:cached_len+num_temporal]
        return tpe
    

model = CausalSTDiT()
print(model.pos_embed_temporal)

tpe = model.forward(17)
print(tpe,len(tpe))
print("-="*80)
# ==================================================
# forward kv cache
# ==================================================
first_k_given = 1
ws = 8
ar_steps = 9
cached_len = 0

# write_kv_cache
tpe0 = model.write_kv_cache(cached_len,first_k_given)
print(tpe0)
cached_len +=first_k_given
for ar_id in range(ar_steps):
    tpe = model.forward_kv_cache(cached_len,denoise_len=ws)
    
    tpe = model.write_kv_cache(cached_len,x_clean_len=ws)
    print(ar_id,tpe)
    cached_len += ws


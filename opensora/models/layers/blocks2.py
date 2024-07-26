class SeqParallelCausalSelfAttention:
    pass

class CausalSelfAttention:
    pass

class SpatialConditionAttention:
    pass

'''
TODO merge these attentions according to opensora_enc_dec
i.e., we do kv_cache in the causal_stdit, not in the attention class

and input cached_kv as context to the attention.forward

Done. refer to :
    opensora/models/causal_stdit2/attention.py
    opensora/models/causal_stdit2/causal_stdit2.py
'''
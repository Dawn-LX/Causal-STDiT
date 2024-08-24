# ======================================================================
# partial causal-attn cyclic tpe, max_cond_len=25,  max_tpe_len=33, training_seqlen=25
# ======================================================================
'''
w/ kv-cache
    ar_step=0 
        latent >> cond: [0] denoise: [1-8]
        tpe >>    cond: [0] denoise: [1-8]
    
    ar_step=1 
        latent >> cond: [0][1-8] denoise [9-16]
        tpe    >> cond: [0][1-8] denoise [9-16]
    
    ar_step=2 
        latent >> cond: [0][1-8][9-16] denoise [17-24]
        tpe    >> cond: [0][1-8][9-16] denoise [17-24]

    ar_step=3 
        latent >> cond: [0][1-8][9-16][17-24] denoise [25-32]
        tpe    >> cond: [0][1-8][9-16][17-24] denoise [25-32]

    ar_step=4 
        latent >> cond: <0-7>[8][9-16][17-24][25-32] denoise [33-40]
        tpe    >> cond:      [8][9-16][17-24][25-32] denoise [0 - 7]   <--- this is seen in training

    ar_step=5
        latent >> cond: [16][17-24][25-32][33-40] denoise [41-48]
        tpe    >> cond: [16][17-24][25-32][0 - 7] denoise [8 -15] <--- this is seen in training
'''


# ======================================================================
# partial causal-attn cyclic tpe, max_cond_len=17,  max_tpe_len=33, training_seqlen=25
# ======================================================================
'''
w/ kv-cache
    ar_step=0 
        latent >> cond: [0] denoise: [1-8]
        tpe >>    cond: [0] denoise: [1-8]
    
    ar_step=1 
        latent >> cond: [0][1-8] denoise [9-16]
        tpe    >> cond: [0][1-8] denoise [9-16]
    
    ar_step=2 
        latent >> cond: [0][1-8][9-16] denoise [17-24]
        tpe    >> cond: [0][1-8][9-16] denoise [17-24]

    ar_step=3 
        latent >> cond: <0-7>[8][9-16][17-24] denoise [25-32]
        tpe    >> cond:      [8][9-16][17-24] denoise [25-32]

    ar_step=4 
        latent >> cond: <0-7><8-15>[16][17-24][25-32] denoise [33-40]
        tpe    >> cond:            [16][17-24][25-32] denoise [0 - 7]   <--- this is seen in training

    ar_step=5
        latent >> cond: [16][17-24][25-32][33-40] denoise [41-48]
        tpe    >> cond: [16][17-24][25-32][0 - 7] denoise [8 -15] <--- this is seen in training
'''
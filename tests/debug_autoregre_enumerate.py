
# ======================================================================
# causal-attn fixed tpe, max_cond_len=25,  max_tpe_len=33
# ======================================================================
''' causal-attn fixed tpe, max_cond_len=25, max_tpe_len=33

w/ kv-cache
    ar_step=0 
        latent >> cond: [0] denoise: [1-8]
        tpe >>    cond: [8] denoise: [1-8]
    
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
        latent >> cond: [8][9-16][17-24][25-32] denoise [33-40]
        tpe    >> cond: [8][9-16][17-24][25-32] denoise [25-32] <-- this is unseen in training

    ar_step=5
        latent >> cond: [16][17-24][25-32][33-40] denoise [41-48]
        tpe    >> cond: [16][17-24][25-32][25-32] denoise [25-32] <-- this is unseen in training

    NOTE 
    w/ kv-cache 的时候, fixed tpe 和前面已经cache 好的kv 已经绑定死了(frame_index=8 对应的tpe_index=8)
    这样和w/o kv-cache 是不一样的, w/o kv-cache 的时候, cond 每次装填, tpe是重新index的, e.g., for ar_step=3: frame_idx=8 corresponds to tpe_idx=8, however, for ar_step=4: frame_idx=8 corresponds to tpe_idx=0

    NOTE LLM 语言模型中是怎么解决的? 语言模型一般 pos_emb 很长, 但是超过长度之后呢?

w/ kv-cache max_cond_len=9, max_tpe_len=33
    ar_step=0 
        latent >> cond: [0] denoise: [1-8]
        tpe >>    cond: [8] denoise: [1-8]
    
    ar_step=1 
        latent >> cond: [0][1-8] denoise [9-16]
        tpe    >> cond: [0][1-8] denoise [9-16]
    
    
    ar_step=2 
        latent >> cond: [8][9-16] denoise [17-24]
        tpe    >> cond: [8][9-16] denoise [17-24]

    ar_step=3 
        latent >> cond: [16][17-24] denoise [25-32]
        tpe    >> cond: [16][17-24] denoise [25-32]

    ar_step=4 
        latent >> cond: [24][25-32] denoise [33-40]
        tpe    >> cond: [24][25-32] denoise [?-?] <-- what should we use for `?` , 用max_tpe[-denoise_len:]也不对， 不管这里tpe 用什么index， 都是和训练不一致的， 因为condtion 部分的末尾，tpe index达到了最大index， 这是训练的时候没见过的

w/o kv-cache
    ar_step=0 
        latent >> cond: [0] denoise: [1-8]
        tpe >>    cond: [8] denoise: [1-8]
    
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
        latent >> cond: [8][9-16][17-24][25-32] denoise [33-40]
        tpe    >> cond: [0][1-8][9-16][17-24]   denoise [25-32]
    
    ar_step=5
        latent >> cond: [16][17-24][25-32][33-40] denoise [41-48]
        tpe    >> cond: [0 ][1 - 8][9 -16][17-24] denoise [25-32]
'''



# ======================================================================
# causal-attn cyclic tpe, max_cond_len=25,  max_tpe_len=33
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
        latent >> cond:     <0-15>[16][17-24][25-32][33-40] denoise [41-48]
        tpe    >> cond:           [16][17-24][25-32][0 - 7] denoise [8 -15] <--- this is seen in training
    
    ar_step=6
        latent >> cond:            <0-23>[24][25-32][33-40][41-48] denoise [49-56]
        tpe    >> cond:                  [24][25-32][0 - 7][8 -15] denoise [16-23] <--- this is seen in training

    ar_step=7
        latent >> cond:                   <0-31>[32][33-40][41-48][49-56] denoise [57-64]
        tpe    >> cond:                         [32][0 - 7][8 -15][16-23] denoise [24-31] <--- this is seen in training
    
    ar_step=8
        latent >> cond:                          <0-39>[40][41-48][49-56][57-64] denoise [65 -72]
        tpe    >> cond:                                [ 7][8 -15][16-23][24-31] denoise [32,0-6] <--- this is seen in training


w/o kv-cache (tpe's cyclic shift will not happen) # w/o kv-cache 的第8帧是不带有前面0-7的信息的，让第8帧含有0-7的信息这件事情，是在write_latent_to_cache的时候发生的
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
        latent >> cond: [8][9-16][17-24][25-32] denoise [33-40]
        tpe    >> cond: [0][1- 8][9- 16][17-24] denoise [25-32]
    
    tpe's cyclic shift will only happen when max_cond_len + denoise_chunk_len > max_tpe_len (which is usually not suggested)

    TODO 比较一下 w/ kv-cache & w/o kv-cache 在推理的时候哪个更容易退化
'''

''' NOTE 结论
1）fixed tpe 无法使用w/ kv-cache， 在 累计生成的 frame 达到 max_tpe_len 的时候，会造成推理和训练不一致的情况， 
    - w/ kv-cache的时候，因为之前的cond frame 已经是cache 好了的，比如max_tpe_len=33, 最末尾一帧cond frame idx 是32 （那么他在write_kv_cache的时候对应的tpe idx 就是32）这时候，denoise 部分的chunk接在后面，比如用25~32 index的tpe（chunk_len=8)，这种组合的tpe，在训练的时候是没见过的 （其实这样很自然的就引出了我们应该用cyclc tpe），这样denoise 部分用0~7， 训练的时候random cyclic shift
    - 与之对应的，w/o kv-cache 的时候，conditon frame 的tpe每次auto-regre step 都是重新装填的。 

    换种说法：
    - w/ kv-cache 时， cached condition 与当时的tpe是一一绑定的，condition会一直取到 max_tpe 的末尾， 这时 denoise part 的tpe，就没法取了，不管取啥，都会和训练不一致（都是训练的时候没见过的tpe组合）。那么，如果可以说让前面的cond 在write kv-cache的时候，再往前挪一个chunk 的tpe？ 这样也是不对的，因为它之前还有很多cached condition，前面全要挪，这样就没法弄了。
    - 对于 w/o kv-cache， 每一步auto-regre 中condition 都是重新装填的，当生成到很长的视频的时候， condition 一直都是取前面的tpe， denoise part 是取尾部的tpe


2）如果采用 cyclic tpe， 我们也无法做到 w/ kv-cache 和 w/o kv-cache 一致的结果。 因为在w/o kv-cache  的时候，  tpe's cyclic shift ， 是不会发生的。
即，累计生成的 frame 达到 max_tpe_len 的时候：
    - w/ kv-cache ， tpe 会 cyclic shift， condition 是一直取到末尾的tpe， denoise 是头部的tpe，
    - w/o kv-cache, cyclic shift 不会发生， because tpe's cyclic shift will only happen when max_cond_len + denoise_chunk_len > max_tpe_len (which is usually not suggested)， 这里，model 只会一直把 cond 部分（dequeue掉之后剩下的部分）从tpe_idx=0开始装填， 然后 denoise 部分是 tpe 末尾部分
 
 TODO
  1）我们现在 cyclic tpe 中 tpe_max_len = 64, 如果max_cond_len=25, 即使 condition dequeue 发生之后的几步，  w/ kv-cache & w/o kv-cache 的 cond 部分与denoise 部分的tpe 还是一样的（即，cyclic shift 没有发生的时候）， 确认一下这个的推理， w/ kv-cache & w/o kv-cache  是否一致

  2） 当w/ kv-cache 的 cyclic shift 发生的时候， 比较一下 w/ kv-cache & w/o kv-cache 在推理的时候哪个更容易退化（即， 一直用 不shift的 tpe 是否会更好？）

  此外，对于causal-attn fixed tpe, 第二个 auto-regre 在开启 cf-attn的时候， w/ kv-cache & w/o kv-cache 的差别也比较大（与第一个chunk相差很多），也需要debug 一下。
'''
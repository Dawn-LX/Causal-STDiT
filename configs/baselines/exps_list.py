
exps_list0 = [
    # dict(
    #     exp_name = "causal_attn_cyclic_tpe",
    #     train_config = "/data/CausalSTDiT_working_dir/CausalSTDiT2-XL2_33x256x256ArSize8pp3_timelapse/training_config_backup.json2024-08-01T16-33-58.json",
    #     # ckpt_path = "/data/CausalSTDiT_working_dir/CausalSTDiT2-XL2_33x256x256ArSize8pp3_timelapse/epoch1-global_step13000",
    #     ckpt_path = "/data/CausalSTDiT_working_dir/CausalSTDiT2-XL2_33x256x256ArSize8pp3_timelapse/epoch1-global_step14000",
    # ),
    # dict(
    #     exp_name = "full_attn_cyclic_tpe",
    #     train_config = "/data/CausalSTDiT_working_dir/CausalSTDiT2-XL2_exp3_BaselineFullAttnCyclicTpe_33x256x256ArSize8pp3/training_config_backup.json",
    #     ckpt_path = "/data/CausalSTDiT_working_dir/CausalSTDiT2-XL2_exp3_BaselineFullAttnCyclicTpe_33x256x256ArSize8pp3/epoch1-global_step13000",
    # ),
    dict(
        exp_name = "causal_attn_fixed_tpe",
        train_config = "/data/CausalSTDiT_working_dir/CausalSTDiT2-XL2_exp2_BaselineCausalAttnFixedTpe_33x256x256ArSize8pp3/training_config_backup.json2024-08-04T21-43-51.json",
        ckpt_path = "/data/CausalSTDiT_working_dir/CausalSTDiT2-XL2_exp2_BaselineCausalAttnFixedTpe_33x256x256ArSize8pp3/epoch1-global_step13000",
    ),
    # dict(
    #     exp_name = "full_attn_fixed_tpe",
    #     train_config = "/data/CausalSTDiT_working_dir/CausalSTDiT2-XL2_BaselineFullAttnFixedTpe_33x256x256ArSize8pp3/training_config_backup.json2024-08-03T21-46-38.json",
    #     ckpt_path = "/data/CausalSTDiT_working_dir/CausalSTDiT2-XL2_BaselineFullAttnFixedTpe_33x256x256ArSize8pp3/epoch1-global_step10000",
    # )
]


exps_list = []
for cond_len in [25]:
# for cond_len in [9,17,25,33,41,49]:
    for i in range(len(exps_list0)):
        exp_info = exps_list0[i].copy()
        # global_step = exp_info["ckpt_path"].split("global_step")[-1]
        exp_name = exp_info["exp_name"]
        if "fixed_tpe" in exp_name:
            if cond_len > 25: continue
        exp_dir = f"/home/gkf/project/CausalSTDiT/working_dirSampleOutput/ablations/{exp_name}_CondLen{cond_len}"
        if "causal_attn" in exp_name:
            exp_info.update(
                exp_dir = exp_dir,
                enable_kv_cache = True,
                kv_cache_dequeue = True,
                kv_cache_max_seqlen = cond_len
            )
        else:
            exp_info.update(
                exp_dir = exp_dir,
                enable_kv_cache = False,
                max_condion_frames = cond_len
            )
        exps_list.append(exp_info)
    


del cond_len,exp_info,exp_name,exp_dir,exps_list0

# for x in exps_list:
#     print(x)
# print(len(exps_list))
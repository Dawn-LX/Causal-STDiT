import math
import random
from collections import OrderedDict

import torch

import colossalai
COLOSSALAI_VERSION = str(colossalai.__version__).split('.') # '0.4.0', '0.4.1', '0.4.2'
assert len(COLOSSALAI_VERSION)==3
COLOSSALAI_VERSION = float('.'.join(COLOSSALAI_VERSION[1:])) # 4.0, 4.1, 4.2

@torch.no_grad()
def update_ema(
    ema_model: torch.nn.Module, model: torch.nn.Module, optimizer=None, decay: float = 0.9999, sharded: bool = True
) -> None:
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        if name == "pos_embed":
            continue
        if param.requires_grad == False:
            continue
        if not sharded:
            param_data = param.data
            ema_params[name].mul_(decay).add_(param_data, alpha=1 - decay)
        else:
            if param.data.dtype != torch.float32:
                param_id = id(param)
                if COLOSSALAI_VERSION < 4.2:
                    master_param = optimizer._param_store.working_to_master_param[param_id] # for colossalai==0.4.0
                else:
                    master_param = optimizer.working_to_master_param[param_id] # for colossalai=0.4.2
                param_data = master_param.data
            else:
                param_data = param.data
            ema_params[name].mul_(decay).add_(param_data, alpha=1 - decay)


class MaskGenerator:
    def __init__(self, mask_ratios):
        valid_mask_names = [
            "mask_no",
            "mask_quarter_random",
            "mask_quarter_head",
            "mask_quarter_tail",
            "mask_quarter_head_tail",
            "mask_image_random",
            "mask_image_head",
            "mask_image_tail",
            "mask_image_head_tail",
        ]
        assert all(
            mask_name in valid_mask_names for mask_name in mask_ratios.keys()
        ), f"mask_name should be one of {valid_mask_names}, got {mask_ratios.keys()}"
        assert all(
            mask_ratio >= 0 for mask_ratio in mask_ratios.values()
        ), f"mask_ratio should be greater than or equal to 0, got {mask_ratios.values()}"
        assert all(
            mask_ratio <= 1 for mask_ratio in mask_ratios.values()
        ), f"mask_ratio should be less than or equal to 1, got {mask_ratios.values()}"
        # sum of mask_ratios should be 1
        assert math.isclose(
            sum(mask_ratios.values()), 1.0, abs_tol=1e-6
        ), f"sum of mask_ratios should be 1, got {sum(mask_ratios.values())}"
        print(f"mask ratios: {mask_ratios}")
        self.mask_ratios = mask_ratios

    def get_mask(self, x):
        mask_type = random.random()
        mask_name = None
        prob_acc = 0.0
        for mask, mask_ratio in self.mask_ratios.items():
            prob_acc += mask_ratio
            if mask_type < prob_acc:
                mask_name = mask
                break

        num_frames = x.shape[2]
        # Hardcoded condition_frames
        condition_frames_max = num_frames // 4

        mask = torch.ones(num_frames, dtype=torch.bool, device=x.device)
        if num_frames <= 1:
            return mask

        if mask_name == "mask_quarter_random":
            random_size = random.randint(1, condition_frames_max)
            random_pos = random.randint(0, x.shape[2] - random_size)
            mask[random_pos : random_pos + random_size] = 0
        elif mask_name == "mask_image_random":
            random_size = 1
            random_pos = random.randint(0, x.shape[2] - random_size)
            mask[random_pos : random_pos + random_size] = 0
        elif mask_name == "mask_quarter_head":
            random_size = random.randint(1, condition_frames_max)
            mask[:random_size] = 0
        elif mask_name == "mask_image_head":
            random_size = 1
            mask[:random_size] = 0
        elif mask_name == "mask_quarter_tail":
            random_size = random.randint(1, condition_frames_max)
            mask[-random_size:] = 0
        elif mask_name == "mask_image_tail":
            random_size = 1
            mask[-random_size:] = 0
        elif mask_name == "mask_quarter_head_tail":
            random_size = random.randint(1, condition_frames_max)
            mask[:random_size] = 0
            mask[-random_size:] = 0
        elif mask_name == "mask_image_head_tail":
            random_size = 1
            mask[:random_size] = 0
            mask[-random_size:] = 0

        return mask

    def get_masks(self, x):
        masks = []
        for _ in range(len(x)):
            mask = self.get_mask(x)
            masks.append(mask)
        masks = torch.stack(masks, dim=0)
        return masks


### noise prior


def build_progressive_noise(alpha, shape, start_noise):
    # shape: (b,c,f,h,w)
    # start_noise (b,c,1,h,w)
    
    noise = torch.randn(shape,device=start_noise.device,dtype=start_noise.dtype)
    if alpha > 0:
        prev_noise = start_noise # (b,c,1,h,w)
        progressive_noises = [prev_noise]
        for i in range(1,noise.shape[2]):
            new_noise = (alpha / math.sqrt(1+alpha**2)) * prev_noise + (1/math.sqrt(1+alpha**2)) * noise[:,:,1:i+1,:,:]
            progressive_noises.append(new_noise)
            prev_noise = new_noise
        progressive_noises = torch.cat(progressive_noises,dim=2) # (b,c,f,h,w)
        noise = progressive_noises
    return noise

### Prefix Frame (frame as prompt) utils:

def _get_prefix_len_choices(ar_size,max_length,n_given_frames=1):
    '''
    - ar_size: auto-regre window size (same at training & inference)
    - max_length: max frames of training samples
    - n_given_frames: n_given_frames at inference time (e.g., given the first frame at inference)
    '''
    prefix_len_choices = []
    accumulate_frames = n_given_frames
    while accumulate_frames < max_length:
        prefix_len_choices.append(accumulate_frames)
        accumulate_frames += ar_size
    
    return prefix_len_choices

class PrefixLenSampler:
    def __init__(self,ar_size,n_given_frames,sampling_strategy=None) -> None:
        self.ar_size = ar_size
        self.n_given_frames = n_given_frames
        self.prefix_len_choices = dict()
        self.sampling_strategy = sampling_strategy # TODO 
        # e.g., sample short prefix at early training epochs and longer prefix later
    
    def random_choose(self,max_len):
        if self.sampling_strategy is not None:
            raise NotImplementedError("TODO")

        if max_len in self.prefix_len_choices:
            pL_choices = self.prefix_len_choices[max_len]
        else:
            pL_choices = _get_prefix_len_choices(self.ar_size,max_len,self.n_given_frames)
            self.prefix_len_choices.update({max_len:pL_choices})
            print(f"update max_len: {max_len}:")
            print(f" >> prefix_len_choices: {pL_choices}")
            print(f" >> min denoise len: {max_len - pL_choices[-1]}")
        pL = random.choice(pL_choices)

        return pL
        
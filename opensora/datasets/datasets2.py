import os
from typing import Union, Dict, List, Optional
import random
import numpy as np
import torch
import decord
decord.bridge.set_bridge("torch")
import torchvision
# from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader

from opensora.registry import DATASETS

from opensora.datasets.utils import (
    load_jsonl,
    load_json
)
from opensora.datasets import video_transforms
# VID_EXTENSIONS, get_transforms_image, get_transforms_video, read_file, temporal_random_crop


def hit_condition(sample:dict,condition_ths:dict):
    if condition_ths is None:
        return True
    
    for k, th_or_bool in condition_ths.items(): # threshold or bool
        # by default, we discard sample that < threshold
        v = sample.get(k,th_or_bool) # if this `k` is not labeled, then we don't discard it
        cond = (v >= th_or_bool) if isinstance(th_or_bool,(int,float)) else (v==th_or_bool)

        if not cond:
            return False
        
        
    return True

@DATASETS.register_module()
class VideoTextDatasetFromJson(object):
    def __init__(
        self,
        video_paths: Union[str, List[str]],
        anno_jsons: Union[str, List[str]],
        n_sample_frames: int = 16,
        sample_interval: int = 3,
        allow_variable_len: bool = True, # this is deprecated, we always allow variable len and use zero-padding
        condition_ths: dict = None,
        aspect_ratio_buckets: tuple = ((256,256),(640,360),(360,640)), # TODO, currenlty we only use 256x256
        sample_repeats: int = 1, # for overft exps on several videos, set a large sample_repeats
        print_fn = print # this can be logger.info
    ) -> None:
        self.video_paths = video_paths.split(';') if isinstance(video_paths,str) else video_paths
        self.anno_jsons = anno_jsons.split(';') if isinstance(anno_jsons,str) else anno_jsons
        assert len(self.video_paths) == len(self.anno_jsons)

        self.n_sample_frames = n_sample_frames
        self.sample_interval = sample_interval
        self.temporal_sample = video_transforms.TemporalRandomCrop(n_sample_frames * sample_interval) # this is deprecated, refer to `temporal_random_sample`
        self.sample_repeats = sample_repeats

        _resolution = aspect_ratio_buckets[0]
        assert len(aspect_ratio_buckets) == 1 and _resolution[0] == _resolution[1], "TODO consider multi-resolution"
        self.image_size = _resolution
        self.transforms = torchvision.transforms.Compose([
            video_transforms.ToTensorVideo(), # TCHW
            video_transforms.RandomHorizontalFlipVideo(),
            video_transforms.ResizeCenterCropVideo(_resolution[0]), # TODO define a ResizeCenterCrop()
            torchvision.transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5],inplace=True) # to -1 ~ 1
        ])

        self.target_resolution = _resolution
        self.print_fn = print_fn
        self.condition_ths = condition_ths
        assert condition_ths is None or all(isinstance(v,(int,float,bool)) for v in condition_ths.values())

        print_fn(f"load and filter annotations, with condition_ths = {self.condition_ths}")
        self.annotations = []
        total_before_filter = 0
        for path, fn in zip(self.video_paths,self.anno_jsons):
            annos_per_dataset = load_jsonl(fn)
            total_before_filter += len(annos_per_dataset)
            _num_keep = 0
            for sample in annos_per_dataset:
                sample["video_fn"] = os.path.join(path,sample["video_fn"])
                if condition_ths is None or hit_condition(sample,condition_ths):
                    self.annotations.append(sample)
                    _num_keep +=1
            if _num_keep == 0:
                _info = "WARNING: no sample meets the conditions, "
                _info += f"dataset: {fn}, with condition_ths: {condition_ths},"
                _info += "make sure the keys in condition_ths match keys in the json file"
                print_fn(_info)
        assert len(self.annotations) > 0
        _ratio = len(self.annotations) / total_before_filter
        print_fn(f"total_before_filter={total_before_filter}, num_samples_keep:{len(self.annotations)}, ratio={_ratio}")
    
    def __len__(self):
        return len(self.annotations) * self.sample_repeats
    
    
    def temporal_random_sample(self,total_frames):
        
        sample_span_size = self.n_sample_frames * self.sample_interval

        rand_end = max(0, total_frames - sample_span_size - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + sample_span_size, total_frames)


        frame_indices = list(range(
            begin_index, end_index, self.sample_interval
        ))[:self.n_sample_frames]

        return frame_indices
    
    def _get_random_frame_indices(self,total_frames):
        '''this func is deprecated, we use self.temporal_random_sample directly
        '''
        start_fid,end_fid = self.temporal_sample(total_frames)
        frame_indices = list(range(
            start_fid, end_fid, self.sample_interval
        ))[:self.n_sample_frames]

        return frame_indices

    def __getitem__(self,index):
        anno = self.annotations[index % len(self.annotations)]
        video_path = anno["video_fn"]
        anno_frames = anno["length"]

        vr = decord.VideoReader(video_path) # 
        # assert anno_frames == len(vr) # ideally this should be `True`, but there maybe some special case ?
        total_frames = min(anno_frames,len(vr))

        frame_indices = self.temporal_random_sample(total_frames)
        video: torch.Tensor = vr.get_batch(frame_indices) #  (T, H, W, C)
        video = video.permute(0, 3, 1, 2) #  (T, C, H, W)
        video = self.transforms(video) # (T, C, H, W)
        actual_length = len(video)
        assert actual_length >=2, "please filter out short videos ahead"
        
        # padding
        if actual_length < self.n_sample_frames:
            # NOTE: zero-padding at the end of video tensor is only suitable for causal temporal attention,
            # Otherwise the feature computation is wrong
            pad_shape = (self.n_sample_frames - actual_length,) + video.shape[1:]
            video = torch.cat([video,torch.zeros(pad_shape)],dim=0)
        
        text = anno["prompt"]
        sample = dict(
            text = text,
            video = video.permute(1,0,2,3), # TCHW -> CTHW
            actual_length = actual_length
        )

        return sample


@DATASETS.register_module()
class VideoDatasetForVal(VideoTextDatasetFromJson):
    def __init__(
        self,
        read_video = True,
        read_first_frame = True,
        **kwargs
    ):
        self.read_video = read_video
        self.read_first_frame = read_first_frame
        super().__init__(**kwargs)
    
    def __len__(self):
        assert self.sample_repeats == 1
        return len(self.annotations)
    
    def __getitem__(self,index):
        anno = self.annotations[index % len(self.annotations)]
        video_path = anno["video_fn"]
        anno_frames = anno["length"]

        text = anno["prompt"]
        sample = {"text":text}

        if self.read_video or self.read_first_frame:
            vr = decord.VideoReader(video_path) # 
            # assert anno_frames == len(vr) # ideally this should be `True`, but there maybe some special case ?
            total_frames = min(anno_frames,len(vr))

            if self.read_first_frame:
                first_frame: torch.Tensor = vr.get_batch([0]) #  (1, H, W, C)
                first_frame = first_frame.permute(0, 3, 1, 2) #  (1, C, H, W)
                first_frame = self.transforms(first_frame) # (1, C, H, W)
                assert first_frame.shape[0] == 1
                sample.update({
                    "first_frame":first_frame.permute(1,0,2,3), # TCHW -> CTHW
                    "actual_length": 1
                })
            
            if self.read_video:
                frame_indices = list(range(
                    0, total_frames, self.sample_interval
                ))[:self.n_sample_frames]
            
                video: torch.Tensor = vr.get_batch(frame_indices) #  (T, H, W, C)
                video = video.permute(0, 3, 1, 2) #  (T, C, H, W)
                video = self.transforms(video) # (T, C, H, W)
                actual_length = len(video)
                assert actual_length > 0
        
                # padding
                if actual_length < self.n_sample_frames:
                    # NOTE: zero-padding at the end of video tensor is only suitable for causal temporal attention,
                    # Otherwise the feature computation is wrong
                    pad_shape = (self.n_sample_frames - actual_length,) + video.shape[1:]
                    video = torch.cat([video,torch.zeros(pad_shape)],dim=0)

                sample.update({
                    "video":video.permute(1,0,2,3), # TCHW -> CTHW
                    "actual_length": actual_length
                })

        return sample
    

def dataloader_demo():
    from torch.utils.data import DataLoader
    
    condition_ths = None #dict()
    dataset = VideoDataset(
        video_paths="/home/gkf/project/VidVRD_VidOR/vidor-dataset/val_videos/0001",
        anno_jsons="/home/gkf/project/CausalSTDiT/assets/video_anno_demo.jsonl",
        n_sample_frames=16,
        sample_interval=3,
        condition_ths=condition_ths,
        aspect_ratio_buckets=[(256,256)],
        sample_repeats=100,
    )
    
    for i in range(len(dataset)):
        data = dataset[i]
        text = data["text"]
        video = data["video"]
        actual_length = data["actual_length"]
        print(i,text,video.shape,actual_length)
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        drop_last=False
    )
    for data in dataloader:
        text = data["text"]
        video = data["video"]
        actual_length = data["actual_length"]
        print(i,text,video.shape,actual_length)

if __name__ == "__main__":
    dataloader_demo()
    '''
    export PYTHONPATH=$PYTHONPATH:/home/gkf/project/CausalSTDiT/opensora
    '''
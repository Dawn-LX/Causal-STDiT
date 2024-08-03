import numpy as np
import random
import torch
import torchvision
from PIL import Image
import os
import os.path
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

from opensora.datasets import video_transforms
from opensora.registry import DATASETS

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    '''
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
    '''
    Im = Image.open(path)
    return Im.convert('RGB')


def make_dataset(dir, nframes, class_to_idx):
    images = []
    n_video = 0 # long-video
    n_clip = 0 # short-video
    for target in sorted(os.listdir(dir)):
        if os.path.isdir(os.path.join(dir,target))==True:
            n_video +=1
            # eg: dir + '/rM7aPu9WV2Q'
            subfolder_path = os.path.join(dir, target) 
            for subsubfold in sorted(os.listdir(subfolder_path) ):
                if os.path.isdir(os.path.join(subfolder_path, subsubfold) ):
                	# eg: dir + '/rM7aPu9WV2Q/1'
                    n_clip += 1
                    subsubfolder_path = os.path.join(subfolder_path, subsubfold) 
                    
                    item_frames = []
                    i = 1
                    for fi in sorted( os.listdir(subsubfolder_path) ):
                        if  is_image_file(fi):
                        # fi is an image in the subsubfolder
                            file_name = fi
                            # eg: dir + '/rM7aPu9WV2Q/1/rM7aPu9WV2Q_frames_00086552.jpg'
                            file_path = os.path.join(subsubfolder_path,file_name) 
                            item = (file_path, class_to_idx[target])
                            item_frames.append(item)
                            if i %nframes == 0 and i >0 :
                                images.append(item_frames) # item_frames is a list containing n frames. 
                                item_frames = []
                            i = i+1
    print('number of long videos:')
    print(n_video)
    print('number of short videos')
    print(n_clip)
    print(f"cut with {nframes} each, number of clips: {len(images)}")
    return images

@DATASETS.register_module()
class SkyTimelapseDataset:
    def __init__(
        self, 
        root, # e.g., 
        n_sample_frames,  
        image_size=(128,128), # (h,w)
        unified_prompt = "a beautiful sky timelapse",
        loader=pil_loader,
        print_fn = print
    ):
        self.print_fn = print_fn
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, n_sample_frames,  class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + 
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        # e.g., 
        # /data/SkyTimelapse/sky_timelapse/sky_timelapse/sky_train
        # /data/SkyTimelapse/sky_timelapse/sky_timelapse/sky_test
        self.split = root.split('/')[-1].split('_')[-1]

        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.loader = loader
        self.n_sample_frames = n_sample_frames
        self.unified_prompt = unified_prompt


        long_vid_labels = classes
        short_vid_labels = []
        for long_vid_dir in long_vid_labels:
            short_vids = os.listdir(os.path.join(root,long_vid_dir))
            # print(short_vids)
            short_vids = [
                sv for sv in short_vids 
                if os.path.isdir(os.path.join(root,long_vid_dir,sv))
            ]
            short_vid_labels.extend(short_vids)
        self.long_vid_labels = long_vid_labels # 997 for train set
        self.short_vid_labels = short_vid_labels # 2392 for train set

        self.image_size = image_size
        self.transforms = torchvision.transforms.Compose([
            video_transforms.ToTensorVideo(), # TCHW
            video_transforms.ResizeCenterCropVideo(image_size),
            torchvision.transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5],inplace=True) # to -1 ~ 1
        ])
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        # clip is a list of 32 frames 
        clip = self.imgs[index] 
        img_clip = []
        for frame in clip:
            path, target = frame
            img = self.loader(path) 
            img = torch.from_numpy(np.array(img)) # (H, W, C)
            img_clip.append(img) 
        video = torch.stack(img_clip,dim=0) # (T, H, W, C)
        video = video.permute(0,3,1,2) # TCHW
        video = self.transforms(video) # TCHW


        if self.unified_prompt is None:
            text = anno["prompt"] # TODO
        else:
            text = self.unified_prompt

        sample = dict(
            text =  text,
            video = video.permute(1,0,2,3), # TCHW -> CTHW
            actual_length = video.shape[0]
        )

        return sample

@DATASETS.register_module()
class SkyTimelapseDatasetForEvalFVD(SkyTimelapseDataset):
    def __init__(
        self,
        read_video = True,
        read_first_frame = True,
        class_balance_sample = True,
        long_vid_as_class = True,
        num_samples_total = None,
        num_samples_per_class = None,
        allow_short_video = False, # allow videos shorter than n_sample_frames
        **kwargs
    ):
        
        super().__init__(**kwargs)

        self.read_video = read_video
        self.read_first_frame = read_first_frame
        self.allow_short_video = allow_short_video

        self.class_balance_sample = class_balance_sample
        if class_balance_sample:
            self.long_vid_as_class = long_vid_as_class
            num_classes = len(self.long_vid_labels) if self.long_vid_as_class else len(self.short_vid_labels)
            assert (num_samples_total is None) or (num_samples_per_class is None)
            assert (num_samples_total,num_samples_per_class) != (None,None)
            if num_samples_per_class is not None:
                self.num_samples_total = num_classes * num_samples_per_class
            else:
                self.num_samples_total = num_samples_total
            

            del self.imgs 
            imgs = make_dataset(self.root, 1,  self.class_to_idx)
            imgs = [x[0] for x in imgs]

            from collections import defaultdict
            
            longvid_to_img_paths = defaultdict(list)
            shortvid_to_img_paths = defaultdict(list)
            for img_path,class_id in imgs:
                long_vid_label, short_vid_label,filename = img_path.split('/')[-3:]
                # 07U1fSrk9oI 07U1fSrk9oI_1 07U1fSrk9oI_frames_00000046.jpg
                assert long_vid_label == filename.split('_frames_')[0], f"{long_vid_label},{filename}"
                cls_id = self.class_to_idx[long_vid_label]
                
                longvid_to_img_paths[long_vid_label].append(img_path)
                shortvid_to_img_paths[short_vid_label].append(img_path)

            
            if allow_short_video:
                longvid_to_img_paths = {k:sorted(v) for k,v in longvid_to_img_paths.items() if len(v) >=2}
                shortvid_to_img_paths = {k:sorted(v) for k,v in shortvid_to_img_paths.items() if len(v) >= 2}
            else:
                longvid_to_img_paths = {k:sorted(v) for k,v in longvid_to_img_paths.items() if len(v) >= self.n_sample_frames}
                shortvid_to_img_paths = {k:sorted(v) for k,v in shortvid_to_img_paths.items() if len(v) >= self.n_sample_frames}

            self.print_fn(f"use class_balance_sample, num_samples_total={self.num_samples_total}, n_sample_frames = {self.n_sample_frames}")
            if self.long_vid_as_class:
                if len(longvid_to_img_paths) < len(self.long_vid_labels):
                    self.print_fn(f"len(longvid_to_img_paths)={len(longvid_to_img_paths)} < num_long_video={len(self.long_vid_labels)}")
            else:
                if len(shortvid_to_img_paths) < len(self.short_vid_labels):
                    self.print_fn(f"len(shortvid_to_img_paths)={len(shortvid_to_img_paths)} < num_short_video={len(self.short_vid_labels)}")
            
            self.longvid_to_img_paths =  longvid_to_img_paths
            self.shortvid_to_img_paths =  shortvid_to_img_paths

    def __len__(self):
        if self.class_balance_sample:
            return self.num_samples_total
        else:
            return super().__len__(self)
    

    def __getitem__(self, index):
        if not self.class_balance_sample:
            clip = self.imgs[index] 
            clip = sorted(clip,key=lambda x:x[0]) # sort by path
            clip_img_paths = [x[0] for x in clip]
        else:
            '''
            follow the style-GAN-V paper, refer to its Appendix
            '''
            class2idx = self.long_vid_labels if self.long_vid_as_class else self.short_vid_labels
            num_classes = len(class2idx)
            class2paths = self.longvid_to_img_paths if self.long_vid_as_class else self.shortvid_to_img_paths
            class_name = class2idx[index % num_classes]
            _i=1
            while class_name not in class2paths.keys():
                # maybe len(class2paths) < num_classes
                index = random.choice(range(len(self)))
                _i+=1
                class_name = class2idx[index % num_classes]
                if _i > 10000:
                    assert False, f"too many bad data: class_name={class_name}"
            
            img_paths = class2paths[class_name]
            assert len(img_paths) >= (2 if self.allow_short_video else self.n_sample_frames)
            valid_start_ids = range(0,len(img_paths)-self.n_sample_frames)

            if len(valid_start_ids) > 0:
                random_start = random.choice(valid_start_ids)
                clip_img_paths = img_paths[random_start:random_start+self.n_sample_frames]
            else:
                clip_img_paths = img_paths

        _1st_path = clip_img_paths[0]
        long_vid_label, short_vid_label,filename = _1st_path.split('/')[-3:]
        # e.g., 07U1fSrk9oI 07U1fSrk9oI_1 07U1fSrk9oI_frames_00000046.jpg
        video_name = f"{index:05d}_{filename}.mp4"

        if self.unified_prompt is None:
            text = anno["prompt"] # TODO
        else:
            text = self.unified_prompt
        
        sample = dict(
            video_name = video_name,
            text =  text,
        )
        if self.read_first_frame:
            first_frame = torchvision.io.read_image(_1st_path,torchvision.io.ImageReadMode.RGB) # (3,h,w)
            first_frame = first_frame.unsqueeze(0) # (1, C, H, W)
            first_frame = self.transforms(first_frame) # TCHW
            sample.update({
                "first_frame":first_frame.permute(1,0,2,3), # TCHW -> CTHW
                "actual_length": 1
            })

        if self.read_video:
            img_clip = []
            for path in clip_img_paths:
                img = self.loader(path) 
                img = torch.from_numpy(np.array(img)) # (H, W, C)
                img_clip.append(img) 
            video = torch.stack(img_clip,dim=0) # (T, H, W, C)
            video = video.permute(0,3,1,2) # TCHW
            video = self.transforms(video) # TCHW
            
            sample.update({
                "video":video.permute(1,0,2,3), # TCHW -> CTHW
                "actual_length": video.shape[0]
            })

        assert not self.allow_short_video, "TODO, do zero-padding for the stack operation in collate_fn"

        return sample
    

@DATASETS.register_module()
class SkyTimelapseDatasetEvalImg2Vid_old:
    def __init__(
        self, 
        root,
        n_sample_frames,
        image_size = (128, 128),
        loader=pil_loader,
        print_fn = print
    ):
        
        classes, class_to_idx = find_classes(root)
        self.root = root
        self.split = root.split('/')[-1].split('_')[-1]
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.loader = loader
        self.n_sample_frames = n_sample_frames

        long_vid_labels = classes

        short_vid_labels = []
        for long_vid_dir in long_vid_labels:
            short_vids = os.listdir(os.path.join(root,long_vid_dir))
            # print(short_vids)
            short_vids = [
                sv for sv in short_vids 
                if os.path.isdir(os.path.join(root,long_vid_dir,sv))
            ]
            short_vid_labels.extend(short_vids)
        self.long_vid_labels = long_vid_labels # 111 for test set
        self.short_vid_labels = short_vid_labels # 225 for test set

        
        image_infos = make_dataset(root,1,class_to_idx)
        image_infos = [x[0] for x in image_infos]

        short_vid_to_1st_frame_map = {label:[] for label in short_vid_labels}
        for img_path,class_id in image_infos:
            long_vid_label, short_vid_label,filename = img_path.split('/')[-3:]
            # 07U1fSrk9oI 07U1fSrk9oI_1 07U1fSrk9oI_frames_00000046.jpg

            short_vid_to_1st_frame_map[short_vid_label].append(img_path)
        # IlMFL8RxgeY_7
        short_vid_to_1st_frame_map = {k:v for k,v in short_vid_to_1st_frame_map.items() if len(v) > 0}
        short_vid_to_1st_frame_map = {k:sorted(v)[0] for k,v in short_vid_to_1st_frame_map.items()}
        self.short_vid_to_1st_frame_map = short_vid_to_1st_frame_map

        self.image_size = image_size
        self.transforms = torchvision.transforms.Compose([
            video_transforms.ToTensorVideo(), # TCHW
            video_transforms.ResizeCenterCropVideo(image_size),
            torchvision.transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5],inplace=True) # to -1 ~ 1
        ])
        
        print_fn(f"SkyTimelapseDatasetEvalImg2Vid is built, containing {len(self)} short videos with 1st frames")


    def __len__(self):
        return len(self.short_vid_to_1st_frame_map)

    def __getitem__(self,idx):
        
        first_frame_path = self.short_vid_to_1st_frame_map[idx]
        img = self.loader(first_frame_path) 
        img = torch.from_numpy(np.array(img)).unsqueeze(0) # (1, H, W, C)
        
        img = img.permute(0,3,1,2) # TCHW
        img = self.transforms(img) # TCHW

        sample = {
            "first_frame":img.permute(1,0,2,3), # TCHW -> CTHW
            "actual_length": 1
        }
        return sample

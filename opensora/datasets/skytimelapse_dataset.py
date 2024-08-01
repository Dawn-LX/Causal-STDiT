import numpy as np
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
            video_transforms.RandomHorizontalFlipVideo(),
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

    
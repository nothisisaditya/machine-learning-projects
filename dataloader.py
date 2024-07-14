import torch
import torchvision
import torch.utils.data

import nvidia.dali
import nvidia.dali.types as types
from nvidia.dali import fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator

import os


class Dataset:
    def __init__(self,
                 data_dir: str,
                 batch_size: int,
                 size: int = 224,
                 val_batch_size: int = None,
                 val_size: int = None,
                 min_crop_size: float = 0.08,
                 workers: int = 2,
                 cuda: bool = True,
                 disable_dali: bool = False,
                 mean: tuple = (0.485 * 255, 0.456 * 255, 0.406 * 255),
                 std: tuple = (0.229 * 255, 0.224 * 255, 0.225 * 255),
                 pin_memory: bool = True):
        self.batch_size = batch_size
        self.size = size
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        self.val_size = val_size if val_size is not None else size
        self.min_crop_size = min_crop_size
        self.workers = workers
        self.cuda = cuda
        self.disable_dali = disable_dali
        self.mean = mean
        self.std = std
        self.pin_memory = pin_memory

        self.train_dir = os.path.join(data_dir, 'train')
        self.val_dir = os.path.join(data_dir, 'val')

        if self.disable_dali:
            print('Using torchvision')
            self._build_torchvision_pipeline()
        else:
            print('Using DALI')
            self._build_dali_pipeline()

    def _build_torchvision_pipeline(self):
        self.train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomResizedCrop(self.size, scale=(self.min_crop_size, 1.0)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.Normalize(mean=self.mean, std=self.std)])

        self.val_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(self.val_size),
            torchvision.transforms.CenterCrop(self.size)])

        self.train_dataset = torchvision.datasets.ImageFolder(self.train_dir,
                                                              self.train_transforms)
        self.val_dataset = torchvision.datasets.ImageFolder(self.val_dir,
                                                            self.train_transforms)

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                        batch_size=self.batch_size,
                                                        shuffle=True,
                                                        num_workers=self.workers,
                                                        pin_memory=self.pin_memory)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset,
                                                      batch_size=self.batch_size,
                                                      shuffle=False,
                                                      num_workers=self.workers,
                                                      pin_memory=self.pin_memory)

    @nvidia.dali.pipeline_def
    def _dali_pipeline(self, dir):
        images, label = fn.readers.file(file_root=dir, device='cpu',
                                        random_shuffle=True, name='data_reader')
        images = fn.decoders.image_random_crop(images,
                                               device='mixed',
                                               output_type=types.RGB,
                                               preallocate_width_hint=250,
                                               preallocate_height_hint=250,
                                               random_aspect_ratio=[0.8, 1.25],
                                               random_area=[0.1, 1.0],
                                               num_attempts=10)
        images = fn.resize(images,
                           device='gpu',
                           resize_x=self.size,
                           resize_y=self.size,
                           interp_type=types.INTERP_LINEAR,
                           antialias=True)
        mirror = fn.random.coin_flip(probability=0.5)
        images = fn.crop_mirror_normalize(images.gpu(),
                                          dtype=types.FLOAT,
                                          output_layout="CHW",
                                          crop=(self.size, self.size),
                                          mean=self.mean,
                                          std=self.std,
                                          mirror=mirror)
        label_gpu = label.gpu()

        return images, label_gpu

    def _build_dali_pipeline(self):
        train_pipe = self._dali_pipeline(self.train_dir, batch_size=self.batch_size,
                                         num_threads=self.workers, device_id=0)
        val_pipe = self._dali_pipeline(self.val_dir, batch_size=self.val_batch_size,
                                       num_threads=self.workers, device_id=0)
        train_pipe.build()
        val_pipe.build()
        self.train_loader = DALIGenericIterator(
            train_pipe, ['data', 'label'],
            reader_name='data_reader'
        )
        self.val_loader = DALIGenericIterator(
            val_pipe, ['data', 'label'],
            reader_name='data_reader'
        )

    def get_train_loader(self):
        return self.train_loader

    def get_val_loader(self):
        return self.val_loader

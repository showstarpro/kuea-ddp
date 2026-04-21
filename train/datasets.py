import json

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import webdataset as wds


class COCOFlickrDataset(Dataset):
    def __init__(
            self,
            image_dir_path,
            annotations_path,
            transform=None,
            is_flickr=False,
            prefix=None,
    ):
        self.image_dir_path = image_dir_path
        self.annotations = json.load(open(annotations_path))["annotations"]
        self.is_flickr = is_flickr
        self.transform = transform
        self.prefix = prefix

    def __len__(self):
        return len(self.annotations)

    def get_img_path(self, idx):
        if self.is_flickr:
            return f"{self.image_dir_path}/{self.annotations[idx]['image_id']}.jpg"
        else:
            return f"{self.image_dir_path}/{self.prefix}{self.annotations[idx]['image_id']:012d}.jpg"

    def __getitem__(self, idx):
        image = Image.open(self.get_img_path(idx))
        caption = self.annotations[idx]["caption"]
        return self.transform(image), caption


class ImageNetDataset(ImageFolder):
    """Class to represent the ImageNet1k dataset."""

    def __init__(self, root, **kwargs):
        super().__init__(root=root, **kwargs)

    def __getitem__(self, idx):
        sample, target = super().__getitem__(idx)
        # target_label = IMAGENET_1K_CLASS_ID_TO_LABEL[target]
        return sample, target


def make_wds_dataset(shards_path, transform, batch_size, steps_per_gpu, tokenizer):
    dataset = (
        wds.WebDataset(
            shards_path,
            shardshuffle=True,
            nodesplitter=wds.split_by_node,
            resampled=True,
        )
        .shuffle(1000)
        .decode("pil")
        .rename(image="jpg;png;jpeg;webp", text="txt")
        .map_dict(
            image=transform,
            text=lambda t: tokenizer(t)[0]  # [0] 去掉 batch 维，shape: (seq_len,)
        )
        .to_tuple("image", "text")
        .batched(batch_size, partial=False)
        .with_epoch(steps_per_gpu)
    )
    return dataset
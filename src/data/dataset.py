import os
import json
from collections import Counter
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms
from .tokenizer import SimpleTokenizer

class FireDataset(Dataset):
    def __init__(self, data_root, transform=None, tokenizer=None, mode="all", split_regions=None):
        """
        Args:
            data_root (str): Path to the data directory.
            transform (callable, optional): Optional transform to be applied on a sample.
            tokenizer (SimpleTokenizer, optional): Tokenizer for text.
            mode (str): 'all', 'train', 'val', or 'test'. 
            split_regions (list): List of region names to include. If None, include all.
        """
        self.data_root = data_root
        self.transform = transform
        self.tokenizer = tokenizer
        self.samples = []
        self.mode = mode

        self.img_dir = os.path.join(data_root, "image")
        
        # Define label mapping based on folder names
        # 5 classes (excluding inaccessible)
        self.label_map = {
            "no_damage": 0,
            "minor": 1,
            "major": 2,
            "destroyed": 3,
            "affected": 4
        }

        self.split_regions = split_regions
        self.annotations = self._load_annotations()
        self._load_data()

    def _load_annotations(self):
        anno_path = os.path.join(self.data_root, "annotations.jsonl")
        annotations = {}
        if os.path.exists(anno_path):
            with open(anno_path, 'r') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        annotations[item['image_id']] = item
                    except json.JSONDecodeError:
                        continue
        else:
            print(f"Warning: Annotations file not found at {anno_path}")
        return annotations

    def _load_data(self):
        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")

        print(f"Scanning images in {self.img_dir} for mode {self.mode}...")
        
        # Walk through the directory structure
        # Structure: data/image/{REGION}/train/{CLASS}/{image}
        
        # Get all regions first
        all_regions = [d for d in os.listdir(self.img_dir) if os.path.isdir(os.path.join(self.img_dir, d))]
        
        for region in all_regions:
            # Filter by split_regions if provided
            if self.split_regions is not None and region not in self.split_regions:
                continue
                
            region_path = os.path.join(self.img_dir, region, "train")
            if not os.path.exists(region_path):
                continue
                
            for class_name in os.listdir(region_path):
                if class_name == "inaccessible":
                    continue
                if class_name in self.label_map:
                    label = self.label_map[class_name]
                    class_path = os.path.join(region_path, class_name)
                    
                    for filename in os.listdir(class_path):
                        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                            img_path = os.path.join(class_path, filename)
                            image_id = os.path.splitext(filename)[0]
                            anno = self.annotations.get(image_id, {})
                            
                            self.samples.append(
                                {
                                    "image_path": img_path,
                                    "label": label,
                                    "coarse_text": anno.get("inference_coarse", ""), 
                                    "fine_text": anno.get("inference_fine", ""),   
                                    "region": region
                                }
                            )

        print(f"Loaded {len(self.samples)} samples for mode {self.mode}")
        if len(self.samples) == 0:
            print(f"Warning: No samples found for mode {self.mode}!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load Image
        try:
            image = Image.open(sample["image_path"]).convert("RGB")
        except Exception as e:
            print(f"Error loading image {sample['image_path']}: {e}")
            image = Image.new("RGB", (224, 224))

        if self.transform:
            image = self.transform(image)

        # Tokenize Text (Empty)
        coarse_tokens = (
            self.tokenizer.encode(sample["coarse_text"])
            if self.tokenizer
            else torch.tensor([])
        )
        fine_tokens = (
            self.tokenizer.encode(sample["fine_text"])
            if self.tokenizer
            else torch.tensor([])
        )

        return image, sample["label"], coarse_tokens, fine_tokens


def create_dataloaders(config):
    """
    Creates train, val, test dataloaders based on config.
    """
    data_path = config["data"]["path"]
    batch_size = config["data"]["batch_size"]
    num_workers = config["data"].get("num_workers", 4)
    vocab_size = config["data"]["vocab_size"]
    seq_len = config["data"]["seq_len"]

    # Transforms
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Tokenizer
    tokenizer = SimpleTokenizer(vocab_size=vocab_size, seq_len=seq_len)
    tokenizer.build_vocab([]) 

    # Define Region Splits
    # Based on analysis to ensure distribution balance and presence of all classes
    train_regions = ['GLA', 'VAL', 'MOS', 'CAS', 'MIL', 'POST', 'CRE', 'DIN', 'MON', 'SCU', 'FOR', 'BEU']
    val_regions = ['BEA', 'AUG', 'ZOG']
    test_regions = ['CAL', 'FAI', 'DIX']

    print(f"Splitting by Region:")
    print(f"Train Regions: {train_regions}")
    print(f"Val Regions: {val_regions}")
    print(f"Test Regions: {test_regions}")

    # Create Datasets per split
    train_dataset = FireDataset(data_path, transform=train_transform, tokenizer=tokenizer, mode="train", split_regions=train_regions)
    val_dataset = FireDataset(data_path, transform=val_transform, tokenizer=tokenizer, mode="val", split_regions=val_regions)
    test_dataset = FireDataset(data_path, transform=val_transform, tokenizer=tokenizer, mode="test", split_regions=test_regions)

    # Implement WeightedRandomSampler for Training to handle class imbalance
    print("Calculating class weights for WeightedRandomSampler...")
    train_labels = [s["label"] for s in train_dataset.samples]
    class_counts = Counter(train_labels)
    print(f"Train Class Counts: {dict(class_counts)}")
    
    # Calculate weight for each class: w = 1 / count
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    
    # Assign weight to each sample
    sample_weights = [class_weights[label] for label in train_labels]
    sample_weights = torch.DoubleTensor(sample_weights)
    
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, tokenizer

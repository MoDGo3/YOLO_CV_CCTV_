#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=============================================================================
FairFace Dataset Downloader
=============================================================================
สคริปต์สำหรับดาวน์โหลด FairFace Dataset จาก HuggingFace
ใช้สำหรับทดสอบ fairness ของระบบจำแนก demographics

FairFace Dataset Info:
- ขนาด: 108,501 รูปภาพ (balanced 7 races)
- Race Classes: White, Black, Indian, East Asian, Southeast Asian, Middle Eastern, Latino
- Gender: Male, Female
- Age Groups: 9 groups (0-2, 3-9, 10-19, ... 60-69, 70+)
- License: CC-BY-4.0

วิธีใช้:
    python benchmark/download_fairface.py [--split train|test|val] [--size large|small]

ตัวอย่าง:
    python benchmark/download_fairface.py --split train --size small
=============================================================================
"""

import os
import sys
import json
import shutil
from pathlib import Path
from argparse import ArgumentParser

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("[WARNING] datasets library not available. Install with: pip install datasets")


def download_fairface(split: str = "train", max_images: int = None, output_dir: str = None):
    """
    Download FairFace dataset from HuggingFace
    
    Args:
        split: 'train', 'test', or 'val'
        max_images: Maximum number of images to download (None = all)
        output_dir: Output directory (default: benchmark/fairface_images/)
    """
    
    if not DATASETS_AVAILABLE:
        print("[ERROR] Please install datasets: pip install datasets")
        return False
    
    print("\n" + "="*70)
    print(f"[INFO] Downloading FairFace Dataset ({split} split)...")
    print("="*70)
    
    # Set output directory
    if output_dir is None:
        output_dir = Path(__file__).parent / "fairface_images"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Output directory: {split_dir}")
    
    # Load dataset
    try:
        print(f"[INFO] Loading FairFace dataset (split='{split}')...")
        # FairFace dataset requires version specification
        dataset = load_dataset('HuggingFaceM4/FairFace', '1.25', split=split, streaming=False)
        print(f"[OK] Dataset loaded. Total images: {len(dataset)}")
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        return False
    
    # Limit images if specified
    if max_images:
        dataset = dataset.select(range(min(max_images, len(dataset))))
        print(f"[INFO] Limiting to {len(dataset)} images")
    
    # Download and organize images
    print(f"\n[INFO] Downloading {len(dataset)} images...")
    
    # Create metadata file
    metadata = {
        "split": split,
        "total_images": len(dataset),
        "races": {},
        "genders": {},
        "images": []
    }
    
    for idx, sample in enumerate(dataset):
        if (idx + 1) % 100 == 0:
            print(f"  Progress: {idx + 1}/{len(dataset)}")
        
        try:
            # Extract labels
            image = sample['image']
            race = sample['race']  # 0=East Asian, 1=Indian, 2=Black, 3=White, 4=Middle Eastern, 5=Latino, 6=Southeast Asian
            gender = sample['gender']  # 0=Male, 1=Female
            age = sample['age']  # 0-8 (age groups)
            
            # Map to string labels
            race_map = {
                0: "East Asian",
                1: "Indian", 
                2: "Black",
                3: "White",
                4: "Middle Eastern",
                5: "Latino_Hispanic",
                6: "Southeast Asian"
            }
            
            gender_map = {0: "Male", 1: "Female"}
            
            age_map = {
                0: "0-2", 1: "3-9", 2: "10-19", 3: "20-29",
                4: "30-39", 5: "40-49", 6: "50-59", 7: "60-69", 8: "70+"
            }
            
            race_name = race_map.get(race, "Unknown")
            gender_name = gender_map.get(gender, "Unknown")
            age_name = age_map.get(age, "Unknown")
            
            # Create filename: race_gender_age_index.jpg
            filename = f"{race_name}_{gender_name}_{age_name}_{idx:06d}.jpg"
            filepath = split_dir / filename
            
            # Save image
            image.save(filepath)
            
            # Update metadata
            if race_name not in metadata["races"]:
                metadata["races"][race_name] = 0
            metadata["races"][race_name] += 1
            
            if gender_name not in metadata["genders"]:
                metadata["genders"][gender_name] = 0
            metadata["genders"][gender_name] += 1
            
            metadata["images"].append({
                "filename": filename,
                "race": race_name,
                "gender": gender_name,
                "age_group": age_name
            })
            
        except Exception as e:
            print(f"  [WARNING] Error processing image {idx}: {e}")
            continue
    
    # Save metadata
    metadata_path = split_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n[OK] Downloaded {len(metadata['images'])} images")
    print(f"[OK] Metadata saved to: {metadata_path}")
    
    # Print statistics
    print("\n[INFO] Race Distribution:")
    for race, count in sorted(metadata["races"].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(metadata["images"])) * 100
        print(f"  {race:<20}: {count:>6} ({percentage:>5.1f}%)")
    
    print("\n[INFO] Gender Distribution:")
    for gender, count in metadata["genders"].items():
        percentage = (count / len(metadata["images"])) * 100
        print(f"  {gender:<20}: {count:>6} ({percentage:>5.1f}%)")
    
    return True


def create_sample_dataset(output_dir: str = None):
    """Create a small sample dataset for quick testing"""
    
    if output_dir is None:
        output_dir = Path(__file__).parent / "fairface_sample"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n[INFO] Note: To use real FairFace dataset, run:")
    print("  python benchmark/download_fairface.py")
    print("\nFor faster testing, you can add a small sample of test images to:")
    print(f"  {output_dir}/")
    
    return True


def main():
    parser = ArgumentParser(description="Download FairFace Dataset from HuggingFace")
    parser.add_argument("--split", default="train", choices=["train", "test", "val"],
                       help="Dataset split to download (default: train)")
    parser.add_argument("--size", default="full", choices=["small", "medium", "large", "full"],
                       help="Dataset size (default: full)")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--sample-only", action="store_true",
                       help="Just create sample directory without downloading")
    
    args = parser.parse_args()
    
    # Map size to max_images
    size_map = {
        "small": 1000,
        "medium": 5000,
        "large": 20000,
        "full": None
    }
    
    max_images = size_map[args.size]
    
    if args.sample_only:
        create_sample_dataset(args.output)
    else:
        success = download_fairface(
            split=args.split,
            max_images=max_images,
            output_dir=args.output
        )
        
        if success:
            print("\n[OK] Download completed successfully!")
            print("[INFO] You can now run benchmark with FairFace dataset")
        else:
            print("\n[ERROR] Download failed")
            sys.exit(1)


if __name__ == "__main__":
    main()

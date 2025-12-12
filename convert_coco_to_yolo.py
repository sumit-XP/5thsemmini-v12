
import json
import os
from pathlib import Path
from tqdm import tqdm

def convert_coco_to_yolo(json_path, output_dir):
    """Converts COCO JSON annotations to YOLO format text files."""
    
    # Load COCO JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    images = {img['id']: img for img in data['images']}
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    
    # Needs to ensure class IDs are 0-indexed and consistent
    # We will sort categories by ID to ensure consistent mapping
    sorted_cats = sorted(categories.keys())
    cat_id_map = {old_id: new_id for new_id, old_id in enumerate(sorted_cats)}
    
    print(f"Loaded {len(images)} images and {len(categories)} categories")
    print(f"Categories mapping: {json.dumps({categories[old]: new for old, new in cat_id_map.items()}, indent=2)}")
    
    # Process annotations
    for ann in tqdm(data['annotations'], desc="Converting labels"):
        img_id = ann['image_id']
        if img_id not in images:
            continue
            
        img_info = images[img_id]
        img_w = img_info['width']
        img_h = img_info['height']
        
        # COCO bbox: [x_min, y_min, width, height]
        bbox = ann['bbox']
        x_min, y_min, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        
        # YOLO bbox: [x_center, y_center, width, height] normalized
        x_center = (x_min + w / 2) / img_w
        y_center = (y_min + h / 2) / img_h
        w_norm = w / img_w
        h_norm = h / img_h
        
        if ann['category_id'] not in cat_id_map:
            continue
            
        class_id = cat_id_map[ann['category_id']]
        
        # Output file
        file_name = img_info['file_name']
        txt_name = os.path.splitext(file_name)[0] + ".txt"
        txt_path = os.path.join(output_dir, txt_name)
        
        # Append to file
        with open(txt_path, 'a') as f:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

def main():
    root = "dataset-9"
    splits = ["train", "valid", "test"]
    
    for split in splits:
        json_file = os.path.join(root, split, "_annotations.coco.json")
        output_dir = os.path.join(root, split)
        
        if os.path.exists(json_file):
            print(f"\nConverting {split} set...")
            convert_coco_to_yolo(json_file, output_dir)
        else:
            print(f"Skipping {split}: {json_file} not found")

if __name__ == "__main__":
    main()

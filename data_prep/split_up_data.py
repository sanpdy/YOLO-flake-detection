import os
import shutil
from sklearn.model_selection import train_test_split

# --- CONFIG ---
BASE        = "/home/sankalp/yolo_flake_detection/monark_data"
IMG_DIR     = os.path.join(BASE, "images")
LBL_DIR     = os.path.join(BASE, "labels")
SPLITS      = ["train", "val"]
TEST_SIZE   = 0.2      # 20% val / 80% train
RANDOM_SEED = 42

# 1. Build set of basenames that actually have a label file
all_labels = [f for f in os.listdir(LBL_DIR) if f.lower().endswith(".txt")]
label_basenames = {os.path.splitext(f)[0] for f in all_labels}

# 2. Gather only those images whose basename is in label_basenames
all_imgs = [
    f for f in os.listdir(IMG_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png")) 
    and os.path.splitext(f)[0] in label_basenames
]

# 3. Split into train/val
train_imgs, val_imgs = train_test_split(
    all_imgs, test_size=TEST_SIZE, random_state=RANDOM_SEED
)

# 4. Make the split folders if they don't exist
for split in SPLITS:
    os.makedirs(os.path.join(IMG_DIR,   split), exist_ok=True)
    os.makedirs(os.path.join(LBL_DIR,   split), exist_ok=True)

# 5. Move each image + its .txt into the appropriate split
for split, img_list in [("train", train_imgs), ("val", val_imgs)]:
    for img_name in img_list:
        base, _ = os.path.splitext(img_name)
        src_img = os.path.join(IMG_DIR, img_name)
        src_lbl = os.path.join(LBL_DIR, base + ".txt")
        dst_img = os.path.join(IMG_DIR,   split, img_name)
        dst_lbl = os.path.join(LBL_DIR,   split, base + ".txt")

        shutil.move(src_img, dst_img)
        shutil.move(src_lbl, dst_lbl)

print("Done! 🎉")
print(f"  • {len(train_imgs)} images → images/train + labels/train")
print(f"  • {len(val_imgs)} images → images/val   + labels/val")

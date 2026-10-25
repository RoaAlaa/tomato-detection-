# %%
import os
import xml.etree.ElementTree as ET



# %%
ANNOTATIONS_DIR = './Tomato/annotations'
LABELS_DIR = './Tomato/labels'
os.makedirs(LABELS_DIR, exist_ok=True)

classes = ['tomato']

# %%

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x_center = (box[0] + box[1]) / 2.0
    y_center = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return x_center * dw, y_center * dh, w * dw, h * dh
for filename in os.listdir(ANNOTATIONS_DIR):
    if not filename.endswith('.xml'):
        continue

    xml_path = os.path.join(ANNOTATIONS_DIR, filename)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    out_file = open(os.path.join(LABELS_DIR, filename.replace('.xml', '.txt')), 'w')

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text),
             int(xmlbox.find('xmax').text),
             int(xmlbox.find('ymin').text),
             int(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(f"{cls_id} {' '.join([str(round(a, 6)) for a in bb])}\n")

# %%
import shutil
!pip install scikit-learn

from sklearn.model_selection import train_test_split

# %%

images_dir = './Tomato/images'
labels_dir = './Tomato/labels'

train_images_dir = './Tomato/images/train'
val_images_dir = './Tomato/images/val'
train_labels_dir = './Tomato/labels/train'
val_labels_dir = './Tomato/labels/val'

for d in [train_images_dir, val_images_dir, train_labels_dir, val_labels_dir]:
    os.makedirs(d, exist_ok=True)

images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)

for img in train_imgs:
    shutil.move(os.path.join(images_dir, img), os.path.join(train_images_dir, img))
    shutil.move(os.path.join(labels_dir, img.replace('.png', '.txt').replace('.jpg', '.txt')), os.path.join(train_labels_dir, img.replace('.png', '.txt').replace('.jpg', '.txt')))

for img in val_imgs:
    shutil.move(os.path.join(images_dir, img), os.path.join(val_images_dir, img))
    shutil.move(os.path.join(labels_dir, img.replace('.png', '.txt').replace('.jpg', '.txt')), os.path.join(val_labels_dir, img.replace('.png', '.txt').replace('.jpg', '.txt')))


# %%
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

# %%
trained = model.train(
    data= './Tomato/data.yaml',
    epochs=20,
    imgsz=640,)

# %%
model = YOLO('runs/detect/train3/weights/last.pt')
model.train(resume=True)


# %%
import random

def create_sample_dataset():
    images_dir = './Tomato/images/val'
    labels_dir = './Tomato/labels/val'

    sample_images_dir = './Tomato/sample/images'
    sample_labels_dir = './Tomato/sample/labels'

    os.makedirs(sample_images_dir, exist_ok=True)
    os.makedirs(sample_labels_dir, exist_ok=True)

    all_images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
    sample_size = min(10, len(all_images))    
    sample_images = random.sample(all_images, sample_size)
    
    for img in sample_images:
        shutil.copy(os.path.join(images_dir, img), os.path.join(sample_images_dir, img))
        label_file = img.replace('.png', '.txt').replace('.jpg', '.txt')
        if os.path.exists(os.path.join(labels_dir, label_file)):
            shutil.copy(os.path.join(labels_dir, label_file), os.path.join(sample_labels_dir, label_file))
    print(f"Found {len(all_images)} images, sampling {sample_size}")

create_sample_dataset()



# %%
model = YOLO('runs/detect/train3/weights/best.pt')

# %%
sample_pred = model.predict(source = './Tomato/sample/images', save=True, save_txt=True, name = 'sample_predictions')

# %%
import os
import cv2
import matplotlib.pyplot as plt


sample_img_dir = './Tomato/sample/images'
gt_sample_label_dir = './Tomato/sample/labels'
pred_sample_label_dir = './Tomato/sample_predictions/labels'

def draw_boxes(img, label_path, color, label_type):
    h, w = img.shape[:2]
    if not os.path.exists(label_path):
        return img
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls, xc, yc, bw, bh = map(float, parts[:5])
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f'{label_type}: {int(cls)}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img

# %%
image_files = [f for f in os.listdir(sample_img_dir) if f.endswith(('.jpg', '.png'))]

for img_file in image_files:
    img_path = os.path.join(sample_img_dir, img_file)
    gt_path = os.path.join(gt_sample_label_dir, os.path.splitext(img_file)[0] + '.txt')
    pred_path = os.path.join(pred_sample_label_dir, os.path.splitext(img_file)[0] + '.txt')

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Draw GT in green, predictions in red
    img_with_gt = draw_boxes(img.copy(), gt_path, (0, 255, 0), 'GT')
    img_with_all = draw_boxes(img_with_gt, pred_path, (255, 0, 0), 'Pred')

    # Show image
    plt.figure(figsize=(10, 10))
    plt.imshow(img_with_all)
    plt.title(f"GT vs Prediction: {img_file}")
    plt.axis('off')
    plt.show()

# %%
import os
import cv2
import matplotlib.pyplot as plt
%matplotlib inline

sample_img_dir = './Tomato/sample/images'
gt_sample_label_dir = './Tomato/sample/labels'
pred_sample_label_dir = './runs/detect/sample_predictions/labels'

def draw_boxes(img, label_path, color, label_type):
    h, w = img.shape[:2]
    if not os.path.exists(label_path):
        return img
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls, xc, yc, bw, bh = map(float, parts[:5])
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f'{label_type}: {int(cls)}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img

# Show overlapped GT and prediction for all samples
sample_images = [f for f in os.listdir(sample_img_dir) if f.endswith(('.jpg', '.png'))]

for img_name in sample_images:
    base_name = os.path.splitext(img_name)[0]
    img_path = os.path.join(sample_img_dir, img_name)
    gt_path = os.path.join(gt_sample_label_dir, base_name + '.txt')
    pred_path = os.path.join(pred_sample_label_dir, base_name + '.txt')

    # Load image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Draw GT and predictions
    img_with_boxes = draw_boxes(img.copy(), gt_path, (0, 255, 0), 'GT')        # Green for GT
    img_with_boxes = draw_boxes(img_with_boxes, pred_path, (255, 0, 0), 'Pred') # Blue for prediction

    # Display
    plt.figure(figsize=(8, 8))
    plt.imshow(img_with_boxes)
    plt.title(img_name)
    plt.axis('off')
    plt.show()


# %%
full_prediction = model.predict(source = './Tomato/images/val', save=True, save_txt=True, name = 'sample_predictions')

# %%
img_dir = './Tomato/images/val'
gt_dir = './Tomato/labels/val'
pred_dir = './runs/detect/sample_predictions2/labels'

image_files = sorted(os.listdir(img_dir))[:5]

# Loop through images
for img_file in image_files:
    base = os.path.splitext(img_file)[0]
    img_path = os.path.join(img_dir, img_file)
    gt_path = os.path.join(gt_dir, base + '.txt')
    pred_path = os.path.join(pred_dir, base + '.txt')

    # Read and draw
    img_gt = cv2.imread(img_path)
    img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
    img_pred = img_gt.copy()

    img_gt = draw_boxes(img_gt, gt_path, (0, 255, 0), 'GT')
    img_pred = draw_boxes(img_pred, pred_path, (255, 0, 0), 'Pred')

    # Plot side-by-side
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(img_gt)
    axs[0].set_title('Ground Truth')
    axs[0].axis('off')

    axs[1].imshow(img_pred)
    axs[1].set_title('Prediction')
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()

# %%
from ultralytics import YOLO

# %%
model = YOLO('runs/detect/train3/weights/best.pt')

# %%
sample_val = model.val(data='./Tomato/data.yaml', save=True, save_txt=True, name='sample_val_predictions')



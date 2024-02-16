import pandas as pd
from pycocotools.coco import COCO
import inflect
import re

def construct_expanded_dataset(annotation_file, captions_file):
    # Load COCO annotations
    coco = COCO(annotation_file)
    # Load COCO captions
    captions_data = COCO(captions_file)
    engine = inflect.engine()
    expanded_data = []
    img_ids = coco.getImgIds()

    for img_id in img_ids:
        added_labels = set() 
        ann_ids_for_image = coco.getAnnIds(imgIds=[img_id])
        annotations_for_image = coco.loadAnns(ann_ids_for_image)
        cat_ids = [ann['category_id'] for ann in annotations_for_image]

        for cat_id in cat_ids:
            cat_name = coco.loadCats([cat_id])[0]['name']
            plural_cat_name = engine.plural(cat_name)
            
            if cat_name in added_labels:
                continue
            
            for ann in captions_data.loadAnns(captions_data.getAnnIds(imgIds=[img_id])):
                caption = ann['caption']
                if re.search(r'\b' + re.escape(cat_name) + r'\b', caption) or re.search(r'\b' + re.escape(plural_cat_name) + r'\b', caption):
                    correct_obj = plural_cat_name if re.search(r'\b' + re.escape(plural_cat_name) + r'\b', caption) else cat_name
                    context = caption.replace(correct_obj, '')
                    expanded_data.append([img_id, cat_name, caption, context, correct_obj])
                    added_labels.add(cat_name)
                    break

    df = pd.DataFrame(expanded_data, columns=['Image ID', 'Label', 'correct_obj+context', 'context', 'correct_obj'])
    return df

    
def main():
    # Replace with your path
    annotation_file = './datasets/coco/annotations/instances_val2017.json'
    captions_file = './datasets/coco/annotations/captions_val2017.json'
    output_file = './datasets/coco/COCO-IT.csv'

    df = construct_expanded_dataset(annotation_file, captions_file)
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    main()

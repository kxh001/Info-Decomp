import json
import csv
import inflect

engine = inflect.engine()


# Load COCO annotations, replace with your path
with open('/data/xkong016/research/datasets/coco/annotations/instances_val2017.json', 'r') as f:
    data = json.load(f)

# Load COCO captions, replace with your path
with open('/data/xkong016/research/datasets/coco/annotations/captions_val2017.json', 'r') as f:
    captions_data = json.load(f)

# Create a dictionary for captions based on image_id
captions_dict = {}
for item in captions_data['annotations']:
    captions_dict[item['image_id']] = item['caption']

# Extract categories and images
categories = [cat['name'] for cat in data['categories']]
images = data['images']

# Create a dictionary for captions based on image_id
captions_dict = {}
for item in captions_data['annotations']:
    if item['image_id'] not in captions_dict:
        captions_dict[item['image_id']] = []
    captions_dict[item['image_id']].append(item['caption'])

# Process data
rows = []
for category in categories[:200]:  # Selecting the first 200 categories
    count = 0
    for image in images:
        if count >= 200:
            break
        image_id = image['id']
        if image_id in captions_dict:
            plural_category = engine.plural(category)
            correct_caption = None
            correct_obj = None
            
            for caption in captions_dict[image_id]:
                # Check if the entire word (singular or plural form) is present in the caption
                if (" " + category + " " in caption or caption.startswith(category + " ") or caption.endswith(" " + category)) or \
                   (" " + plural_category + " " in caption or caption.startswith(plural_category + " ") or caption.endswith(" " + plural_category)):
                    if plural_category in caption:
                        correct_obj = plural_category
                    else:
                        correct_obj = category
                    correct_caption = caption
                    break
            
            if not correct_caption:
                continue  # Skip this image if no captions match the criteria
            
            row = []
            row.append(image_id)
            row.append(category)
            row.append(correct_caption)
            row.append(correct_caption.replace(correct_obj, ''))
            row.append(correct_obj)
            rows.append(row)
            count += 1

# Write to CSV, replace with your output path
with open('output.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Image ID', 'Label', 'correct_obj+context', 'context', 'correct_obj'])
    csvwriter.writerows(rows)







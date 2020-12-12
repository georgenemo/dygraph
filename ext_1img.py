import numpy as np
import json
from pycocotools.coco import COCO
from IPython import embed

#split = 'train' 
split = 'val'
annFile = './instances_{}2017.json'.format(split)
old_json = json.load(open(annFile,'r'))

newFile = './instances_{}2017_1img.json'.format(split)
new_json = {}
new_json.update({'categories':old_json['categories']})# person 0

coco=COCO(annFile)

# get all images containing given categories, select one at random
catIds = coco.getCatIds(catNms=['person', 'dog'])
imgIds = coco.getImgIds(catIds=catIds)

person_images = []
person_anns_info = []

for i in range(len(imgIds)):
    img = coco.loadImgs(imgIds[i])[0]
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns_info = coco.loadAnns(annIds)

    person_images.append(img)
    person_anns_info.extend(anns_info) ##
    if i==0:
        break#print(i)

new_json.update({'images':person_images, 'annotations':person_anns_info})

with open(newFile, 'w') as f:
    json.dump(new_json, f)

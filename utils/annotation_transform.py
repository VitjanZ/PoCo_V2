import csv
import glob
import csv
import pandas as pd
import os

annotations_rect = []
dir_name = "F:/Downloads/dataset-kristjan/"
gt_files = glob.glob(dir_name+"/*.bbox")
gt_files = sorted(gt_files)

image_files = glob.glob(dir_name+"/*.jpg")
image_files = sorted(image_files)

tmp_arr = []
for gt, img in zip(gt_files,image_files):
    polyps = pd.read_csv(gt, header=None, delimiter=r"\s+")
    tmp_arr = []
    for ind, polyp in polyps.iterrows():
        x, y, w, h = polyp
        b_name = os.path.basename(img)
        tmp_arr.append([x+w//2, y+h//2, min(w,h)//2,min(w,h)//2])
    image_base_name = ".".join(os.path.basename(img).split(".")[:-1])
    if len(tmp_arr) != 0:
        with open(dir_name + '/' + image_base_name + '_gt.csv', 'w+') as f:
            writer = csv.writer(f, delimiter=',', lineterminator='\n')
            writer.writerows(tmp_arr)
            f.close()

# with open(dir_name +'/annotations.csv', 'w+') as f:
#     writer = csv.writer(f, delimiter=',', lineterminator='\n')
#     writer.writerows(tmp_arr)
#     f.close()
#
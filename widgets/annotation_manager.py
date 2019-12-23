import csv
import os
import glob
from cv2 import convexHull, pointPolygonTest
import numpy as np


class annotation_manager():
    def __init__(self, parent):
        self.images = {}
        self.annotations_rect = {}
        self.negative_annotations_rect = {}
        self.roi_points = {}
        self.annotations_changed = {}
        self.predicted_annotations_rect = {}
        self.image_shapes = {}

        self.dir_name = ""
        self.parent = parent
        self.image_names = []


    def change_annotations(self, detections):
        for name in detections.keys():
            det_list = [(x[0], x[1], x[2], x[3]) for x in detections[name]]
            if name in self.roi_points and len(self.roi_points[name]) != 0:
                det_list = list(filter(
                    lambda x: pointPolygonTest(np.array(self.roi_points[name]).astype(np.int32), (x[0], x[1]),
                                               False) != -1, det_list))
            detection_set = set(det_list)

            if name not in self.negative_annotations_rect:
                self.negative_annotations_rect[name] = set()

            self.annotations_changed[name] = True
            if len(self.annotations_rect[name]) == 0:
                self.annotations_rect[name] = detection_set
            else:
                points = np.array([(int(x[0]), int(x[1])) for x in self.annotations_rect[name]], dtype=np.int32)
                if len(points > 0):
                    c_hull = convexHull(points)
                    detection_set = set(
                        list(filter(lambda x: pointPolygonTest(c_hull, (x[0], x[1]), False) == -1, det_list)))
                    self.annotations_rect[name].update(detection_set)
                    # self.annotations_rect[name] = detection_set

            print(len(self.annotations_rect[name]))

    def save_annotations_old(self):
        tmp_arr = []
        for key in self.annotations_rect:
            for annot in self.annotations_rect[key]:
                x, y, w, h = annot
                tmp_arr.append([key, x, y, w, h])
        with open(self.dir_name + '/annotations.csv', 'w+') as f:
            writer = csv.writer(f, delimiter=',', lineterminator='\n')
            writer.writerows(tmp_arr)
            f.close()

        tmp_arr = []
        for key in self.negative_annotations_rect:
            for annot in self.negative_annotations_rect[key]:
                x, y, w, h = annot
                tmp_arr.append([key, x, y, w, h])

        with open(self.dir_name + '/negative_annotations.csv', 'w+') as f:
            writer = csv.writer(f, delimiter=',', lineterminator='\n')
            writer.writerows(tmp_arr)
            f.close()

        tmp_arr = []
        for key in self.roi_points:
            for annot in self.roi_points[key]:
                x, y = annot
                tmp_arr.append([key, x, y])

        with open(self.dir_name + '/roi_points.csv', 'w+') as f:
            writer = csv.writer(f, delimiter=',', lineterminator='\n')
            writer.writerows(tmp_arr)
            f.close()

    def save_annotations(self):
        for k in self.annotations_rect:
            if k != None:
                tmp_arr = []
                for annot in self.annotations_rect[k]:
                    tmp_arr.append(annot)
                image_base_name = ".".join(k.split(".")[:-1])
                if len(tmp_arr) != 0:
                    with open(self.dir_name + '/' + image_base_name + '_gt.csv', 'w+') as f:
                        writer = csv.writer(f, delimiter=',', lineterminator='\n')
                        writer.writerows(tmp_arr)
                        f.close()

                if len(self.roi_points[k]) != 0:
                    with open(self.dir_name + '/' + image_base_name + '_gt_poly.csv', 'w+') as f:
                        writer = csv.writer(f, delimiter=',', lineterminator='\n')
                        writer.writerows(self.roi_points[k])
                        f.close()

    def load_annotations(self, dir_name):
        if dir_name == "":
            return []

        self.dir_name = dir_name
        self.annotations_rect = {}

        image_names = glob.glob(dir_name + "/*.jpg")
        image_names.extend(glob.glob(dir_name + "/*.png"))
        image_names.extend(glob.glob(dir_name + "/*.bmp"))
        image_names.extend(glob.glob(dir_name + "/*.tif"))
        self.image_names = [os.path.basename(x) for x in image_names]

        for name in self.image_names:
            image_base_name = ".".join(name.split(".")[:-1])
            if os.path.exists(dir_name + '/' + image_base_name + '_gt.csv'):
                with open(dir_name + '/' + image_base_name + '_gt.csv', 'r') as f:
                    is_first = True
                    reader = csv.reader(f, delimiter=',')
                    for annot in reader:
                        x, y, w, h = annot
                        if name not in self.annotations_rect:
                            self.annotations_rect[name] = set()
                        self.annotations_rect[name].add((float(x), float(y), float(w), float(h)))
                    f.close()

            # POLY FILES
            if os.path.exists(dir_name + '/' + image_base_name + '_gt_poly.csv'):
                with open(dir_name + '/' + image_base_name + '_gt_poly.csv', 'r') as f:
                    reader = csv.reader(f, delimiter=',')
                    for annot in reader:
                        x, y = annot
                        if name not in self.roi_points:
                            self.roi_points[name] = []
                        self.roi_points[name].append((float(x), float(y)))
                    f.close()

        for name in self.image_names:
            if name not in self.roi_points:
                self.roi_points[name] = []
            if name not in self.annotations_rect:
                self.annotations_rect[name] = set()
            if name not in self.negative_annotations_rect:
                self.negative_annotations_rect[name] = set()

        return image_names

    def load_annotations_old(self):
        dir_name = self.showDirectoryDialog()
        if dir_name == "":
            return []
        self.dir_name = dir_name
        self.annotations_rect = {}
        self.negative_annotations_rect = {}
        if os.path.exists(dir_name + '/annotations.csv'):
            with open(dir_name + '/annotations.csv', 'r') as f:
                reader = csv.reader(f, delimiter=',')
                for annot in reader:
                    name, x, y, w, h = annot
                    if name not in self.annotations_rect:
                        self.annotations_rect[name] = set()
                    self.annotations_rect[name].add((float(x), float(y), float(w), float(h)))
                f.close()

        if os.path.exists(dir_name + '/negative_annotations.csv'):
            with open(dir_name + '/negative_annotations.csv', 'r') as f:
                reader = csv.reader(f, delimiter=',')
                for annot in reader:
                    name, x, y, w, h = annot
                    if name not in self.negative_annotations_rect:
                        self.negative_annotations_rect[name] = set()
                    self.negative_annotations_rect[name].add((float(x), float(y), float(w), float(h)))
                f.close()

        if os.path.exists(dir_name + '/roi_points.csv'):
            with open(dir_name + '/roi_points.csv', 'r') as f:
                reader = csv.reader(f, delimiter=',')
                for annot in reader:
                    name, x, y = annot
                    if name not in self.roi_points:
                        self.roi_points[name] = []
                    self.roi_points[name].append((float(x), float(y)))
                f.close()

        image_names = glob.glob(dir_name + "/*.jpg")
        image_names.extend(glob.glob(dir_name + "/*.png"))
        image_names.extend(glob.glob(dir_name + "/*.bmp"))
        self.image_names = [os.path.basename(x) for x in image_names]
        for name in self.image_names:
            if name not in self.roi_points:
                self.roi_points[name] = []
            if name not in self.annotations_rect:
                self.annotations_rect[name] = set()
            if name not in self.negative_annotations_rect:
                self.negative_annotations_rect[name] = set()
        print(self.roi_points)

        return image_names

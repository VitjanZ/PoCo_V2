import cv2
import glob
import os
import numpy as np
import pickle
from PyQt5.QtCore import QRunnable, QObject, pyqtSignal

class data_signals(QObject):
    progress_signal = pyqtSignal(object)
    result_signal = pyqtSignal(int)


class prepare_worker(QRunnable):
    def __init__(self, annotation_manager, data_dialog=None):
        super(prepare_worker, self).__init__()
        self.annotation_manager = annotation_manager
        self.data_handler = data_handler(annotation_manager)
        self.setAutoDelete(True)
        self.signals = self.data_handler.signals
        self.data_dialog = data_dialog
        self.data_dialog.stop_prepare_signal.connect(self.data_handler.handle_stop_prepare_signal)

    def run(self):
        self.data_handler.prepare_data_whole()

class data_handler():
    def __init__(self, annotation_manager):
        self.annotation_manager = annotation_manager
        self.dir_name = annotation_manager.dir_name
        self.image_names = annotation_manager.image_names
        self.signals = data_signals()
        self.stop_process = False

    def handle_stop_prepare_signal(self,val):
        self.stop_process = True

    def mask_out_polygon(self, label_mask, points, is_poly=False):
        if not is_poly:
            polygon = cv2.convexHull(np.array(points).astype(np.int32))
        else:
            polygon = points
        mask = np.zeros(label_mask.shape)
        mask = cv2.fillPoly(mask, [np.array(polygon)], 1)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = np.logical_not(mask)

        label_mask[mask] = 0.5
        return label_mask

    def prepare_mask_whole(self, image, image_annotations, negative_annotations, roi_poly):
        mask = np.zeros(image.shape[:2])
        white_mask = np.zeros(image.shape[:2])
        points = []
        for annot in image_annotations:
            x, y, w, h = annot
            p1 = [min(max(x,0),image.shape[1]-1),min(max(y,0),image.shape[0]-1)]
            p2 = [min(max(x+w,0),image.shape[1]-1),min(max(y+h,0),image.shape[0]-1)]
            p3 = [min(max(x+w,0),image.shape[1]-1),min(max(y,0),image.shape[0]-1)]
            p4 = [min(max(x,0),image.shape[1]-1),min(max(y+h,0),image.shape[0]-1)]
            points.append(p1)
            points.append(p2)
            points.append(p3)
            points.append(p4)

            r = int(max(w, h))//2
            #cv2.circle(mask, (int(x + w / 2), int(y + h / 2)), 2*r, 0.5, -1)
            cv2.circle(mask, (int(x), int(y)), 2*r, 0.5, -1)
            #cv2.circle(mask, (int(x + w / 2), int(y + h / 2)), r, 1, -1)
            cv2.circle(mask, (int(x), int(y)), r, 1, -1)
            #cv2.circle(white_mask, (int(x + w / 2), int(y + h / 2)), r, 1, -1)
            cv2.circle(white_mask, (int(x), int(y)), r, 1, -1)

        for annot in negative_annotations:
            x, y, w, h = annot
            p1 = [min(max(x,0),image.shape[1]-1),min(max(y,0),image.shape[0]-1)]
            p2 = [min(max(x+w,0),image.shape[1]-1),min(max(y+h,0),image.shape[0]-1)]
            p3 = [min(max(x+w,0),image.shape[1]-1),min(max(y,0),image.shape[0]-1)]
            p4 = [min(max(x,0),image.shape[1]-1),min(max(y+h,0),image.shape[0]-1)]
            points.append(p1)
            points.append(p2)
            points.append(p3)
            points.append(p4)
            cv2.rectangle(white_mask, (int(x), int(y)), (int(x+w),int(y+h)), 0.5, -1)

        print(roi_poly)
        if roi_poly != None and len(roi_poly) > 0:
            points = roi_poly

        min_x = int(min(max(np.min(np.array(points)[:,0])-512,0), image.shape[1]-1))
        min_y = int(min(max(np.min(np.array(points)[:,1])-512,0), image.shape[0]-1))
        max_x = int(min(max(np.max(np.array(points)[:,0])+512,0), image.shape[1]-1))
        max_y = int(min(max(np.max(np.array(points)[:,1])+512,0), image.shape[0]-1))

        mask = self.mask_out_polygon(mask, points)
        mask[white_mask == 1] = 1
        mask[white_mask == 0.5] = 0
        return mask, min_x,min_y,max_x,max_y


    def prepare_mask(self, image, image_annotations, negative_annotations, parts_x,parts_y):
        mask = np.zeros(image.shape[:2])
        white_mask = np.zeros(image.shape[:2])
        points = []
        part_size_x = mask.shape[1]//parts_x
        part_size_y = mask.shape[0]//parts_y
        sector_annotations = np.zeros((parts_y,parts_x))
        for annot in image_annotations:
            x, y, w, h = annot
            p1 = [min(max(x,0),image.shape[1]-1),min(max(y,0),image.shape[0]-1)]
            p2 = [min(max(x+w,0),image.shape[1]-1),min(max(y+h,0),image.shape[0]-1)]
            p3 = [min(max(x+w,0),image.shape[1]-1),min(max(y,0),image.shape[0]-1)]
            p4 = [min(max(x,0),image.shape[1]-1),min(max(y+h,0),image.shape[0]-1)]
            points.append(p1)
            points.append(p2)
            points.append(p3)
            points.append(p4)

            r = int(max(w, h))//2
            sector_annotations[int(y/part_size_y), int(x/part_size_x)] += 1
            cv2.circle(mask, (int(x + w / 2), int(y + h / 2)), int(r*1.4), 0.5, -1)
            cv2.circle(mask, (int(x + w / 2), int(y + h / 2)), r // 2, 1, -1)
            cv2.circle(white_mask, (int(x + w / 2), int(y + h / 2)), r // 2, 1, -1)

        for annot in negative_annotations:
            x, y, w, h = annot
            cv2.rectangle(white_mask, (int(x), int(y)), (int(x+w),int(y+h)), 0.5, -1)
            p1 = [min(max(x,0),image.shape[1]-1),min(max(y,0),image.shape[0]-1)]
            p2 = [min(max(x+w,0),image.shape[1]-1),min(max(y+h,0),image.shape[0]-1)]
            p3 = [min(max(x+w,0),image.shape[1]-1),min(max(y,0),image.shape[0]-1)]
            p4 = [min(max(x,0),image.shape[1]-1),min(max(y+h,0),image.shape[0]-1)]

            sector_annotations[min(int(p1[1] / part_size_y), parts_y-1), min(int(p1[0] / part_size_x), parts_x-1)] += 1
            sector_annotations[min(int(p2[1] / part_size_y), parts_y-1), min(int(p2[0] / part_size_x), parts_x-1)] += 1
            sector_annotations[min(int(p3[1] / part_size_y), parts_y-1), min(int(p3[0] / part_size_x), parts_x-1)] += 1
            sector_annotations[min(int(p4[1] / part_size_y), parts_y-1), min(int(p4[0] / part_size_x), parts_x-1)] += 1

        mask = self.mask_out_polygon(mask, points)
        mask[white_mask == 1] = 1
        mask[white_mask == 0.5] = 0
        return mask, sector_annotations

    def split_image_parts(self, parts_x, parts_y, image):
        x_parts = [(image.shape[1]//parts_x*x,image.shape[1]//parts_x*(x+1)) for x in range(parts_x)]
        y_parts = [(image.shape[0]//parts_y*x,image.shape[0]//parts_y*(x+1)) for x in range(parts_y)]

        total_image = None
        if len(image.shape) > 2:
            total_image = np.zeros((parts_x*parts_y, image.shape[0]//parts_y,image.shape[1]//parts_x, image.shape[2]))
        else:
            total_image = np.zeros((parts_x*parts_y, image.shape[0]//parts_y,image.shape[1]//parts_x))
        #print(total_image.shape)
        #print(image.shape)
        cnt = 0
        for y1,y2 in y_parts:
            for x1,x2 in x_parts:
                #print("%d %d %d %d" % (y1,y2,x1,x2))
                t_img = image[y1:y2,x1:x2]
                #print(t_img.shape)
                total_image[cnt] = t_img
                cnt += 1

        return total_image

    def split_image_quarters(self, image, mask):
        half_y = image.shape[0]//2
        half_x = image.shape[1]//2
        t_image = np.zeros((4, half_y, half_x, image.shape[2]))
        t_mask = np.zeros((4, half_y, half_x))

        t_image[0] = image[:half_y, :half_x,:]
        t_image[1] = image[:half_y, half_x:,:]
        t_image[2] = image[half_y:, :half_x,:]
        t_image[3] = image[half_y:, half_x:,:]

        t_mask[0] = mask[:half_y, :half_x]
        t_mask[1] = mask[:half_y, half_x:]
        t_mask[2] = mask[half_y:, :half_x]
        t_mask[3] = mask[half_y:, half_x:]

        return t_image, t_mask

    def prepare_data_whole(self):
        print("Preparing data")
        #images = glob.glob(self.dir_name + "/*.jpg")
        #annotations = glob.glob(self.dir_name + "/*.csv")
        #if not os._exists(self.dir_name+"/annotations.csv"):
        #    print("No file named annotations.csv in " + self.dir_name)
        print(self.dir_name)
        reference_image = cv2.imread(self.dir_name + "/" + self.image_names[0], cv2.IMREAD_COLOR)
        reference_shape = reference_image.shape

        total_save = False
        annotations = self.annotation_manager.annotations_rect
        negative_annotations = self.annotation_manager.negative_annotations_rect
        polygon_interest = self.annotation_manager.roi_points
        if not os.path.exists(self.dir_name + "/training_data/"):
            os.makedirs(self.dir_name + "/training_data/")
            total_save = True

        if not os.path.exists(self.dir_name + "/training_data/masks/masks/"):
            os.makedirs(self.dir_name + "/training_data/masks/masks/")
            total_save = True

        if not os.path.exists(self.dir_name + "/training_data/images/images/"):
            os.makedirs(self.dir_name + "/training_data/images/images/")
            total_save = True

        if not os.path.exists(self.dir_name + "/training_data/cache/"):
            os.makedirs(self.dir_name + "/training_data/cache/")
            total_save = True

        # delete existing files
        files = list(filter(lambda f: "".join(os.path.basename(f).split("_")[1:]) not in self.image_names, glob.glob(self.dir_name + "/training_data/images/images/*")))
        for f in files:
            os.remove(f)

        files = list(filter(lambda f: "".join(os.path.basename(f).split("_")[1:]) not in self.image_names, glob.glob(self.dir_name + "/training_data/masks/masks/*")))
        for f in files:
            os.remove(f)

        #files which are currently saved
        saved_files = set([ "".join(os.path.basename(f).split("_")[1:]) for f in files])

        mean_sum = np.zeros(3, np.float64)
        std_sqe_sum = np.zeros(3, np.float64)
        std_pix_num = 0
        cnt=0
        for image_name in self.image_names:
            if self.stop_process:
                self.signals.result_signal.emit(1)
                return

            print(image_name)
            if image_name in annotations:
                image_annotations = annotations[image_name]
                roi_poly = polygon_interest[image_name]
                print(len(image_annotations))
            else:
                #Skip image if no annotations present
                continue
            if len(image_annotations) <= 0:
                continue


            image = cv2.imread(self.dir_name + "/" + image_name, cv2.IMREAD_COLOR)
            mean_sum[0] += np.sum(image[:,:,0])
            mean_sum[1] += np.sum(image[:,:,1])
            mean_sum[2] += np.sum(image[:,:,2])
            std_pix_num += image.shape[0] * image.shape[1]

            if (not total_save) and ((image_name not in self.annotation_manager.annotations_changed or not self.annotation_manager.annotations_changed[image_name])) and image_name in saved_files:
                print(image_name+" unchanged. Not saving data.")
                continue



            if image_name in negative_annotations:
                image_negative_annotations = negative_annotations[image_name]
            else:
                continue


            mask,min_x,min_y,max_x,max_y= self.prepare_mask_whole(image, image_annotations,image_negative_annotations, roi_poly)
            #image = cv2.resize(image, (reference_shape[1], reference_shape[0]), interpolation=cv2.INTER_AREA)
            #mask = cv2.resize(mask, (reference_shape[1], reference_shape[0]), interpolation=cv2.INTER_AREA)
            image_crop = image[min_y:max_y,min_x:max_x]
            mask_crop = mask[min_y:max_y,min_x:max_x]


            cv2.imwrite(self.dir_name+"/training_data/images/images/"+image_name,image_crop)
            #multiply by 255 to visualize
            cv2.imwrite(self.dir_name+"/training_data/masks/masks/"+image_name,np.uint8(mask_crop*255))
            self.signals.progress_signal.emit((cnt,len(self.image_names)))
            cnt+=1

        mean = mean_sum/std_pix_num

        for image_name in self.image_names:
            if self.stop_process:
                self.signals.result_signal.emit(1)
                return

            print(image_name)
            if image_name in annotations:
                image_annotations = annotations[image_name]
                print(len(image_annotations))
            else:
                #Skip image if no annotations present
                continue
            if len(image_annotations) <= 0:
                continue

            image = cv2.imread(self.dir_name + "/" + image_name, cv2.IMREAD_COLOR)
            std_sqe_sum[0] += np.sum((image[:,:,0]-mean[0])**2)
            std_sqe_sum[1] += np.sum((image[:,:,0]-mean[1])**2)
            std_sqe_sum[2] += np.sum((image[:,:,0]-mean[2])**2)

        std = (std_sqe_sum/(std_pix_num-1))**0.5
        attrs = {}
        f_name = self.dir_name+"/training_data/cache/mean_std.pckl"
        if os.path.exists(f_name):
            f = open(f_name, 'rb')
            attrs = pickle.load(f)
            f.close()

        f = open(f_name,'wb+')
        attrs['mean'] = mean
        attrs['std'] = std
        pickle.dump(attrs,f)
        f.close()
        self.signals.result_signal.emit(1)

    def prepare_data(self):
        print("Preparing data")
        #images = glob.glob(self.dir_name + "/*.jpg")
        #annotations = glob.glob(self.dir_name + "/*.csv")
        #if not os._exists(self.dir_name+"/annotations.csv"):
        #    print("No file named annotations.csv in " + self.dir_name)
        print(self.dir_name)
        reference_image = cv2.imread(self.dir_name + "/" + self.image_names[0], cv2.IMREAD_COLOR)
        reference_shape = reference_image.shape
        parts_y = reference_shape[0]//512
        parts_x = reference_shape[1]//512

        block_numbers = 4
        divisor_power = 1
        while parts_y % (2 ** divisor_power) == 0 and parts_x % (2 ** divisor_power) == 0:
            divisor_power += 1
        divisor_required = (2 ** (block_numbers - divisor_power + 2))

        reference_shape = [reference_shape[0] // (parts_y * divisor_required) * (parts_y * divisor_required),
                           reference_shape[1] // (parts_x * divisor_required) * (parts_x * divisor_required),
                           reference_shape[2]]

        total_save = False
        annotations = self.annotation_manager.annotations_rect
        negative_annotations = self.annotation_manager.negative_annotations_rect
        if not os.path.exists(self.dir_name + "/training_data/"):
            os.makedirs(self.dir_name + "/training_data/")
            total_save = True

        if not os.path.exists(self.dir_name + "/training_data/masks/masks/"):
            os.makedirs(self.dir_name + "/training_data/masks/masks/")
            total_save = True

        if not os.path.exists(self.dir_name + "/training_data/images/images/"):
            os.makedirs(self.dir_name + "/training_data/images/images/")
            total_save = True

        if not os.path.exists(self.dir_name + "/training_data/cache/"):
            os.makedirs(self.dir_name + "/training_data/cache/")
            total_save = True

        # delete existing files
        files = list(filter(lambda f: "".join(os.path.basename(f).split("_")[1:]) not in self.image_names, glob.glob(self.dir_name + "/training_data/images/images/*")))
        for f in files:
            os.remove(f)

        files = list(filter(lambda f: "".join(os.path.basename(f).split("_")[1:]) not in self.image_names, glob.glob(self.dir_name + "/training_data/masks/masks/*")))
        for f in files:
            os.remove(f)

        #files which are currently saved
        saved_files = set([ "".join(os.path.basename(f).split("_")[1:]) for f in files])

        mean_sum = np.zeros(3, np.float64)
        std_sqe_sum = np.zeros(3, np.float64)
        std_pix_num = 0

        for image_name in self.image_names:
            print(image_name)
            image = cv2.imread(self.dir_name + "/" + image_name, cv2.IMREAD_COLOR)
            mean_sum[0] += np.sum(image[:,:,0])
            mean_sum[1] += np.sum(image[:,:,1])
            mean_sum[2] += np.sum(image[:,:,2])
            std_pix_num += image.shape[0] * image.shape[1]

            if (not total_save) and ((image_name not in self.annotation_manager.annotations_changed or not self.annotation_manager.annotations_changed[image_name])) and image_name in saved_files:
                print(image_name+" unchanged. Not saving data.")
                continue

            print(mean_sum)

            image_annotations = set()
            image_negative_annotations = set()

            if image_name in negative_annotations:
                image_negative_annotations = negative_annotations[image_name]
            else:
                continue

            if image_name in annotations:
                image_annotations = annotations[image_name]
            else:
                #Skip image if no annotations present
                continue
            if len(image_annotations) <= 0:
                continue

            mask, sector_annotations = self.prepare_mask(image, image_annotations,image_negative_annotations, parts_x, parts_y)
            sector_annot_flat = np.zeros(parts_x*parts_y)
            cnt = 0
            for py in range(parts_y):
                for px in range(parts_x):
                    sector_annot_flat[cnt] = (sector_annotations[py,px])
                    cnt += 1


            image = cv2.resize(image, (reference_shape[1], reference_shape[0]), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (reference_shape[1], reference_shape[0]), interpolation=cv2.INTER_AREA)
            n_image= self.split_image_parts(parts_x,parts_y,image)
            n_mask= self.split_image_parts(parts_x,parts_y,mask)

            for i in range(parts_x*parts_y):
                if sector_annot_flat[i] > 0:
                    cv2.imwrite(self.dir_name+"/training_data/images/images/"+str(i)+"_"+image_name,n_image[i])
                    #multiply by 255 to visualize
                    cv2.imwrite(self.dir_name+"/training_data/masks/masks/"+str(i)+"_"+image_name,n_mask[i]*255)

        print(mean_sum)
        mean = mean_sum/std_pix_num

        for image_name in self.image_names:
            print(image_name)
            image = cv2.imread(self.dir_name + "/" + image_name, cv2.IMREAD_COLOR)
            std_sqe_sum[0] += np.sum((image[:,:,0]-mean[0])**2)
            std_sqe_sum[1] += np.sum((image[:,:,0]-mean[1])**2)
            std_sqe_sum[2] += np.sum((image[:,:,0]-mean[2])**2)

        print(std_sqe_sum)
        print(std_pix_num)
        std = (std_sqe_sum/(std_pix_num-1))**0.5
        print(std)
        print(mean)
        attrs = {}
        f_name = self.dir_name+"/training_data/cache/mean_std.pckl"
        if os.path.exists(f_name):
            f = open(f_name, 'rb')
            attrs = pickle.load(f)
            f.close()

        f = open(f_name,'wb+')
        attrs['mean'] = mean
        attrs['std'] = std
        pickle.dump(attrs,f)
        f.close()

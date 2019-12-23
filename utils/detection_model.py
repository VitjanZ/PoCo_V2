from keras.layers import Conv2D, MaxPooling2D, Dropout, Input, UpSampling2D, concatenate
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import cv2
from .data_otf_generator_tf import dataset_generator
from .non_maxima_suppression import non_max_suppression_reverse
from PyQt5.QtCore import QRunnable, QObject, pyqtSignal

class detection_signals(QObject):
    epoch_signal = pyqtSignal(int)
    loss_signal = pyqtSignal(float)
    progress_signal = pyqtSignal(int)

    detection_result = pyqtSignal(object)
    predict_image_signal = pyqtSignal(object)

    training_done_signal = pyqtSignal(bool)
    predicting_done_signal = pyqtSignal(bool)

class training_worker(QRunnable):

    def __init__(self, dir_name, image_names, train_dialog=None):
        super(training_worker, self).__init__()
        self.dir_name = dir_name
        self.image_names = image_names
        self.det_model = detection_model(dir_name, image_names, train_dialog=train_dialog)
        self.setAutoDelete(True)

    def run(self):
        self.det_model.train()


class predicting_worker(QRunnable):
    def __init__(self, dir_name, image_names, predict_dialog=None, size_range=None):
        super(predicting_worker, self).__init__()
        self.dir_name = dir_name
        self.image_names = image_names
        self.det_model = detection_model(dir_name, image_names, predict_dialog=predict_dialog, load_model=True, size_range=size_range)
        self.setAutoDelete(True)
        self.signals = self.det_model.det_signals
        if predict_dialog is not None:
            self.model_name = predict_dialog.model_name
        else:
            self.model_name = None


    def run(self):
        self.det_model.predict(self.model_name)


class detection_model():
    def __init__(self, dir_name, image_names, load_model = False, train_dialog=None, predict_dialog=None, size_range=None):
        self.dir_name = dir_name
        self.image_names = image_names

        block_number=4
        self.block_number = block_number

        self.attributes = {'std':None, 'mean':None}

        self.dataset_generator = dataset_generator(dir_name,image_names,self.attributes)
        self.generator = None
        self.model = None
        self.load_model = load_model
        self.stop_training = False
        self.stop_predicting = False
        self.done_training = True
        self.train_dialog = train_dialog
        self.predict_dialog = predict_dialog
        self.det_signals = detection_signals()
        self.size_range = size_range


        if self.train_dialog != None:
            self.train_dialog.stop_training_signal.connect(self.handle_stop_training_signal)
            self.train_dialog.start_training_signal.connect(self.handle_start_training_signal)
            self.train_dialog.connect_detection_object(self)

        if self.predict_dialog != None:
            self.predict_dialog.stop_predicting_signal.connect(self.handle_stop_predicting_signal)
            self.predict_dialog.connect_detection_object(self)

    def handle_stop_training_signal(self, val):
        self.stop_training = val

    def handle_start_training_signal(self, val):
        self.train()

    def handle_stop_predicting_signal(self, val):
        self.stop_predicting = val

    def binary_crossentropy_label_loss_removal(self, y_true, y_pred):
        labels = y_true
        #get labels around 0.5
        cce = tf.keras.backend.binary_crossentropy(labels, y_pred)

        mask_out_loss_greater = tf.where(tf.greater(labels, 0.45), tf.ones_like(labels), tf.zeros_like(labels))
        mask_out_loss_less = tf.where(tf.less_equal(labels, 0.55), tf.ones_like(labels), tf.zeros_like(labels))
        mask_out_loss = tf.multiply(mask_out_loss_greater, mask_out_loss_less)
        mask_out_loss = tf.cast(tf.logical_not(tf.cast(mask_out_loss, tf.bool)), tf.float32)

        loss = cce * mask_out_loss

        r_loss = tf.keras.backend.mean(loss, axis=-1)

        return r_loss

    def unet_model_blocks(self,inputs=None, block_number=4, filter_number=16):
        if inputs is None:
            inputs = Input((None, None, 3))
        filter_num = filter_number
        x = inputs
        block_features = []
        for i in range(block_number):
            fn_cur = filter_num*(2**(i))
            conv1 = Conv2D(fn_cur, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
            conv1 = Conv2D(fn_cur, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
            block_features.append(conv1)
            pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
            x = pool1

        fn_cur = filter_num*(2**(block_number))
        conv3 = Conv2D(fn_cur, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
        conv3 = Conv2D(fn_cur, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        drop3 = Dropout(0.5)(conv3)
        x = drop3
        for i in range(block_number):
            fn_cur = filter_num*(2**(block_number - i - 1))
            up8 = Conv2D(fn_cur, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                UpSampling2D(size=(2, 2))(x))
            merge8 = concatenate([block_features.pop(), up8], axis=3)

            conv8 = Conv2D(fn_cur, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
            conv8 = Conv2D(fn_cur, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)


            x = conv8

        conv10 = Conv2D(1, 1, activation='sigmoid')(x)
        model = Model(inputs, conv10)

        return inputs,conv10,model

    def train(self):
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.4
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            self.done_training = False
            model_name = self.dir_name+"/model.hdf5"
            if self.train_dialog is not None and self.train_dialog.model_name != "":
                model_name = self.train_dialog.model_name
            self.generator, self.crop_generator = self.dataset_generator.prepare_train_generator()

            inputs,outputs, model = self.unet_model_blocks(block_number=4)

            optimizer = Adam(lr=5e-5)
            loss_fn = self.binary_crossentropy_label_loss_removal
            y_true = Input((None, None, 1))
            y_pred = model(inputs)
            loss = loss_fn(y_true, y_pred)
            updates_op = optimizer.get_updates(
                params=model.trainable_weights,
                loss=loss)

            train = K.function(
                inputs=[inputs, y_true],
                outputs=[loss],
                updates=updates_op)
            num_epochs = 100

            for epoch in range(num_epochs):
                iterations = len(self.image_names)*50
                for it in range(iterations):
                    image,mask = sess.run(self.generator)

                    loss_train = train([image, mask])
                    if self.stop_training:
                        self.stopped_epoch = epoch
                        break

                    self.det_signals.progress_signal.emit(int(((epoch*iterations+it) / (num_epochs*iterations)) * 100))
                    self.det_signals.epoch_signal.emit(epoch)
                    self.det_signals.loss_signal.emit(float(np.mean(loss_train)))

                self.model = model
                model.save(model_name)
                if self.stop_training:
                    break


        self.done_training = True
        print("Stopped training..")
        self.det_signals.progress_signal.emit(100)

    def split_and_predict(self,p_x,p_y,image):
        parts_x = p_x*2 - 1
        parts_y = p_y*2 - 1
        x_parts = [((image.shape[1]//p_x)*x//2,(image.shape[1]//(p_x))*(x+2)//2) for x in range(parts_x)]
        y_parts = [((image.shape[0]//p_y)*x//2,(image.shape[0]//(p_y))*(x+2)//2) for x in range(parts_y)]
        total_image = np.zeros((image.shape[0],image.shape[1]))
        for y1,y2 in y_parts:
            for x1,x2 in x_parts:
                t_img = np.float32(image[y1:y2,x1:x2,:3])
                in_image = np.expand_dims(t_img,axis=0)
                t_img = self.model.predict(in_image, batch_size=1)
                total_image[y1:y2,x1:x2] = np.maximum(t_img[0,...,0], total_image[y1:y2,x1:x2])
        return total_image

    def generate_detections_from_mask(self, mask, threshold=127):
        uint_mask = np.array(mask * 255).astype(np.uint8)
        ret, thresh = cv2.threshold(uint_mask, threshold, 255, 0)
        #thresh = cv2.dilate(thresh, np.ones((5,5)), iterations=2)
        dist_transform = cv2.distanceTransform(thresh.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        skeleton = non_max_suppression_reverse(dist_transform, 11)
        row_ind, col_ind = np.where(skeleton > 0)
        boxes = np.array(
            [[x[1]-dist_transform[x[0],x[1]]*1, x[0]-dist_transform[x[0],x[1]]*1, 2 * int(dist_transform[x[0], x[1]]), 2 * int(dist_transform[x[0], x[1]])] for x in
             zip(row_ind, col_ind)])

        return boxes

    def non_max_sup_boxes(self, boxes, overlapThresh=0.5):
        # if there are no boxes, return an empty list
        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:

            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))

        return boxes[pick].astype("int")

    def predict(self, model_name):
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.4
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            if self.load_model:
                _, _, self.model = self.unet_model_blocks(block_number=self.block_number)
                print(model_name)
                if model_name is None or model_name == "":
                    print(model_name)
                    self.model.load_weights(self.dir_name+"/model.hdf5")
                else:
                    print(model_name)
                    self.model.load_weights(model_name)


            elif self.model == None:
                return {}
            detections = {}
            count = 1
            block_numbers = self.block_number
            for name in self.image_names:
                self.det_signals.predict_image_signal.emit((count,len(self.image_names), name))
                count += 1
                if self.stop_predicting:
                    break
                orig_image = cv2.imread(self.dir_name+"/"+name, cv2.IMREAD_COLOR)
                orig_image = orig_image[..., ::-1].astype(np.float32)/255.0

                orig_shape = orig_image.shape
                reference_shape = orig_image.shape
                parts_y = reference_shape[0] // 512
                parts_x = reference_shape[1] // 512
                divisor_power = 1
                while parts_y % (2 ** divisor_power) == 0 and parts_x % (2 ** divisor_power) == 0:
                    divisor_power += 1
                divisor_required = (2 ** (block_numbers - divisor_power + 2))

                reference_shape = [reference_shape[0] // (parts_y * divisor_required) * (parts_y * divisor_required),
                                   reference_shape[1] // (parts_x * divisor_required) * (parts_x * divisor_required),
                                   reference_shape[2]]
                orig_image = cv2.resize(orig_image, (reference_shape[1], reference_shape[0]), interpolation=cv2.INTER_AREA)

                mask_result = self.split_and_predict(parts_x, parts_y, orig_image)

                mask_result = cv2.resize(mask_result, (mask_result.shape[1]//2, mask_result.shape[0]//2), interpolation=cv2.INTER_AREA)
                mask_result = mask_result / np.max(mask_result)
                t_boxes  = self.generate_detections_from_mask(mask_result, threshold=50)
                if(len(t_boxes) != 0):
                    t_boxes[:,0] = t_boxes[:,0] / mask_result.shape[0] * orig_shape[0]
                    t_boxes[:,1] = t_boxes[:,1] / mask_result.shape[1] * orig_shape[1]
                    t_boxes[:,2] = t_boxes[:,2] / mask_result.shape[0] * orig_shape[0]
                    t_boxes[:,3] = t_boxes[:,3] / mask_result.shape[1] * orig_shape[1]

                    t_boxes = np.array(list(filter(lambda x: (x[2] * x[3] > 0) and (x[0] > 0) and (x[1] > 0), t_boxes)))
                    if self.size_range != None:
                        t_boxes = np.array(list(filter(lambda x: x[2] < 1*self.size_range[1] and x[3] < 1*self.size_range[1] and
                                                        x[2] >= 0.5*self.size_range[0] and x[3] >= 0.5*self.size_range[0] and
                                                        x[0] > 0 and x[1] > 0, t_boxes)))


                    t_boxes[:,0] = t_boxes[:,0] + t_boxes[:,2]//2
                    t_boxes[:,1] = t_boxes[:,1] + t_boxes[:,3]//2

                    # if model is not trained well enough the segmentation masks might not be that
                    # clean and multiple detections can appear at the same object.
                    t_boxes = self.non_max_sup_boxes(t_boxes, overlapThresh=0.7)

                    detections[name] = t_boxes
        K.clear_session()
        self.det_signals.detection_result.emit(detections)


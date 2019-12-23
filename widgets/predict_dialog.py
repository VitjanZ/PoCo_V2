from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
from functools import reduce
from utils.detection_model import predicting_worker

class predict_dialog(QtCore.QObject):

    stop_predicting_signal = QtCore.pyqtSignal(bool)
    predicting_result_signal = QtCore.pyqtSignal(object)

    def __init__(self, parent, Dialog):
        super(predict_dialog, self).__init__()
        self.parent = parent
        self.detection_object = None
        self.model_name = ""
        self.Dialog = Dialog
        self.dir_name = self.parent.dir_name

    def setupUi(self, predict_dialog):
        self.main_dialog = predict_dialog
        predict_dialog.setObjectName("predict_dialog")
        predict_dialog.resize(336, 197)
        self.gridLayout = QtWidgets.QGridLayout(predict_dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")

        # Chose Dataset Line
        self.dataset_title_label = QtWidgets.QLabel(predict_dialog)
        self.dataset_title_label.setObjectName("dataset_title_label")
        self.dataset_title_label.setText("Dataset folder:")
        self.verticalLayout.addWidget(self.dataset_title_label)

        self.dataset_layout = QtWidgets.QHBoxLayout()
        self.dataset_layout.setContentsMargins(-1, 10, -1, 10)

        self.chosen_dataset_label = QtWidgets.QLabel(predict_dialog)
        self.chosen_dataset_label.setObjectName("chosen_dataset_label")
        metrics = QtGui.QFontMetrics(self.chosen_dataset_label.font())
        elided_text = metrics.elidedText(self.parent.dir_name, QtCore.Qt.ElideRight, 400)
        self.chosen_dataset_label.setText(elided_text)

        self.chosen_dataset_button = QtWidgets.QPushButton(predict_dialog)
        self.chosen_dataset_button.setObjectName("chosen_dataset_button")
        self.chosen_dataset_button.setText("Change Dataset")
        self.chosen_dataset_button.clicked.connect(self.change_dataset_click)
        self.dataset_layout.addWidget(self.chosen_dataset_label)
        self.dataset_layout.addWidget(self.chosen_dataset_button)

        self.verticalLayout.addLayout(self.dataset_layout)
        self.line1 = QtWidgets.QFrame(predict_dialog)
        self.line1.setFrameShape(QtWidgets.QFrame.HLine)
        self.line1.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line1.setObjectName("line1")
        self.verticalLayout.addWidget(self.line1)

        # Chose Model Line
        self.model_title_label = QtWidgets.QLabel(predict_dialog)
        self.model_title_label.setObjectName("model_title_label")
        self.model_title_label.setText("Model")
        self.model_title_label.setContentsMargins(-1, 10, -1, 10)
        self.verticalLayout.addWidget(self.model_title_label)

        self.model_layout = QtWidgets.QHBoxLayout()
        self.model_layout.setContentsMargins(-1, 10, -1, 10)

        self.chosen_model_label = QtWidgets.QLabel(predict_dialog)
        self.chosen_model_label.setObjectName("chosen_model_label")
        metrics = QtGui.QFontMetrics(self.chosen_model_label.font())
        if self.parent.model_name == "":
            elided_text = metrics.elidedText("Please pick a model.", QtCore.Qt.ElideRight, 400)
        else:
            elided_text = metrics.elidedText(self.parent.model_name, QtCore.Qt.ElideRight, 400)
            self.model_name = self.parent.model_name
        self.chosen_model_label.setText(elided_text)
        self.chosen_model_button = QtWidgets.QPushButton(predict_dialog)
        self.chosen_model_button.setObjectName("chosen_model_button")
        self.chosen_model_button.setText("Change Model")
        self.chosen_model_button.clicked.connect(self.change_model_click)

        self.model_layout.addWidget(self.chosen_model_button)

        self.verticalLayout.addWidget(self.chosen_model_label)
        self.verticalLayout.addLayout(self.model_layout)

        self.line2 = QtWidgets.QFrame(predict_dialog)
        self.line2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line2.setObjectName("line2")
        self.verticalLayout.addWidget(self.line2)

        self.predict_progress_bar = QtWidgets.QProgressBar(predict_dialog)
        self.predict_progress_bar.setProperty("value", 0)
        self.predict_progress_bar.setObjectName("predict_progress_bar")
        self.verticalLayout.addWidget(self.predict_progress_bar)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSpacing(6)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.stop_predict_button = QtWidgets.QPushButton(predict_dialog)
        self.stop_predict_button.setObjectName("pushButton_2")
        self.stop_predict_button.clicked.connect(self.on_click_stop_button)

        self.start_predict_button = QtWidgets.QPushButton(predict_dialog)
        self.start_predict_button.setObjectName("pushButton_3")
        self.start_predict_button.clicked.connect(self.on_click_start_button)
        self.start_predict_button.setText("Predict")

        if self.parent.model_name == "":
            self.start_predict_button.setEnabled(False)

        self.horizontalLayout.addWidget(self.stop_predict_button)
        self.horizontalLayout.addWidget(self.start_predict_button)

        self.verticalLayout.addLayout(self.horizontalLayout)
        self.gridLayout.addLayout(self.verticalLayout, 0, 0, 1, 1)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label = QtWidgets.QLabel(predict_dialog)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)

        self.label.setFont(font)
        self.label.setObjectName("label")
        self.verticalLayout_2.addWidget(self.label)

        self.image_predicted_label = QtWidgets.QLabel(predict_dialog)
        self.image_predicted_label.setObjectName("label_2")
        self.verticalLayout_2.addWidget(self.image_predicted_label)

        self.image_progress_label = QtWidgets.QLabel(predict_dialog)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.image_progress_label.setFont(font)
        self.image_progress_label.setObjectName("label_3")
        self.verticalLayout_2.addWidget(self.image_progress_label)
        self.gridLayout.addLayout(self.verticalLayout_2, 1, 0, 1, 1)

        self.retranslateUi(predict_dialog)
        QtCore.QMetaObject.connectSlotsByName(predict_dialog)

    def connect_detection_object(self, detection_object):
        self.detection_object = detection_object
        self.detection_object.det_signals.predict_image_signal.connect(self.handle_progress_signal)
        self.detection_object.det_signals.detection_result.connect(self.handle_detection_result_signal)

    def handle_detection_result_signal(self, val):
        self.detections = val
        self.predicting_result_signal.emit(self.detections)
        self.main_dialog.close()

    def handle_progress_signal(self, val):
        self.predict_progress_bar.setProperty("value", int(val[0]/val[1]*100))
        self.image_progress_label.setText("Image %d / %d" % (val[0], val[1]))
        self.image_predicted_label.setText(val[2])

    def change_dataset_click(self):
        dir_name = QtWidgets.QFileDialog.getExistingDirectory(self.Dialog, 'Chose dataset directory', '.')
        self.parent.load_dataset(dir_name)
        metrics = QtGui.QFontMetrics(self.chosen_dataset_label.font())
        self.dir_name = self.parent.dir_name
        elided_text = metrics.elidedText(self.parent.dir_name, QtCore.Qt.ElideMiddle, 400)
        self.chosen_dataset_label.setText(elided_text)

    def change_model_click(self):
        name, type = QtWidgets.QFileDialog.getOpenFileName(self.Dialog, 'Open Model File', '.',
                                                           "Models (*.hdf5 *.h5);;All Files (*)")
        print(name)
        metrics = QtGui.QFontMetrics(self.chosen_model_label.font())
        elided_text = metrics.elidedText(name, QtCore.Qt.ElideMiddle, 400)
        self.chosen_model_label.setText(elided_text)
        self.parent.new_model_gui_update(name)
        self.model_name = name
        self.start_predict_button.setEnabled(True)

    def on_click_stop_button(self):
        self.stop_predict_button.setText("Stopping inference, please wait..")
        self.stop_predict_button.setEnabled(False)
        self.stop_predicting_signal.emit(True)

    def on_click_start_button(self):
        annot_list =  [[[x[2], x[3]] for x in v] for v in
                                                        self.parent.annotation_manager.annotations_rect.values() if
                                                        len(v) != 0]
        if len(annot_list) > 0:
            annotation_dims = reduce((lambda x, y: x + y), annot_list)
            min_size,max_size = 0,1000000
            if len(annotation_dims) > 0:
                min_size = max(10, np.min(annotation_dims))
                max_size = np.max(annotation_dims)
            size_range = (min_size, max_size)
        else:
            size_range = None

        predict_worker = predicting_worker(self.dir_name, self.parent.annotation_manager.image_names,
                                           predict_dialog=self, size_range=size_range)
        self.parent.threadpool.start(predict_worker)


    def retranslateUi(self, predict_dialog):
        _translate = QtCore.QCoreApplication.translate
        predict_dialog.setWindowTitle(_translate("predict_dialog", "Predicting"))
        self.stop_predict_button.setText(_translate("predict_dialog", "Cancel"))
        self.label.setText(_translate("predict_dialog", "Predicting image:"))
        self.image_predicted_label.setText(_translate("predict_dialog", "Preparing images.."))
        self.image_progress_label.setText(_translate("predict_dialog", "Image 0/0"))




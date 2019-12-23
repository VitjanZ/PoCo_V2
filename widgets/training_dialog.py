from PyQt5 import QtCore, QtGui, QtWidgets,Qt
from PyQt5.QtCore import pyqtSignal
from utils.detection_model import training_worker

class TrainingDialog(QtCore.QObject):

    stop_training_signal = pyqtSignal(bool)
    start_training_signal = pyqtSignal(bool)

    def __init__(self, parent, dir_name, Dialog):
        super(TrainingDialog, self).__init__()
        self.parent = parent
        self.stop_training = False
        self.detection_object = None
        self.model_name = parent.model_name
        self.Dialog = Dialog
        Dialog.setObjectName("Dialog")
        Dialog.resize(315, 156)
        self.gridLayout_2 = QtWidgets.QGridLayout(Dialog)
        self.gridLayout_2.setVerticalSpacing(10)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setContentsMargins(-1, -1, -1, -1)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")

        #Chose Dataset Line
        self.dataset_title_label = QtWidgets.QLabel(Dialog)
        self.dataset_title_label.setObjectName("dataset_title_label")
        self.dataset_title_label.setText("Training Dataset")
        self.verticalLayout.addWidget(self.dataset_title_label)

        self.dataset_layout = QtWidgets.QHBoxLayout()
        self.dataset_layout.setContentsMargins(-1, 10, -1, 10)

        self.chosen_dataset_label = QtWidgets.QLabel(Dialog)
        self.chosen_dataset_label.setObjectName("chosen_dataset_label")
        metrics = QtGui.QFontMetrics(self.chosen_dataset_label.font())
        elided_text = metrics.elidedText(self.parent.dir_name, QtCore.Qt.ElideRight,400)
        self.chosen_dataset_label.setText(elided_text)

        self.chosen_dataset_button = QtWidgets.QPushButton(Dialog)
        self.chosen_dataset_button.setObjectName("chosen_dataset_button")
        self.chosen_dataset_button.setText("Change Dataset")
        self.chosen_dataset_button.clicked.connect(self.change_dataset_click)
        self.dataset_layout.addWidget(self.chosen_dataset_label)
        self.dataset_layout.addWidget(self.chosen_dataset_button)

        self.verticalLayout.addLayout(self.dataset_layout)
        self.line1 = QtWidgets.QFrame(Dialog)
        self.line1.setFrameShape(QtWidgets.QFrame.HLine)
        self.line1.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line1.setObjectName("line1")
        self.verticalLayout.addWidget(self.line1)


        #Chose Model Line
        self.model_title_label = QtWidgets.QLabel(Dialog)
        self.model_title_label.setObjectName("model_title_label")
        self.model_title_label.setText("Model")
        self.model_title_label.setContentsMargins(-1, 10, -1, 10)
        self.verticalLayout.addWidget(self.model_title_label)


        self.model_layout = QtWidgets.QHBoxLayout()
        self.model_layout.setContentsMargins(-1, 10, -1, 10)

        self.chosen_model_label = QtWidgets.QLabel(Dialog)
        self.chosen_model_label.setObjectName("chosen_model_label")
        metrics = QtGui.QFontMetrics(self.chosen_model_label.font())
        elided_text = metrics.elidedText(self.parent.model_name, QtCore.Qt.ElideRight,400)
        self.chosen_model_label.setText(elided_text)
        self.chosen_model_button = QtWidgets.QPushButton(Dialog)
        self.chosen_model_button.setObjectName("chosen_model_button")
        self.chosen_model_button.setText("Change Model")
        self.chosen_model_button.clicked.connect(self.change_model_click)
        self.new_model_button = QtWidgets.QPushButton(Dialog)
        self.new_model_button.setObjectName("new_model_button")
        self.new_model_button.setText("Create New Model")
        self.new_model_button.clicked.connect(self.new_model_click)

        self.model_layout.addWidget(self.new_model_button)
        self.model_layout.addWidget(self.chosen_model_button)

        self.verticalLayout.addWidget(self.chosen_model_label)
        self.verticalLayout.addLayout(self.model_layout)

        self.line2 = QtWidgets.QFrame(Dialog)
        self.line2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line2.setObjectName("line2")
        self.verticalLayout.addWidget(self.line2)

        self.epoch_label = QtWidgets.QLabel(Dialog)
        self.epoch_label.setObjectName("epoch_label")
        self.verticalLayout.addWidget(self.epoch_label)
        self.error_label = QtWidgets.QLabel(Dialog)
        self.error_label.setObjectName("error_label")
        self.verticalLayout.addWidget(self.error_label)
        self.line = QtWidgets.QFrame(Dialog)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout.addWidget(self.line)
        self.percentage_label = QtWidgets.QLabel(Dialog)
        self.percentage_label.setObjectName("percentage_label")
        self.verticalLayout.addWidget(self.percentage_label)
        self.train_progress_bar = QtWidgets.QProgressBar(Dialog)
        self.train_progress_bar.setProperty("value", 0)
        self.train_progress_bar.setObjectName("train_progress_bar")
        self.verticalLayout.addWidget(self.train_progress_bar)
        self.line_2 = QtWidgets.QFrame(Dialog)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.verticalLayout.addWidget(self.line_2)

        self.train_button_layout = QtWidgets.QHBoxLayout()
        self.train_button_layout.setContentsMargins(-1, -1, -1, -1)
        self.train_button_layout.setSpacing(0)
        self.train_button_layout.setObjectName("train_button_layout")
        self.stop_training_button = QtWidgets.QPushButton(Dialog)
        self.stop_training_button.setObjectName("stop_training_button")
        self.stop_training_button.clicked.connect(self.on_click_stop_button)
        self.stop_training_button.setText("Stop Training")

        self.start_training_button = QtWidgets.QPushButton(Dialog)
        self.start_training_button.setObjectName("start_training_button")
        self.start_training_button.clicked.connect(self.on_click_start_button)
        self.start_training_button.setText("Start Training")
        self.train_button_layout.addWidget(self.start_training_button)
        self.train_button_layout.addWidget(self.stop_training_button)
        self.verticalLayout.addLayout(self.train_button_layout)

        self.gridLayout_2.addLayout(self.verticalLayout, 0, 0, 1, 1)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def change_dataset_click(self):
        dir_name = QtWidgets.QFileDialog.getExistingDirectory(self.Dialog, 'Chose dataset directory', '.')
        self.parent.load_dataset(dir_name)
        metrics = QtGui.QFontMetrics(self.chosen_dataset_label.font())
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

    def new_model_click(self):
        name, type = QtWidgets.QFileDialog.getSaveFileName(self.Dialog, 'Create Model File','.',
                                                           "Models (*.hdf5 *.h5);;All Files (*)")
        print(name)
        metrics = QtGui.QFontMetrics(self.chosen_model_label.font())
        elided_text = metrics.elidedText(name, QtCore.Qt.ElideMiddle, 400)
        self.chosen_model_label.setText(elided_text)
        self.parent.new_model_gui_update(name)
        self.model_name = name

    def handle_epoch_signal(self, val):
        self.epoch_label.setText("Epoch %d" % (val))

    def handle_loss_signal(self, val):
        self.error_label.setText("Loss %f" % (val))

    def handle_progress_signal(self, val):
        self.train_progress_bar.setProperty("value", val)

    def handle_result_signal(self, detections):
        pass

    def connect_detection_object(self, detection_object):
        self.detection_object = detection_object
        self.detection_object.det_signals.epoch_signal.connect(self.handle_epoch_signal)
        self.detection_object.det_signals.loss_signal.connect(self.handle_loss_signal)
        self.detection_object.det_signals.progress_signal.connect(self.handle_progress_signal)


    def on_click_stop_button(self):
        self.stop_training = True
        self.stop_training_button.setText("Stopping training, please wait..")
        self.stop_training_button.setEnabled(False)
        self.stop_training_signal.emit(True)

    def on_click_start_button(self):
        self.start_training_button.setText("Started Training")
        self.start_training_button.setEnabled(False)
        self.start_training_signal.emit(True)
        train_worker = training_worker(self.parent.annotation_manager.dir_name, self.parent.annotation_manager.image_names,
                                       train_dialog=self)
        self.parent.threadpool.start(train_worker)



    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Model training"))
        self.epoch_label.setText(_translate("Dialog", "Epoch: 0"))
        self.error_label.setText(_translate("Dialog", "Loss: ..."))
        self.percentage_label.setText(_translate("Dialog", "Percentage of epochs processed:"))
        self.stop_training_button.setText(_translate("Dialog", "Stop Training"))

from PyQt5 import QtCore, QtGui, QtWidgets

class prepare_data_dialog(QtCore.QObject):

    stop_prepare_signal = QtCore.pyqtSignal(bool)
    predicting_result_signal = QtCore.pyqtSignal(object)

    def __init__(self, parent, Dialog):
        super(prepare_data_dialog, self).__init__()
        self.parent = parent
        self.detection_object = None
        self.model_name = ""
        self.Dialog = Dialog
        self.dir_name = self.parent.dir_name

    def setupUi(self, prepare_data_dialog):
        self.main_dialog = prepare_data_dialog
        prepare_data_dialog.setObjectName("prepare_data_dialog")
        prepare_data_dialog.resize(336, 197)
        self.gridLayout = QtWidgets.QGridLayout(prepare_data_dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")

        # Chose Model Line
        self.model_title_label = QtWidgets.QLabel(prepare_data_dialog)
        self.model_title_label.setObjectName("model_title_label")
        self.model_title_label.setText("Saving data:")
        self.model_title_label.setContentsMargins(-1, 10, -1, 10)
        self.verticalLayout.addWidget(self.model_title_label)


        self.predict_progress_bar = QtWidgets.QProgressBar(prepare_data_dialog)
        self.predict_progress_bar.setProperty("value", 0)
        self.predict_progress_bar.setObjectName("predict_progress_bar")
        self.verticalLayout.addWidget(self.predict_progress_bar)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSpacing(6)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.stop_predict_button = QtWidgets.QPushButton(prepare_data_dialog)
        self.stop_predict_button.setObjectName("pushButton_2")
        self.stop_predict_button.clicked.connect(self.on_click_stop_button)


        self.horizontalLayout.addWidget(self.stop_predict_button)

        self.verticalLayout.addLayout(self.horizontalLayout)
        self.gridLayout.addLayout(self.verticalLayout, 0, 0, 1, 1)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label = QtWidgets.QLabel(prepare_data_dialog)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)

        self.label.setFont(font)
        self.label.setObjectName("label")
        self.verticalLayout_2.addWidget(self.label)


        self.image_progress_label = QtWidgets.QLabel(prepare_data_dialog)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.image_progress_label.setFont(font)
        self.image_progress_label.setObjectName("label_3")
        self.verticalLayout_2.addWidget(self.image_progress_label)
        self.gridLayout.addLayout(self.verticalLayout_2, 1, 0, 1, 1)

        self.retranslateUi(prepare_data_dialog)
        QtCore.QMetaObject.connectSlotsByName(prepare_data_dialog)

    def connect_detection_object(self, detection_object):
        self.detection_object = detection_object
        self.detection_object.signals.progress_signal.connect(self.handle_progress_signal)
        self.detection_object.signals.result_signal.connect(self.handle_result_signal)

    def handle_result_signal(self, val):
        self.parent.main_window.statusBar().showMessage("Saved dataset to: " + self.dir_name)
        self.main_dialog.close()

    def handle_progress_signal(self, val):
        self.predict_progress_bar.setProperty("value", int(val[0]/val[1]*100))
        self.image_progress_label.setText("Image %d / %d" % (val[0], val[1]))

    def on_click_stop_button(self):
        self.stop_predict_button.setText("Stopping the process, please wait..")
        self.stop_predict_button.setEnabled(False)
        self.stop_prepare_signal.emit(True)


    def retranslateUi(self, prepare_data_dialog):
        _translate = QtCore.QCoreApplication.translate
        prepare_data_dialog.setWindowTitle(_translate("prepare_data_dialog", "Saving.."))
        self.stop_predict_button.setText(_translate("prepare_data_dialog", "Cancel"))
        self.label.setText(_translate("prepare_data_dialog", "Saving training images:"))
        self.image_progress_label.setText(_translate("prepare_data_dialog", "Image 0/0"))




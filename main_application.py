
from PyQt5 import QtCore, QtGui, QtWidgets
from widgets.image_display import *
from widgets.annotation_manager import annotation_manager
import sys

sys.path.append('..')
#Only used for training
from widgets.prepare_data_dialog import prepare_data_dialog
from utils.prepare_data import prepare_worker
import os


class Ui_MainWindow(object):

    def __init__(self, MainWindow):
        self.image_names = []
        self.training_images = []
        self.dir_name = None
        self.model_name = ""
        self.threadpool = QtCore.QThreadPool()

        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1094, 839)
        self.main_window = MainWindow

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.centralwidget.setStyleSheet("background-color:rgb(80,82,89)")

        self.number_annotations_label = QtWidgets.QLabel(self.centralwidget)
        self.number_annotations_label.setObjectName("number_annotations_label")
        self.number_annotations_label.setStyleSheet("color:rgb(200,200,210)")

        self.annotation_manager = annotation_manager(self.centralwidget)

        self.main_image_graphics_view = ImageDisplayView(self.centralwidget)
        self.scene = GraphicsScene(self.annotation_manager, self.number_annotations_label,
                                   parent=self.main_image_graphics_view)
        self.scene.curr_view = self.main_image_graphics_view

        self.main_image_graphics_view.setScene(self.scene)
        self.main_image_graphics_view.resize(self.scene.width(), self.scene.height())

        self.horizontalLayout.addWidget(self.main_image_graphics_view)
        self.horizontalLayout_2.addLayout(self.horizontalLayout)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")

        self.load_dataset_button = QtWidgets.QPushButton(self.centralwidget)
        self.load_dataset_button.setObjectName("load_dataset_button")
        self.load_dataset_button.clicked.connect(self.load_dataset_click)
        self.load_dataset_button.setStyleSheet("background-color:rgb(120,120,130); color:rgb(200,200,210)");
        self.load_dataset_button.setToolTip("Loads a folder of images into the program.")
        self.verticalLayout_2.addWidget(self.load_dataset_button)

        self.pick_model_button = QtWidgets.QPushButton(self.centralwidget)
        self.pick_model_button.setObjectName("pick_model_button")
        self.pick_model_button.clicked.connect(self.pick_model_click)
        self.pick_model_button.setStyleSheet("background-color:rgb(120,120,130); color:rgb(200,200,210)");
        self.pick_model_button.setToolTip("Change the model file.")
        self.verticalLayout_2.addWidget(self.pick_model_button)


        self.save_training_dataset_button = QtWidgets.QPushButton(self.centralwidget)
        self.save_training_dataset_button.setObjectName("save_training_dataset_button")
        self.save_training_dataset_button.clicked.connect(self.save_training_data)
        self.save_training_dataset_button.setStyleSheet("background-color:rgb(120,120,130); color:rgb(200,200,210)");
        self.save_training_dataset_button.setToolTip("Saves the current annotations and detections.")
        self.verticalLayout_2.addWidget(self.save_training_dataset_button)

        self.train_on_dataset_button = QtWidgets.QPushButton(self.centralwidget)
        self.train_on_dataset_button.setObjectName("train_on_dataset_button")
        self.train_on_dataset_button.clicked.connect(self.train_on_data)
        self.train_on_dataset_button.setStyleSheet("background-color:rgb(120,120,130); color:rgb(200,200,210)");
        self.train_on_dataset_button.setEnabled(True)
        self.train_on_dataset_button.setToolTip(
            "Runs the training procedure for the images marked as part of the training set.")
        self.verticalLayout_2.addWidget(self.train_on_dataset_button)

        self.predict_with_model_button = QtWidgets.QPushButton(self.centralwidget)
        self.predict_with_model_button.setObjectName("predict_with_model_button")
        self.predict_with_model_button.clicked.connect(self.predict_data)
        self.predict_with_model_button.setStyleSheet("background-color:rgb(120,120,130); color:rgb(200,200,210)");
        self.predict_with_model_button.setToolTip(
            "Detects the structures with the current trained model on all images.")
        self.verticalLayout_2.addWidget(self.predict_with_model_button)

        # TOOL BAR ########################
        bot_frame = QtWidgets.QFrame()
        bot_frame.setFrameShape(QtWidgets.QFrame.HLine)
        bot_frame.setFrameShadow(QtWidgets.QFrame.Sunken)

        self.verticalLayout_2.addWidget(bot_frame)
        self.tools_label = QtWidgets.QLabel(self.centralwidget)
        self.tools_label.setText("Tools")
        self.tools_label.setStyleSheet("color:rgb(200,200,210)")
        self.tools_label.setAlignment(Qt.AlignCenter)
        self.verticalLayout_2.addWidget(self.tools_label)

        self.horizontal_layout_tools = QtWidgets.QHBoxLayout()
        self.annotation_plus_tool = QtWidgets.QPushButton(self.centralwidget)
        self.annotation_plus_tool.setCheckable(True)
        self.annotation_plus_tool.setChecked(True)
        self.annotation_plus_tool.setIcon(QtGui.QIcon('./resources/icons/annotation_plus.png'))
        self.annotation_plus_tool.setIconSize(QtCore.QSize(30, 30))
        self.annotation_plus_tool.setFlat(True)
        self.annotation_plus_tool.setToolTip("Adds or removes one annotation at a time.")
        self.annotation_plus_tool.clicked.connect(self.pick_add_tool_click)

        self.annotation_minus_tool = QtWidgets.QPushButton(self.centralwidget)
        self.annotation_minus_tool.setCheckable(True)
        self.annotation_minus_tool.setIcon(QtGui.QIcon('./resources/icons/annotation_delete.png'))
        self.annotation_minus_tool.setIconSize(QtCore.QSize(30, 30))
        self.annotation_minus_tool.setFlat(True)
        self.annotation_minus_tool.setToolTip("Removes annotations in an area.")
        self.annotation_minus_tool.clicked.connect(self.pick_remove_tool_click)

        # for adding negative annotations
        self.annotation_negative_tool = QtWidgets.QPushButton(self.centralwidget)
        self.annotation_negative_tool.setCheckable(True)
        self.annotation_negative_tool.setIcon(QtGui.QIcon('./resources/icons/annotation_minus.png'))
        self.annotation_negative_tool.setIconSize(QtCore.QSize(30, 30))
        self.annotation_negative_tool.setFlat(True)
        self.annotation_negative_tool.setToolTip("Marks a region with no structures of interest as a training example.")
        self.annotation_negative_tool.clicked.connect(self.annotation_negative_tool_click)


        #"""
        self.polygon_tool = QtWidgets.QPushButton(self.centralwidget)
        self.polygon_tool.setCheckable(True)
        self.polygon_tool.setIcon(QtGui.QIcon('./resources/icons/poly_tool.png'))
        self.polygon_tool.setIconSize(QtCore.QSize(30, 30))
        self.polygon_tool.setFlat(True)
        self.polygon_tool.setToolTip("Marks the region of interest rectangle for training and detection.")
        self.polygon_tool.clicked.connect(self.pick_poly_tool_click)
        #"""


        self.horizontal_layout_tools.addWidget(self.annotation_plus_tool)
        self.horizontal_layout_tools.addWidget(self.annotation_negative_tool)
        self.horizontal_layout_tools.addWidget(self.annotation_minus_tool)
        self.horizontal_layout_tools.addWidget(self.polygon_tool)

        self.verticalLayout_2.addLayout(self.horizontal_layout_tools)
        bot_frame = QtWidgets.QFrame()
        bot_frame.setFrameShape(QtWidgets.QFrame.HLine)
        bot_frame.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.verticalLayout_2.addWidget(bot_frame)
        # REMOVAL TOOL ##############

        self.brush_tool_options_label = QtWidgets.QLabel(self.centralwidget)
        self.brush_tool_options_label.setText("Brush size")
        self.brush_tool_options_label.setStyleSheet("color:rgb(200,200,210)")
        self.brush_tool_options_label.setAlignment(Qt.AlignCenter)
        self.verticalLayout_2.addWidget(self.brush_tool_options_label)

        self.horizontal_layout_removal_tool = QtWidgets.QHBoxLayout()
        self.brush_tool_size_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.brush_tool_size_slider.setMinimum(0)
        self.brush_tool_size_slider.setMaximum(1000)
        self.brush_tool_size_slider.setTickInterval(1)
        self.brush_tool_size_slider.setValue(100)
        self.brush_tool_size_slider.valueChanged.connect(self.removal_slider_changed)

        self.brush_tool_size_edit = QtWidgets.QLineEdit()
        self.brush_tool_size_edit.setStyleSheet("color:rgb(200,200,210)")
        self.only_int_validator = QtGui.QIntValidator(0, 1000)
        self.brush_tool_size_edit.setValidator(self.only_int_validator)
        self.brush_tool_size_edit.setText("100")
        self.brush_tool_size_edit.textChanged.connect(self.removal_text_edit_changed)


        self.horizontal_layout_removal_tool.addWidget(self.brush_tool_size_slider, 3)
        self.horizontal_layout_removal_tool.addWidget(self.brush_tool_size_edit, 1)

        self.verticalLayout_2.addLayout(self.horizontal_layout_removal_tool)

        self.brush_tool_size_slider.setVisible(False)
        self.brush_tool_options_label.setVisible(False)
        self.brush_tool_size_edit.setVisible(False)


        bot_frame = QtWidgets.QFrame()
        bot_frame.setFrameShape(QtWidgets.QFrame.HLine)
        bot_frame.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.verticalLayout_2.addWidget(bot_frame)

        # TOOL BAR END ###########################
        self.images_dataset_label = QtWidgets.QLabel("Dataset Images")
        self.images_dataset_label.setAlignment(Qt.AlignCenter)
        self.images_dataset_label.setStyleSheet("color:rgb(200,200,210)")
        self.verticalLayout_2.addWidget(self.images_dataset_label)

        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout_2.addItem(spacerItem)
        self.selected_model_label = QtWidgets.QLabel(self.centralwidget)
        self.selected_model_label.setObjectName("selected_model_label")
        self.selected_model_label.setStyleSheet("color:rgb(200,200,210)")
        self.verticalLayout_2.addWidget(self.selected_model_label)

        self.selected_file_label = QtWidgets.QLabel(self.centralwidget)
        self.selected_file_label.setObjectName("selected_file_label")
        self.selected_file_label.setStyleSheet("color:rgb(200,200,210)")
        self.verticalLayout_2.addWidget(self.selected_file_label)

        self.verticalLayout_2.addWidget(self.number_annotations_label)

        bot_frame = QtWidgets.QFrame()
        bot_frame.setFrameShape(QtWidgets.QFrame.HLine)
        bot_frame.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.verticalLayout_2.addWidget(bot_frame)

        # LIST LABEL SETUP
        self.horizontal_list_label_layout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.list_train_label = QtWidgets.QLabel(self.centralwidget)
        self.list_train_label.setObjectName("list_train_label")
        self.list_train_label.setStyleSheet("color:rgb(200,200,210)")
        self.list_train_label.setText("Train")
        self.list_train_label.setToolTip("Checked images are included in the training set.")

        self.list_names_label = QtWidgets.QLabel(self.centralwidget)
        self.list_names_label.setObjectName("list_names_label")
        self.list_names_label.setStyleSheet("color:rgb(200,200,210)")
        self.list_names_label.setText("Image files")

        self.horizontal_list_spacer = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding,
                                                            QtWidgets.QSizePolicy.Minimum)
        self.horizontal_list_label_layout.addWidget(self.list_train_label)
        self.horizontal_list_label_layout.addWidget(self.list_names_label)
        self.horizontal_list_label_layout.addItem(self.horizontal_list_spacer)
        self.verticalLayout_2.addLayout(self.horizontal_list_label_layout)


        self.file_list_list_widget = QtWidgets.QListWidget(self.centralwidget)
        self.file_list_list_widget.setGeometry(QtCore.QRect(0, 0, 269, 501))
        self.file_list_list_widget.setObjectName("file_list_list_widget")
        self.file_list_list_widget.itemClicked.connect(self.list_item_clicked)
        self.verticalLayout_2.addWidget(self.file_list_list_widget)
        self.horizontalLayout_2.addLayout(self.verticalLayout_2)
        self.horizontalLayout_2.setStretch(0, 4)
        self.horizontalLayout_2.setStretch(1, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1094, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # SHORTCUTS
        save_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+S"), MainWindow)
        save_shortcut.activated.connect(self.save_training_data)

        undo_shorcut = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+Z"), MainWindow)
        undo_shorcut.activated.connect(self.scene.handle_undo_action)

    def load_dataset_click(self):
        dir_name = QtWidgets.QFileDialog.getExistingDirectory(self.centralwidget, 'Chose dataset directory', '.')
        self.load_dataset(dir_name)

    def load_dataset(self,dir_name):
        image_paths = self.annotation_manager.load_annotations(dir_name)
        if len(image_paths) == 0:
            self.main_window.statusBar().showMessage(
                "No directory chosen or there were no suitable images in the chosen directory.")
        else:
            self.file_list_list_widget.clear()
            self.dir_name = dir_name
            self.file_list_list_widget.setStyleSheet("color:rgb(200,200,210)")
            for name in image_paths:
                t_name = name.replace('\\', '/').split("/")[-1]
                self.image_names.append(t_name)
                item = QtWidgets.QListWidgetItem(t_name)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Unchecked)
                self.file_list_list_widget.addItem(item)

            self.file_list_list_widget.setCurrentRow(0)
            self.list_item_clicked()
            self.main_window.statusBar().showMessage("Chosen directory: " + self.annotation_manager.dir_name)

    def new_model_gui_update(self,name):
        self.model_name = name
        self.main_window.statusBar().showMessage("Chosen a new model: " + self.model_name)
        self.selected_model_label.setText("Selected model: " + str(os.path.basename(self.model_name)))

    def pick_model_click(self):
        fname = QtWidgets.QFileDialog.getOpenFileName(self.centralwidget, 'Open file', '.', "Model Files (*.h5 *.hdf5)")
        self.new_model_gui_update(fname[0])

    def save_dataset_click(self):
        self.main_window.statusBar().showMessage("Saving dataset.")

        if self.scene != None:
            close = QtWidgets.QMessageBox()
            close.setWindowTitle("Save data confirm")
            close.setStyleSheet("background-color:rgb(120,120,130)")
            close.setText("Are you sure you want to update your annotation files?")
            close.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel)
            close = close.exec()

            if close == QtWidgets.QMessageBox.Yes:
                self.scene.saved = True
                self.annotation_manager.save_annotations()
                self.main_window.statusBar().showMessage("Saving dataset to: " + self.annotation_manager.dir_name)

                self.threadpool.clear()
                Dialog = QtWidgets.QDialog()
                prepare_dialog = prepare_data_dialog(self, Dialog)
                prepare_dialog.setupUi(Dialog)

                self.prepare_dialog = Dialog
                prepare_data_worker = prepare_worker(self.annotation_manager,data_dialog=prepare_dialog)
                prepare_dialog.connect_detection_object(prepare_data_worker.data_handler)
                self.threadpool.start(prepare_data_worker)
                Dialog.exec()

            else:
                self.main_window.statusBar().showMessage("Saving dataset canceled.")


    # REMOVAL TOOL CALLBACKS #################
    def removal_slider_changed(self):
        input_val = self.brush_tool_size_slider.value()
        self.brush_tool_size_edit.setText(str(input_val))
        self.scene.removal_circle_radius = input_val

    def removal_text_edit_changed(self):
        input_str = self.brush_tool_size_edit.text()
        if input_str == '':
            input_str = '0'
        valid = self.only_int_validator.validate(input_str, 0)
        if valid != QtGui.QIntValidator.Invalid:
            value = int(input_str)
            if value > self.brush_tool_size_slider.maximum():
                value = self.brush_tool_size_slider.maximum()
            self.brush_tool_size_slider.setValue(value)
            self.scene.removal_circle_radius = value
        else:
            self.brush_tool_size_edit.setText(str(self.brush_tool_size_slider.value()))

    # TOOL PICKING ##################

    def handle_tool_click(self, tool):
        tools = [self.annotation_plus_tool, self.annotation_minus_tool, self.annotation_negative_tool,
                 self.polygon_tool]
        self.scene.picked_tool = tool
        if tool != 1 and tool != 4:
            # if tool not annotation minus tool
            self.brush_tool_size_slider.setVisible(False)
            self.brush_tool_options_label.setVisible(False)
            self.brush_tool_size_edit.setVisible(False)

        for t in range(len(tools)):
            if t == tool:
                tools[t].setChecked(True)
            else:
                tools[t].setChecked(False)

    def pick_remove_tool_click(self):
        self.handle_tool_click(1)
        self.brush_tool_size_slider.setVisible(True)
        self.brush_tool_options_label.setVisible(True)
        self.brush_tool_size_edit.setVisible(True)

    def pick_add_tool_click(self):
        self.handle_tool_click(0)
        if self.scene.removal_circle_item != None:
            self.scene.removeItem(self.scene.removal_circle_item)
            self.scene.removal_circle_item = None

    def pick_poly_tool_click(self):
        self.handle_tool_click(3)

    def annotation_negative_tool_click(self):
        self.handle_tool_click(2)


    # TOOL PICKING END ################

    def save_training_data(self):
        self.save_dataset_click()
        self.train_on_dataset_button.setEnabled(True)

    def list_item_clicked(self):
        selected_item = self.file_list_list_widget.currentItem()
        dir_name = self.annotation_manager.dir_name
        self.scene.change_image(dir_name + "/" + selected_item.text(), selected_item.text())
        self.selected_file_label.setText(selected_item.text())
        self.main_window.setWindowTitle(selected_item.text() + " - PoCo - v2")

    def train_on_data(self):
        from widgets.training_dialog import TrainingDialog
        self.threadpool.clear()
        Dialog = QtWidgets.QDialog()
        train_dialog = TrainingDialog(self, self.dir_name, Dialog)
        self.train_dialog = Dialog
        Dialog.exec()

    def handle_dataset_attribute_signal(self, attrs):
        # get std, mean of training set
        self.dataset_attributes = attrs


    def predict_data(self):
        from widgets.predict_dialog import predict_dialog
        self.threadpool.clear()
        Dialog = QtWidgets.QDialog()
        pred_dialog = predict_dialog(self,Dialog)
        pred_dialog.setupUi(Dialog)
        pred_dialog.predicting_result_signal.connect(self.handle_detection_result_signal)
        Dialog.exec()

    def handle_detection_result_signal(self, detections):
        self.annotation_manager.change_annotations(detections)
        self.scene.change_annotations_on_image()
        self.threadpool.clear()
        self.threadpool = QtCore.QThreadPool()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "PoCo - v2"))
        self.load_dataset_button.setText(_translate("MainWindow", "Load Dataset"))
        self.pick_model_button.setText(_translate("MainWindow", "Change Model"))

        self.save_training_dataset_button.setText(_translate("MainWindow", "Save Training Dataset"))
        self.train_on_dataset_button.setText(_translate("MainWindow", "Train on current dataset"))
        self.predict_with_model_button.setText(_translate("MainWindow", "Predict with model"))
        self.selected_model_label.setText(_translate("MainWindow", "Selected Model: None"))

        self.selected_file_label.setText(_translate("MainWindow", "Selected file: None"))
        self.number_annotations_label.setText(_translate("MainWindow", "Number of annotations: 0"))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        quit_action = QtWidgets.QAction("Quit", self)
        quit_action.triggered.connect(self.close)
        self.scene = None

    def closeEvent(self, event):
        close = QtWidgets.QMessageBox()
        close.setWindowTitle("Confirm Exit")
        close.setStyleSheet("background-color:rgb(120,120,130)")
        if self.scene != None and self.scene.saved == False:
            close.setText("You have unsaved data. Are you sure you want to quit?")
        else:
            close.setText("Are you sure you want to quit?")

        close.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel)
        close = close.exec()

        if close == QtWidgets.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    pixmap = QPixmap('./resources/icons/splash.png')
    splash = QtWidgets.QSplashScreen(pixmap)
    splash.show()
    splash.showMessage("Loading")
    app.processEvents()

    #trayIcon = QtWidgets.QSystemTrayIcon(QtGui.QIcon("./resources/icons/logo.png"), app)
    MainWindow = MainWindow()
    MainWindow.setWindowIcon(QtGui.QIcon('./resources/icons/logo.png'))
    #trayIcon.show()
    ui = Ui_MainWindow(MainWindow)

    MainWindow.scene = ui.scene
    MainWindow.statusBar().setStyleSheet("background-color:rgb(120,120,130)")
    MainWindow.statusBar().showMessage("PoCo-v2.")
    MainWindow.showMaximized()
    splash.finish(MainWindow)
    sys.exit(app.exec_())

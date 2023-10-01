import cv2
import sys
import numpy as np

import pandas as pd
from PyQt5.QtCore import pyqtSignal, QThread, Qt
from PyQt5 import QtCore, QtGui, QtWidgets

from image_helper_functions import *
from deepface_functions import *
from deepface_models import *


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self._run_flag = True
        self.camera = cv2.VideoCapture(0)
        self.WIDTH = 1920
        self.HEIGHT = 1080
        self.pause = False

    def run(self):
        while self._run_flag:
            if not self.pause:
                try:
                    _, img = self.camera.read() # Read in as 1920x1080p                
                    img = cv2.flip(img, 1)
                    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    face_detected = getImageContents(image)
                    if self.parent.predict_avail:
                        if face_detected is not None:
                            image = np.multiply(face_detected[0], 255).astype(np.uint8)[0]
                            result, image = self.parent.predict(image)
                            self.parent.impersonation_result.setText(result) # Update impersonation result to GUI
                    self.change_pixmap_signal.emit(image)
                
                except Exception as e:
                    print("in thread")
                    print(e)
                    self.change_pixmap_signal.emit(image)
            else:
                continue
    
    def pause_stream(self):
        self.pause = True
    
    def resume_stream(self):
        self.pause = False


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1277, 978)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.widget_layout = QtWidgets.QVBoxLayout()
        self.widget_layout.setObjectName("widget_layout")
        self.result_layout = QtWidgets.QGridLayout()
        self.result_layout.setObjectName("result_layout")

        self.impersonation_result = QtWidgets.QLabel(self.centralwidget)
        self.impersonation_result.setMinimumSize(QtCore.QSize(0, 100))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.impersonation_result.setFont(font)
        self.impersonation_result.setText("")
        self.impersonation_result.setAlignment(QtCore.Qt.AlignCenter)
        self.impersonation_result.setObjectName("impersonation_result")
        self.result_layout.addWidget(self.impersonation_result, 0, 1, 1, 1)
        
        self.impersonation_label = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.impersonation_label.sizePolicy().hasHeightForWidth())
        self.impersonation_label.setSizePolicy(sizePolicy)
        self.impersonation_label.setMinimumSize(QtCore.QSize(0, 100))
        self.impersonation_label.setMaximumSize(QtCore.QSize(16777215, 100))

        self.impersonation_label.setFont(font)
        self.impersonation_label.setAlignment(QtCore.Qt.AlignCenter)
        self.impersonation_label.setObjectName("impersonation_label")
        self.result_layout.addWidget(self.impersonation_label, 0, 0, 1, 1)

        self.widget_layout.addLayout(self.result_layout)
        
        self.video_display = QtWidgets.QLabel(self.centralwidget)
        self.video_display.setPixmap(QtGui.QPixmap())
        self.video_display.setAlignment(QtCore.Qt.AlignCenter)

        self.video_display.setObjectName("video_display")
        self.widget_layout.addWidget(self.video_display)

        self.attack_layout = QtWidgets.QHBoxLayout()
        self.attack_layout.setObjectName("attack_layout")
        self.digital_attack_btn = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.digital_attack_btn.sizePolicy().hasHeightForWidth())
        self.digital_attack_btn.setSizePolicy(sizePolicy)
        self.digital_attack_btn.setMinimumSize(QtCore.QSize(0, 50))

        font.setPointSize(15)
        self.digital_attack_btn.setFont(font)
        self.digital_attack_btn.setObjectName("digital_attack_btn")
        self.digital_attack_btn.clicked.connect(lambda: self.set_attack_mode("digital"))
        self.attack_layout.addWidget(self.digital_attack_btn)
        self.physical_attack_btn = QtWidgets.QPushButton(self.centralwidget)
        self.physical_attack_btn.setSizePolicy(sizePolicy)
        self.physical_attack_btn.setFont(font)
        self.physical_attack_btn.setObjectName("physical_attack_btn")
        self.physical_attack_btn.setMinimumSize(QtCore.QSize(0, 50))
        self.physical_attack_btn.clicked.connect(lambda: self.set_attack_mode("physical"))
        self.attack_layout.addWidget(self.physical_attack_btn)
        self.widget_layout.addLayout(self.attack_layout)
        self.verticalLayout_3.addLayout(self.widget_layout)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1277, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.attack_mode = "digital" 
        self.accessory_type = "facemask"
        self.impersonation_type = "Gender"
        self.target = "Woman"
        self.accessory, self.mask = self.load_accesory(self.accessory_type, self.target.lower())
        self.selection_window = None
        self.deepface_model = attributeModel(self.impersonation_type.lower())
        self.predict_avail = True

        self.thread = VideoThread(self)
        self.thread.change_pixmap_signal.connect(self.set_video_display)
        self.thread.start()

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Project Demo"))
        self.impersonation_label.setText(_translate("MainWindow", "Gender:"))
        self.video_display.setText(_translate("MainWindow", "Video Frame"))
        self.digital_attack_btn.setText(_translate("MainWindow", "Digital Attack"))
        self.physical_attack_btn.setText(_translate("MainWindow", "Physical Attack"))
        
    def set_video_display(self, image):
        size = self.video_display.size()
        try:
            h, w, ch = image.shape
            bytesPerLine = ch * w
            convert_to_qt = QtGui.QImage(image.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
            p = convert_to_qt.scaled(size, QtCore.Qt.KeepAspectRatio)
            self.video_display.setPixmap(QtGui.QPixmap.fromImage(p))
            
        except Exception as e:
            print("in set video display")
            print(e)
        
    def set_accessory(self, accessory_type):
        self.accessory_type = accessory_type
        self.accessory, self.mask = self.load_accesory(accessory_type, self.target)
        
    def set_impersonation_type(self, impersonation_type):
        self.impersonation_type = impersonation_type
        self.deepface_model = attributeModel(impersonation_type.lower())

    def set_target(self, target):
        self.target = target

    def set_attack_mode(self, attack_mode):
        self.attack_mode = attack_mode
    
    def update_settings(self, attack_mode, impersonation_type, target=None, accessory_type=None):
        self.thread.pause_stream()
        self.attack_mode = attack_mode
        if target is not None:
            self.set_target(target)
        self.set_impersonation_type(impersonation_type)
        if accessory_type is not None:
            self.set_accessory(accessory_type)
        self.thread.resume_stream()
        
    def predict(self, image):
        self.predict_avail = False
        try:
            if self.attack_mode == "digital":
                image = apply_accessory(image, self.accessory, self.mask)
            mask_applied = np.expand_dims(image, axis=0)
            mask_applied = mask_applied.astype(np.float32)
            mask_applied = np.divide(mask_applied, 255)
            prediction = self.deepface_model.predict_verbose(mask_applied)
            self.predict_avail = True
            return prediction[f"dominant_{self.impersonation_type.lower()}"], image
                
        except Exception as e:
            print("in predict")
            print(e)
            self.predict_avail = True
            return "NULL", image
        
    def load_accesory(self, accessory_type, target):
        # Load the accessory from the accessories folder
        accessory = cv2.imread("./experiment/trained_accessories/" + accessory_type + "/" + target + ".png")
        accessory = cv2.cvtColor(accessory, cv2.COLOR_BGR2RGB)
        accessory = cv2.resize(accessory, (224, 224))
        mask = cv2.imread(f"./experiment/assets/{accessory_type}.png")
        return accessory, mask

    def set_attack_mode(self, attack_mode):
        if attack_mode == "digital":
            self.selection_window = Digital_Popup(self)
            self.selection_window.show()
        else:
            self.selection_window = Physical_Popup(self)
            self.selection_window.show()

class Physical_Popup(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.setObjectName("self")
        self.setFixedSize(624, 180)
        self.gridLayout_2 = QtWidgets.QGridLayout(self)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.impersonation_label = QtWidgets.QLabel(self)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.impersonation_label.sizePolicy().hasHeightForWidth())
        self.impersonation_label.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.impersonation_label.setFont(font)
        self.impersonation_label.setObjectName("impersonation_label")
        self.impersonation_label.setText("Impersonation Type:")
        self.gridLayout_2.addWidget(self.impersonation_label, 0, 0, 1, 1)
        self.type_box = QtWidgets.QComboBox(self)
        self.type_box.setMinimumSize(QtCore.QSize(300, 50))
        self.type_box.setMaximumSize(QtCore.QSize(300, 16777215))
        self.type_box.setObjectName("type_box")
        self.type_box.addItems(["Gender", "Race", "Emotion"])
        self.type_box.setFont(font)
        self.gridLayout_2.addWidget(self.type_box, 0, 1, 1, 1)
        self.select_btn = QtWidgets.QPushButton(self)
        self.select_btn.setMinimumSize(QtCore.QSize(300, 50))
        self.select_btn.setMaximumSize(QtCore.QSize(300, 16777215))

        self.select_btn.setFont(font)
        self.select_btn.setObjectName("select_btn")
        self.select_btn.clicked.connect(self.complete_selection)
        self.select_btn.setText("Complete Selection")
        self.gridLayout_2.addWidget(self.select_btn, 1, 0, 1, 1)
        self.cancel_btn = QtWidgets.QPushButton(self)
        self.cancel_btn.setMinimumSize(QtCore.QSize(300, 50))
        self.cancel_btn.setMaximumSize(QtCore.QSize(300, 16777215))

        self.cancel_btn.setFont(font)
        self.cancel_btn.setObjectName("cancel_btn")
        self.cancel_btn.clicked.connect(self.cancel)
        self.cancel_btn.setText("Cancel")
        self.gridLayout_2.addWidget(self.cancel_btn, 1, 1, 1, 1)

    def complete_selection(self):
        self.parent.update_settings("physical", self.type_box.currentText())
        self.close()
        
    def cancel(self):
        self.close()

class Digital_Popup(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__()
        font = QtGui.QFont()
        font.setPointSize(20)
        self.parent = parent
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.accessory_box = QtWidgets.QComboBox()
        self.accessory_box.setMinimumSize(QtCore.QSize(200, 50))
        self.accessory_box.setObjectName("accessory_box")
        self.accessory_box.addItems(["facemask", "glasses", "bandana"])
        self.accessory_box.setFont(font)
        self.gridLayout.addWidget(self.accessory_box, 0, 1, 1, 1)
        
        self.impersonation_label = QtWidgets.QLabel()
        self.impersonation_label.setFont(font)
        self.impersonation_label.setObjectName("impersonation_label")
        self.impersonation_label.setText("Impersonation Type:")
        self.gridLayout.addWidget(self.impersonation_label, 1, 0, 1, 1)
        self.type_box = QtWidgets.QComboBox()
        self.type_box.setMinimumSize(QtCore.QSize(200, 50))
        self.type_box.setObjectName("type_box")
        self.type_box.addItems(["Gender", "Race", "Emotion"])
        self.type_box.setFont(font)
        self.type_box.currentTextChanged.connect(lambda: self.set_target(self.type_box.currentText()))
        self.gridLayout.addWidget(self.type_box, 1, 1, 1, 1)
        
        self.accessory_label = QtWidgets.QLabel()
        self.accessory_label.setFont(font)
        self.accessory_label.setObjectName("accessory_label")
        self.accessory_label.setText("Accessory Type:")
        self.gridLayout.addWidget(self.accessory_label, 0, 0, 1, 1)
        self.target_label = QtWidgets.QLabel()
        self.target_label.setFont(font)
        self.target_label.setObjectName("target_label")
        self.target_label.setText("Impersonation Target:")
        self.gridLayout.addWidget(self.target_label, 2, 0, 1, 1)
        self.target_box = QtWidgets.QComboBox()
        self.target_box.setMinimumSize(QtCore.QSize(200, 50))
        self.target_box.setObjectName("target_box")
        self.target_box.setFont(font)
        self.set_target(self.type_box.currentText())
        self.gridLayout.addWidget(self.target_box, 2, 1, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)
        
        self.save_btn = QtWidgets.QPushButton()
        self.save_btn.setMinimumSize(QtCore.QSize(0, 50))
        self.save_btn.setFont(font)
        self.save_btn.setObjectName("save_btn")
        self.save_btn.setText("Complete Selection")
        self.save_btn.clicked.connect(self.complete_selection)
        self.gridLayout.addWidget(self.save_btn, 3, 0, 1, 1)
        self.cancel_btn = QtWidgets.QPushButton()
        self.cancel_btn.setMinimumSize(QtCore.QSize(0, 50))
        self.cancel_btn.setFont(font)
        self.cancel_btn.setObjectName("cancel_btn")
        self.cancel_btn.setText("Cancel")
        self.cancel_btn.clicked.connect(self.cancel)
        self.gridLayout.addWidget(self.cancel_btn, 3, 1, 1, 1)
        
        self.setLayout(self.gridLayout_2)
        self.setWindowTitle("Selection Window")

    def set_target(self, impersonate_type):
        self.target_box.clear()
        if impersonate_type == "Gender":
            self.target_box.addItems(["Man", "Woman"])
        elif impersonate_type == "Race":
            self.target_box.addItems(["White", "Black", "Asian"])
        else:
            self.target_box.addItems(["Happy", "Sad", "Angry", "Surprise", "Neutral"])
    
    def complete_selection(self):
        self.parent.update_settings("digital", self.type_box.currentText(), self.target_box.currentText(), self.accessory_box.currentText())
        self.close()
    
    def cancel(self):
        self.close()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
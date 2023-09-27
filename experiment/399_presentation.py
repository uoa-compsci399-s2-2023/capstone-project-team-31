import cv2
import numpy as np
from image_helper_functions import *
from deepface_functions import *
from deepface_models import *
import pandas as pd
from PyQt5.QtCore import pyqtSignal, QThread, Qt
from PyQt5 import QtCore, QtGui, QtWidgets

# Default thread (Sleep mode) - Streams bench camera feed 
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.camera = cv2.VideoCapture(0)
        self.WIDTH = 1920
        self.HEIGHT = 1080
    
    # Update signals for GUI
    # ~ update_info = pyqtSignal(str, bool)
    update_instructions = pyqtSignal(str)
    finished = pyqtSignal()
    
    def run(self):
        try:
            while self._run_flag:
                # Update instructions tab in main GUI window
                self.update_instructions.emit("System is in sleep mode...")
                
                # ~ self.update_info.emit("System is in sleep mode...", False)
                
                _, img = self.camera.read() # Read in as 1920x1080p
                print("Running in default thread")
                
                # Image augmentation to crop out bench area 
                # frames_benchBGR = cv2.flip(cap, 0)
                img = cv2.flip(img, 1)
                # frames_benchBGR = cv2.resize(frames_benchBGR, (int(self.WIDTH/2), int(self.HEIGHT/2)))
                # frames_benchBGR = frames_benchBGR[0:1080, 290:750]  
                
                self.change_pixmap_signal.emit(img)
                
        except Exception as e:
            # Open a txt file to record down any errors
            print(e)



    
    
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
        font = QtGui.QFont()
        font.setPointSize(20)
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

        font = QtGui.QFont()
        font.setPointSize(15)
        self.digital_attack_btn.setFont(font)
        self.digital_attack_btn.setObjectName("digital_attack_btn")
        self.digital_attack_btn.clicked.connect(lambda: self.set_attack("digital"))
        self.attack_layout.addWidget(self.digital_attack_btn)
        self.physical_attack_btn = QtWidgets.QPushButton(self.centralwidget)
        self.physical_attack_btn.setSizePolicy(sizePolicy)
        self.physical_attack_btn.setFont(font)
        self.physical_attack_btn.setObjectName("physical_attack_btn")
        self.physical_attack_btn.setMinimumSize(QtCore.QSize(0, 50))
        self.physical_attack_btn.clicked.connect(lambda: self.set_attack("physical"))
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
        self.target = "Male"
        self.accessory = self.load_accesory(self.accessory_type)
        self.selection_window = None

        self.thread = VideoThread()
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
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            convert_to_qt = QtGui.QImage(image.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
            p = convert_to_qt.scaled(size, QtCore.Qt.KeepAspectRatio)
            
            self.video_display.setPixmap(QtGui.QPixmap.fromImage(p))
        except Exception as e:
            print(e)
        
    def set_accessory(self, accessory_type):
        self.accessory_type = accessory_type
        
    def set_impersonation_type(self, impersonation_type):
        self.impersonation_type = impersonation_type
        self.impersonation_result.setText(impersonation_type)
            
    def set_target(self, target):
        self.target = target
    
    def predict(self, image):
        pass

    def load_accesory(self, accessory_type):
        pass
    
    def set_attack(self, attack_mode):
        if attack_mode == "digital":
            self.attack_mode = "digital"
            self.selection_window = Selection_Window(self)
            self.selection_window.show()
        else:
            self.attack_mode = "physical"


class Selection_Window(QtWidgets.QWidget):
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
        self.type_box.addItems(["Gender", "Ethnicity", "Emotion"])
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
        elif impersonate_type == "Ethnicity":
            self.target_box.addItems(["White", "Black", "Asian"])
        else:
            self.target_box.addItems(["Happy", "Sad", "Angry", "Surprise", "Neutral"])
    
    def complete_selection(self):
        self.parent.set_impersonation_type(self.type_box.currentText())
        self.parent.set_accessory(self.accessory_box.currentText())
        self.parent.set_target(self.target_box.currentText())
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
import cv2
import sys
import numpy as np

from PyQt5.QtCore import pyqtSignal, QThread
from PyQt5 import QtCore, QtGui, QtWidgets

from image_helper_functions import *
from deepface_functions import *
from deepface_models import *


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray, object)

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
                    _, img = self.camera.read() # Read in as 1280x720 from webcam
                    mask_applied = None           
                    img = cv2.flip(img, 1)
                    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    if self.parent.face_align:
                        try: # Try to detect the face in the image
                            face_detected = getImageContents(image)
                            mask_prep = np.multiply(face_detected[0], 255).astype(np.uint8)[0]
                            if self.parent.attack_mode == "digital":
                                mask_applied = apply_accessory(mask_prep, self.parent.accessory, self.parent.mask)
                            else:
                                mask_applied = mask_prep
                        except Exception as e:
                            print(e)
                    else:
                        mask_applied = self.parent.apply_faceframe(image.copy())
                except Exception as e:
                    print("in thread")
                    print(e)
                finally:
                    self.change_pixmap_signal.emit(image, mask_applied)
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
        MainWindow.setWindowTitle("Project Demo")

        self.aug_image = None
        self.raw_image = None
        self.attack_mode = "digital" 
        self.accessory_type = "facemask"
        self.impersonation_type = "Gender"
        self.target = "Woman"
        self.predict_avail = True
        self.face_align = True
        self.accessory, self.mask = self.load_accesory(self.accessory_type, self.target.lower())
        self.frame = cv2.imread("./experiment/assets/frame.png")
        self.selection_window = None
        self.face_frame, self.frame_mask = self.load_face_frame()
        self.frame_mask = np.where(self.frame_mask == 1)
        self.deepface_model = attributeModel(self.impersonation_type.lower())

        font = QtGui.QFont()
        font.setPointSize(20)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.widget_layout = QtWidgets.QVBoxLayout()
        self.widget_layout.setObjectName("widget_layout")
        self.result_layout = QtWidgets.QGridLayout()
        self.result_layout.setObjectName("result_layout")

        self.attack_mode_label = QtWidgets.QLabel(self.centralwidget)
        self.attack_mode_label.setMinimumSize(QtCore.QSize(0, 50))
        self.attack_mode_label.setMaximumSize(QtCore.QSize(16777215, 50))
        self.attack_mode_label.setFont(font)
        self.attack_mode_label.setObjectName("attack_mode_label")
        self.attack_mode_label.setText("Attack Mode:")
        self.attack_mode_label.setAlignment(QtCore.Qt.AlignCenter)
        self.result_layout.addWidget(self.attack_mode_label, 0, 0, 1, 1)

        self.attack_mode_disp = QtWidgets.QLabel(self.centralwidget)
        self.attack_mode_disp.setMinimumSize(QtCore.QSize(0, 50))
        self.attack_mode_disp.setMaximumSize(QtCore.QSize(16777215, 50))
        self.attack_mode_disp.setFont(font)
        self.attack_mode_disp.setObjectName("attack_mode_disp")
        self.attack_mode_disp.setText(self.attack_mode)
        self.result_layout.addWidget(self.attack_mode_disp, 0, 1, 1, 1)

        self.impersonation_type_label = QtWidgets.QLabel(self.centralwidget)
        self.impersonation_type_label.setMinimumSize(QtCore.QSize(0, 50))
        self.impersonation_type_label.setMaximumSize(QtCore.QSize(16777215, 50))
        self.impersonation_type_label.setFont(font)
        self.impersonation_type_label.setAlignment(QtCore.Qt.AlignCenter)
        self.impersonation_type_label.setObjectName("impersonation_type")
        self.impersonation_type_label.setText("Impersonation Type:")

        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.impersonation_type_label.sizePolicy().hasHeightForWidth())
        self.impersonation_type_label.setSizePolicy(sizePolicy)

        self.result_layout.addWidget(self.impersonation_type_label, 1, 0, 1, 1)

        self.impersonation_type_disp = QtWidgets.QLabel(self.centralwidget)
        self.impersonation_type_disp.setMinimumSize(QtCore.QSize(0, 50))
        self.impersonation_type_disp.setMaximumSize(QtCore.QSize(16777215, 50))
        self.impersonation_type_disp.setFont(font)
        self.impersonation_type_disp.setText(self.impersonation_type)
        self.impersonation_type_disp.setAlignment(QtCore.Qt.AlignCenter)
        self.impersonation_type_disp.setObjectName("impersonation_type_disp")
        self.result_layout.addWidget(self.impersonation_type_disp, 1, 1, 1, 1)

        self.impersonation_target_label = QtWidgets.QLabel(self.centralwidget)
        self.impersonation_target_label.setMinimumSize(QtCore.QSize(0, 50))
        self.impersonation_target_label.setMaximumSize(QtCore.QSize(16777215, 50))
        self.impersonation_target_label.setFont(font)
        self.impersonation_target_label.setObjectName("impersonation_target_label")
        self.impersonation_target_label.setText("Impersonation Target:")
        self.result_layout.addWidget(self.impersonation_target_label, 1, 2, 1, 1)

        self.accessory_type_label = QtWidgets.QLabel(self.centralwidget)
        self.accessory_type_label.setMinimumSize(QtCore.QSize(0, 50))
        self.accessory_type_label.setMaximumSize(QtCore.QSize(16777215, 50))
        self.accessory_type_label.setFont(font)
        self.accessory_type_label.setObjectName("accessory_type_label")
        self.accessory_type_label.setText("Accessory Type:")
        self.result_layout.addWidget(self.accessory_type_label, 1, 4, 1, 1)

        self.target_disp = QtWidgets.QLabel(self.centralwidget)
        self.target_disp.setMinimumSize(QtCore.QSize(0, 50))
        self.target_disp.setMaximumSize(QtCore.QSize(16777215, 50))
        self.target_disp.setFont(font)
        self.target_disp.setText(self.target)
        self.target_disp.setObjectName("target_disp")
        self.target_disp.setAlignment(QtCore.Qt.AlignCenter)
        self.result_layout.addWidget(self.target_disp, 1, 3, 1, 1)

        self.accessory_disp = QtWidgets.QLabel(self.centralwidget)
        self.accessory_disp.setMinimumSize(QtCore.QSize(0, 50))
        self.accessory_disp.setMaximumSize(QtCore.QSize(16777215, 50))
        self.accessory_disp.setFont(font)
        self.accessory_disp.setText(self.accessory_type)
        self.accessory_disp.setObjectName("accessory_disp")
        self.result_layout.addWidget(self.accessory_disp, 1, 5, 1, 1)
        self.widget_layout.addLayout(self.result_layout)
        
        self.video_display = QtWidgets.QLabel(self.centralwidget)
        self.video_display.setPixmap(QtGui.QPixmap())
        self.video_display.setAlignment(QtCore.Qt.AlignCenter)

        self.video_display.setObjectName("video_display")
        self.widget_layout.addWidget(self.video_display)

        self.predict_btn = QtWidgets.QPushButton(self.centralwidget)
        self.predict_btn.setMinimumSize(QtCore.QSize(0, 50))
        self.predict_btn.setFont(font)
        self.predict_btn.setObjectName("predict_btn")
        self.predict_btn.setText("Predict")
        self.predict_btn.clicked.connect(self.predict)
        self.widget_layout.addWidget(self.predict_btn)

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
        self.digital_attack_btn.setText("Digital Attack")
        self.digital_attack_btn.clicked.connect(lambda: self.set_attack_mode("digital"))
        self.attack_layout.addWidget(self.digital_attack_btn)
        self.physical_attack_btn = QtWidgets.QPushButton(self.centralwidget)
        self.physical_attack_btn.setSizePolicy(sizePolicy)
        self.physical_attack_btn.setFont(font)
        self.physical_attack_btn.setObjectName("physical_attack_btn")
        self.physical_attack_btn.setMinimumSize(QtCore.QSize(0, 50))
        self.physical_attack_btn.setText("Physical Attack")
        self.physical_attack_btn.clicked.connect(lambda: self.set_attack_mode("physical"))
        self.attack_layout.addWidget(self.physical_attack_btn)
        self.widget_layout.addLayout(self.attack_layout)
        self.verticalLayout_3.addLayout(self.widget_layout)

        MainWindow.setCentralWidget(self.centralwidget)

        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1273, 21))
        self.menubar.setObjectName("menubar")
        self.menusettings = QtWidgets.QMenu(self.menubar)
        self.menusettings.setObjectName("menusettings")
        self.menusettings.setTitle("settings")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionFace_Align = QtWidgets.QAction(MainWindow)
        self.actionFace_Align.setObjectName("actionFace_Align")
        self.menusettings.addAction(self.actionFace_Align)
        self.actionFace_Align.setText(f"Switch Face Align to {not self.face_align}")
        self.actionFace_Align.triggered.connect(self.set_face_align)
        self.menubar.addAction(self.menusettings.menuAction())

        self.thread = VideoThread(self)
        self.thread.change_pixmap_signal.connect(self.set_video_display)
        self.thread.start()

        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def set_face_align(self):
        self.face_align = not self.face_align
        self.actionFace_Align.setText(f"Switch Face Align to {not self.face_align}")
        
    def load_face_frame(self):
        frame = cv2.imread("./experiment/assets/frame.png")
        frame = cv2.resize(frame, (480, 480)) 
        # make a colour mask of the chosen colour
        colour = (0, 0, 255)
        frame = np.bitwise_not(frame)
        frame = cv2.threshold(frame, 0, 1, cv2.THRESH_BINARY)[1]
            
        coloured_matrix = np.array([[colour for i in range(frame.shape[1])] for j in range(frame.shape[0])])
        coloured_frame = np.multiply(coloured_matrix, frame).astype(np.uint8)
        coloured_frame = cv2.cvtColor(coloured_frame, cv2.COLOR_RGB2BGR)

        return coloured_frame, frame
    
    def set_video_display(self, org_img, aug_img):
        self.org_image = org_img
        self.aug_image = aug_img
        size = self.video_display.size()
        
        try:
            if self.aug_image is not None:
                if self.attack_mode == "digital":
                    if self.face_align:
                        image = aug_img
                    else:
                        image = org_img
                else:
                    image = aug_img
            else:
                image = org_img
            
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
        self.accessory_disp.setText(self.accessory_type if self.accessory_type is not None else "")
        if self.accessory_type is not None:
            self.accessory, self.mask = self.load_accesory(accessory_type, self.target)
        
    def set_impersonation_type(self, impersonation_type):
        self.impersonation_type = impersonation_type
        self.deepface_model = attributeModel(impersonation_type.lower())
        self.impersonation_type_disp.setText(self.impersonation_type)

    def set_target(self, target):
        self.target = target
        self.target_disp.setText(self.target if self.target is not None else "")

    def set_attack(self, attack_mode):
        self.attack_mode = attack_mode
        self.attack_mode_disp.setText(self.attack_mode)
    
    def update_settings(self, attack_mode, impersonation_type, target=None, accessory_type=None):
        try:
            self.set_attack(attack_mode)
            self.set_target(target)
            self.set_impersonation_type(impersonation_type)
            self.set_accessory(accessory_type)
            if attack_mode == "physical" and self.face_align:
                self.set_face_align()
            elif attack_mode == "digital" and not self.face_align:
                self.set_face_align()
        except Exception as e:
            print("in update settings")
            print(e)

    def apply_faceframe(self, image): # The current image from webcam is 1280x720
        img_shape = image.shape
        print(img_shape)
        start_x = int((img_shape[0] - 480) / 2)
        start_y = int((img_shape[1] - 480) / 2)

        partition = image[start_x:start_x+480, start_y:start_y+480]
        partition[self.frame_mask] = self.face_frame[self.frame_mask]
        image[start_x:start_x+480, start_y:start_y+480] = partition
        return image
        
    def predict(self):
        self.predict_avail = False
        try:
            face_detected = None
            if self.face_align:
                try:
                    face_detected = getImageContents(self.org_image)
                except Exception as e:
                    print(e)
                    face_detected = None

            img_shape = self.org_image.shape
            start_x = int((img_shape[0] - 480) / 2)
            start_y = int((img_shape[1] - 480) / 2)
            
            image = self.org_image[start_x:start_x+480, start_y:start_y+480]
            image = cv2.resize(image, (224, 224))

            if face_detected is not None:
                image = np.multiply(face_detected[0], 255).astype(np.uint8)[0]
            
            processed_org = np.expand_dims(image, axis=0)
            processed_org = processed_org.astype(np.float32)
            processed_org = np.divide(processed_org, 255)
            org_prediction = self.deepface_model.predict_verbose(processed_org)
            if self.attack_mode == "digital":
                mask_applied = apply_accessory(image.copy(), self.accessory, self.mask)
                mask_applied_post = np.expand_dims(mask_applied, axis=0)
                mask_applied_post = mask_applied_post.astype(np.float32)
                mask_applied_post = np.divide(mask_applied_post, 255)
                imp_prediction = self.deepface_model.predict_verbose(mask_applied_post)

                self.result_window = Digital_Prediction_Popup(self, image, mask_applied, org_prediction[f"dominant_{self.impersonation_type.lower()}"], imp_prediction[f"dominant_{self.impersonation_type.lower()}"])
                self.result_window.show()
            else:
                self.result_window = Physical_Prediction_Popup(self, image, org_prediction[f"dominant_{self.impersonation_type.lower()}"])
                self.result_window.show()
            
        except Exception as e:
            print("in predict")
            print(e)
        
    def load_accesory(self, accessory_type, target):
        # Load the accessory from the accessories folder
        accessory = cv2.imread("./experiment/trained_accessories/" + accessory_type.lower() + "/" + target.lower() + ".png")
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
        self.accessory_box.addItems(["facemask", "glasses"])
        self.accessory_box.currentTextChanged.connect(self.set_type_box)
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
        self.set_type_box()
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

    def set_type_box(self):
        self.type_box.clear()
        if self.accessory_box.currentText() == "glasses":
            self.type_box.addItems(["Gender", "Race"])
        else:
            self.type_box.addItems(["Gender", "Race", "Emotion"])

    def set_target(self, impersonate_type):
        self.target_box.clear()
        if impersonate_type == "Gender":
            self.target_box.addItems(["Man", "Woman"])
        elif impersonate_type == "Race":
            self.target_box.addItems(["White", "Black", "Asian", "Indian", "Middle Eastern"])
        else:
            self.target_box.addItems(["Happy", "Sad", "Angry", "Surprise", "Neutral"])
    
    def complete_selection(self):
        self.parent.update_settings("digital", self.type_box.currentText(), self.target_box.currentText(), self.accessory_box.currentText())
        self.close()
    
    def cancel(self):
        self.close()

class Digital_Prediction_Popup(QtWidgets.QWidget):
    def __init__(self, parent, original_image, impersonation_image, original_result, impersonation_result):
        super().__init__()
        self.parent = parent
        self.gridLayout_2 = QtWidgets.QGridLayout()
        # self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        # self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        # self.verticalLayout.setObjectName("verticalLayout")
        self.original_label = QtWidgets.QLabel()
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.original_label.sizePolicy().hasHeightForWidth())
        self.original_label.setSizePolicy(sizePolicy)
        self.original_label.setMaximumSize(QtCore.QSize(16777215, 50))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.original_label.setFont(font)
        self.original_label.setAlignment(QtCore.Qt.AlignCenter)
        self.original_label.setText("Original:")
        # self.original_label.setObjectName("original_label")
        self.verticalLayout.addWidget(self.original_label)
        self.original_img_display = QtWidgets.QLabel()
        # self.original_img_display.setObjectName("original_img_display")
        self.verticalLayout.addWidget(self.original_img_display)
        self.original_result_label = QtWidgets.QLabel()
        self.original_result_label.setMaximumSize(QtCore.QSize(16777215, 50))
        self.original_result_label.setFont(font)
        self.original_result_label.setAlignment(QtCore.Qt.AlignCenter)
        self.original_result_label.setText(original_result)
        # self.original_result.setObjectName("original_result")
        self.verticalLayout.addWidget(self.original_result_label)
        self.gridLayout.addLayout(self.verticalLayout, 0, 0, 1, 1)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        # self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.imperonsation_label = QtWidgets.QLabel()
        self.imperonsation_label.setMaximumSize(QtCore.QSize(16777215, 50))

        self.imperonsation_label.setFont(font)
        self.imperonsation_label.setAlignment(QtCore.Qt.AlignCenter)
        self.imperonsation_label.setText("Impersonated:")
        # self.imperonsation_label.setObjectName("imperonsation_label")
        self.verticalLayout_2.addWidget(self.imperonsation_label)
        self.impersonation_img_display = QtWidgets.QLabel()
        # self.impersonation_img_display.setObjectName("impersonation_img_display")
        self.verticalLayout_2.addWidget(self.impersonation_img_display)
        self.impersonation_result_label = QtWidgets.QLabel()
        self.impersonation_result_label.setMaximumSize(QtCore.QSize(16777215, 50))
        self.impersonation_result_label.setFont(font)
        self.impersonation_result_label.setAlignment(QtCore.Qt.AlignCenter)
        self.impersonation_result_label.setText(impersonation_result)
        # self.impersonation_result.setObjectName("impersonation_result")
        self.verticalLayout_2.addWidget(self.impersonation_result_label)
        self.gridLayout.addLayout(self.verticalLayout_2, 0, 1, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)

        self.set_image(original_image, impersonation_image)
        self.setLayout(self.gridLayout_2)
        self.setWindowTitle("Prediction Window")

    def set_image(self, original_image, impersonation_image):
        original_size = self.original_img_display.size()
        impersonation_size = self.impersonation_img_display.size()
        try:
            h, w, ch = original_image.shape
            bytesPerLine = ch * w
            convert_to_qt = QtGui.QImage(original_image.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
            original_p = convert_to_qt.scaled(original_size, QtCore.Qt.KeepAspectRatio)

            h, w, ch = impersonation_image.shape
            bytesPerLine = ch * w
            convert_to_qt = QtGui.QImage(impersonation_image.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
            impersonation_p = convert_to_qt.scaled(impersonation_size, QtCore.Qt.KeepAspectRatio)
            
            self.original_img_display.setPixmap(QtGui.QPixmap.fromImage(original_p))
            self.impersonation_img_display.setPixmap(QtGui.QPixmap.fromImage(impersonation_p))
        except Exception as e:
            print(e)

    def close(self):
        self.parent.predict_avail = True

class Physical_Prediction_Popup(QtWidgets.QWidget):
    def __init__(self, parent, image, result):
        super().__init__()
        self.parent = parent
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.title_label = QtWidgets.QLabel()
        self.title_label.setMaximumSize(QtCore.QSize(16777215, 100))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.title_label.setFont(font)
        self.title_label.setAlignment(QtCore.Qt.AlignCenter)
        self.title_label.setObjectName("title_label")
        self.title_label.setText("Physical Attack Prediction")
        self.verticalLayout_2.addWidget(self.title_label)
        self.image_disp = QtWidgets.QLabel()
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.image_disp.sizePolicy().hasHeightForWidth())
        self.image_disp.setSizePolicy(sizePolicy)
        self.image_disp.setMinimumSize(QtCore.QSize(720, 720))
        self.image_disp.setAlignment(QtCore.Qt.AlignCenter)
        self.image_disp.setObjectName("image_disp")
        self.set_image(image)
        self.verticalLayout_2.addWidget(self.image_disp)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.prediction_label = QtWidgets.QLabel()
        self.prediction_label.setMinimumSize(QtCore.QSize(0, 50))
        self.prediction_label.setMaximumSize(QtCore.QSize(16777215, 50))
        self.prediction_label.setFont(font)
        self.prediction_label.setAlignment(QtCore.Qt.AlignCenter)
        self.prediction_label.setObjectName("prediction_label")
        self.prediction_label.setText("Prediction Result:")
        self.horizontalLayout.addWidget(self.prediction_label)
        self.result_disp = QtWidgets.QLabel()
        self.result_disp.setMinimumSize(QtCore.QSize(0, 50))
        self.result_disp.setMaximumSize(QtCore.QSize(16777215, 50))
        self.result_disp.setFont(font)
        self.result_disp.setText(result)
        self.result_disp.setAlignment(QtCore.Qt.AlignCenter)
        self.result_disp.setObjectName("result_disp")
        self.horizontalLayout.addWidget(self.result_disp)
        self.verticalLayout_2.addLayout(self.horizontalLayout)

        self.setLayout(self.verticalLayout_2)

    def set_image(self, image):
        size = self.image_disp.size()
        h, w, ch = image.shape
        bytesPerLine = ch * w
        convert_to_qt = QtGui.QImage(image.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
        p = convert_to_qt.scaled(size, QtCore.Qt.KeepAspectRatio)
        self.image_disp.setPixmap(QtGui.QPixmap.fromImage(p))
        

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
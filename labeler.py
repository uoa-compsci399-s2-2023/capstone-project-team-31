from PyQt5 import QtCore, QtGui, QtWidgets
import os, cv2, json, sqlite3, base64


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1086, 787)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.display_layout = QtWidgets.QHBoxLayout()
        self.display_layout.setObjectName("display_layout")
        self.prev_image = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.prev_image.sizePolicy().hasHeightForWidth())
        self.prev_image.setSizePolicy(sizePolicy)
        self.prev_image.setMaximumSize(QtCore.QSize(200, 16777215))
        self.prev_image.clicked.connect(self.set_prev_image)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.prev_image.setFont(font)
        self.prev_image.setObjectName("prev_image")
        self.display_layout.addWidget(self.prev_image)
        self.Image = QtWidgets.QLabel(self.centralwidget)
        self.Image.setMinimumSize(QtCore.QSize(0, 500))
        self.Image.setObjectName("Image")
        self.Image.setPixmap(QtGui.QPixmap())
        self.Image.setScaledContents(True)
        self.display_layout.addWidget(self.Image)
        self.next_image = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.next_image.sizePolicy().hasHeightForWidth())
        self.next_image.setSizePolicy(sizePolicy)
        self.next_image.setMaximumSize(QtCore.QSize(200, 16777215))
        self.next_image.clicked.connect(self.set_next_image)

        self.next_image.setFont(font)
        self.next_image.setObjectName("next_image")
        self.display_layout.addWidget(self.next_image)
        self.verticalLayout.addLayout(self.display_layout)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.age_label = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.age_label.sizePolicy().hasHeightForWidth())
        self.age_label.setSizePolicy(sizePolicy)
        self.age_label.setMinimumSize(QtCore.QSize(0, 100))
        self.age_label.setMaximumSize(QtCore.QSize(16777215, 100))

        self.age_label.setFont(font)
        self.age_label.setAlignment(QtCore.Qt.AlignCenter)
        self.age_label.setObjectName("age_label")
        self.gridLayout.addWidget(self.age_label, 0, 0, 1, 1)
        self.gender_label = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.gender_label.sizePolicy().hasHeightForWidth())
        self.gender_label.setSizePolicy(sizePolicy)
        self.gender_label.setMinimumSize(QtCore.QSize(0, 100))
        self.gender_label.setMaximumSize(QtCore.QSize(16777215, 100))

        self.gender_label.setFont(font)
        self.gender_label.setAlignment(QtCore.Qt.AlignCenter)
        self.gender_label.setObjectName("gender_label")
        self.gridLayout.addWidget(self.gender_label, 0, 1, 1, 1)
        self.ethnicity_label = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ethnicity_label.sizePolicy().hasHeightForWidth())
        self.ethnicity_label.setSizePolicy(sizePolicy)
        self.ethnicity_label.setMinimumSize(QtCore.QSize(0, 100))
        self.ethnicity_label.setMaximumSize(QtCore.QSize(16777215, 100))
        self.ethnicity_label.setFont(font)
        self.ethnicity_label.setAlignment(QtCore.Qt.AlignCenter)
        self.ethnicity_label.setObjectName("ethnicity_label")
        self.gridLayout.addWidget(self.ethnicity_label, 0, 3, 1, 1)
        self.emotion_label = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.emotion_label.sizePolicy().hasHeightForWidth())
        self.emotion_label.setSizePolicy(sizePolicy)
        self.emotion_label.setMinimumSize(QtCore.QSize(0, 100))
        self.emotion_label.setMaximumSize(QtCore.QSize(16777215, 100))

        self.emotion_label.setFont(font)
        self.emotion_label.setAlignment(QtCore.Qt.AlignCenter)
        self.emotion_label.setObjectName("emotion_label")
        self.gridLayout.addWidget(self.emotion_label, 0, 2, 1, 1)
        self.age_combo = QtWidgets.QComboBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.age_combo.sizePolicy().hasHeightForWidth())
        self.age_combo.setSizePolicy(sizePolicy)
        self.age_combo.setMinimumSize(QtCore.QSize(0, 50))
        self.age_combo.setObjectName("age_combo")
        self.gridLayout.addWidget(self.age_combo, 1, 0, 1, 1)
        self.setup_age_combo()
        self.age_combo.setFont(font)
        
        self.gender_combo = QtWidgets.QComboBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.gender_combo.sizePolicy().hasHeightForWidth())
        self.gender_combo.setSizePolicy(sizePolicy)
        self.gender_combo.setMinimumSize(QtCore.QSize(0, 50))
        self.gender_combo.setObjectName("gender_combo")
        self.gridLayout.addWidget(self.gender_combo, 1, 1, 1, 1)
        self.setup_gender_combo()
        self.gender_combo.setFont(font)
        
        self.emotion_combo = QtWidgets.QComboBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.emotion_combo.sizePolicy().hasHeightForWidth())
        self.emotion_combo.setSizePolicy(sizePolicy)
        self.emotion_combo.setMinimumSize(QtCore.QSize(0, 50))
        self.emotion_combo.setObjectName("emotion_combo")
        self.gridLayout.addWidget(self.emotion_combo, 1, 2, 1, 1)
        self.setup_emotion_combo()
        self.emotion_combo.setFont(font)
        
        self.ethnicity_combo = QtWidgets.QComboBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ethnicity_combo.sizePolicy().hasHeightForWidth())
        self.ethnicity_combo.setSizePolicy(sizePolicy)
        self.ethnicity_combo.setMinimumSize(QtCore.QSize(0, 50))
        self.ethnicity_combo.setObjectName("ethnicity_combo")
        self.gridLayout.addWidget(self.ethnicity_combo, 1, 3, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        self.setup_ethnicity_combo()
        self.ethnicity_combo.setFont(font)
        
        self.save_labels = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.save_labels.sizePolicy().hasHeightForWidth())
        self.save_labels.setSizePolicy(sizePolicy)
        self.save_labels.setMinimumSize(QtCore.QSize(0, 100))
        self.save_labels.clicked.connect(self.save_labels_to_db)
        self.save_labels.setFont(font)
        self.save_labels.setObjectName("pushButton")
        self.verticalLayout.addWidget(self.save_labels)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1086, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.image_dir_select = QtWidgets.QAction(MainWindow)
        self.image_dir_select.setObjectName("image_dir_select")
        self.image_dir_select.triggered.connect(self.set_image_directory)
        self.menuFile.addAction(self.image_dir_select)
        self.menubar.addAction(self.menuFile.menuAction())

        self.image_directory = None
        self.images = None
        self.image_count = 0
        self.curr_image = None
        self.json_file = "./Faces.json"
        self.db = "./Faces.db"
        self.db_conn = sqlite3.connect(self.db)
        self.db_cursor = self.db_conn.cursor()

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
    def setup_emotion_combo(self):
        self.emotion_combo.addItems(["", "happy", "sad", "angry", "neutral", "surprised"])
    def setup_age_combo(self):
        self.age_combo.addItems([""] + [str(i) for i in range(100)])
    def setup_gender_combo(self):
        self.gender_combo.addItems(["", "Male", "Female"])
    def setup_ethnicity_combo(self):
        self.ethnicity_combo.addItems(["", "White", "Black", "Asian", "Maori", "Pacific Islander", "Other"])
        
    def set_image_directory(self):
        self.image_directory = QtWidgets.QFileDialog.getExistingDirectory(None, 'Select a folder:')
        self.images = os.listdir(self.image_directory)
        self.set_image(self.images[self.image_count])
        
    def set_next_image(self):
        if self.image_directory is not None:
            # Set the image displayed
            if self.image_count < len(self.images) - 1:
                with open(self.json_file, "r") as f:
                    data = json.load(f)
                    while self.images[self.image_count] in data.keys():
                        self.image_count += 1
                        if self.image_count >= len(self.images):
                            msg = QtWidgets.QMessageBox()
                            msg.setText("All images have been labelled")
                            msg.exec_()
                            self.image_count = 0
                            return
                if self.set_image(self.images[self.image_count]):
                    self.age_combo.setCurrentIndex(0)
                    self.gender_combo.setCurrentIndex(0)
                    self.ethnicity_combo.setCurrentIndex(0)
                    self.emotion_combo.setCurrentIndex(0)
                
    def set_prev_image(self):
        if self.image_directory is not None:
            if self.image_count > 0:
                with open(self.json_file, "r") as f:
                    data = json.load(f)
                    while self.images[self.image_count] in data.keys():
                        self.image_count -= 1
                        if self.image_count < 0:
                            msg = QtWidgets.QMessageBox()
                            msg.setText("All images have been labelled")
                            msg.exec_()
                            self.image_count = 0
                            return
                if self.set_image(self.images[self.image_count]):
                    self.age_combo.setCurrentIndex(0)
                    self.gender_combo.setCurrentIndex(0)
                    self.ethnicity_combo.setCurrentIndex(0)
                    self.emotion_combo.setCurrentIndex(0)

    def set_image(self, image_dir):
        size = self.Image.size()
        try:
            self.curr_image = cv2.imread(self.image_directory + "/" + image_dir)
            image = cv2.cvtColor(self.curr_image, cv2.COLOR_BGR2RGB)
            h, w, ch = image.shape
            bytesPerLine = ch * w
            convert_to_qt = QtGui.QImage(image.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
            p = convert_to_qt.scaled(size, QtCore.Qt.KeepAspectRatio, transformMode=QtCore.Qt.SmoothTransformation)
            self.Image.setPixmap(QtGui.QPixmap.fromImage(p))
            return True
        except:
            print("Error loading image: {}".format(image_dir))
            return False
        
    def save_labels_to_file(self):
        try:
            with open(self.json_file, "r+") as outfile:
                objs = json.load(outfile)
                objs[self.images[self.image_count]] = {
                    "ethnicity": self.ethnicity_combo.currentText(),
                    "age": self.age_combo.currentText(),
                    "gender": self.gender_combo.currentText(),
                    "emotion": self.emotion_combo.currentText()
                }
                outfile.seek(0)
                json.dump(objs, outfile, indent=4)
                
        except Exception as e:
            print(e)
            print("Error saving labels")
            
    def save_labels_to_db(self):
        try:
            img_b64 = self.convert_base64()
            if img_b64 is not None:
                self.db_cursor.execute("SELECT * FROM images WHERE img_base64=?", (img_b64,))
                if self.db_cursor.fetchone() is None: # If the image is not in the databse, add it to the database
                    self.db_cursor.execute("INSERT INTO images VALUES (?,?,?,?,?)", (img_b64, 
                                                                                    self.ethnicity_combo.currentText(),
                                                                                    self.age_combo.currentText(),
                                                                                    self.gender_combo.currentText(),
                                                                                    self.emotion_combo.currentText()))
                    self.db_conn.commit()
                    self.save_labels_to_file()
        except:
            print("Error saving labels to database")
        
    def convert_base64(self):
        try:
            if self.images[self.image_count].endswith('.jpg') or self.images[self.image_count].endswith('.jpeg'):
                _, img_b64 = cv2.imencode('.jpg', self.curr_image)
            elif self.images[self.image_count].endswith('.png'):
                _, img_b64 = cv2.imencode('.png', self.curr_image)
            else:
                print("Error: Image format not supported")
                return None
            img_b64 = base64.b64encode(img_b64)[2:-1]
            return img_b64
        except:
            print("Error converting image to base64")
            return None
        
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "399 Image Labeler"))
        self.prev_image.setText(_translate("MainWindow", "Previous Image"))
        self.Image.setText(_translate("MainWindow", ""))
        self.next_image.setText(_translate("MainWindow", "Next Image"))
        self.age_label.setText(_translate("MainWindow", "Age"))
        self.gender_label.setText(_translate("MainWindow", "Gender"))
        self.ethnicity_label.setText(_translate("MainWindow", "Ethnicity"))
        self.emotion_label.setText(_translate("MainWindow", "Emotion"))
        self.save_labels.setText(_translate("MainWindow", "Save Labels"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.image_dir_select.setText(_translate("MainWindow", "Select Image Directory"))
        
        
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setStyleSheet('Fusion')
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
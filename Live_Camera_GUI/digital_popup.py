from PyQt5 import QtCore, QtGui, QtWidgets


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
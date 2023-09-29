from PyQt5 import QtCore, QtGui, QtWidgets


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
        self.type_box.addItems(["Gender", "Ethnicity", "Emotion"])
        self.type_box.setFont(font)
        self.type_box.currentTextChanged.connect(lambda: self.set_target_box(self.type_box.currentText()))
        self.gridLayout_2.addWidget(self.type_box, 0, 1, 1, 1)
        self.select_btn = QtWidgets.QPushButton(self)
        self.select_btn.setMinimumSize(QtCore.QSize(300, 50))
        self.select_btn.setMaximumSize(QtCore.QSize(300, 16777215))

        self.target_label = QtWidgets.QLabel(self)
        self.target_label.setFont(font)
        self.target_label.setObjectName("target_label")
        self.target_label.setText("Impersonation Target:")
        self.gridLayout_2.addWidget(self.target_label, 1, 0, 1, 1)

        self.target_box = QtWidgets.QComboBox(self)
        self.target_box.setMinimumSize(QtCore.QSize(300, 50))
        self.target_box.setMaximumSize(QtCore.QSize(300, 16777215))
        self.target_box.setObjectName("target_box")
        self.target_box.setFont(font)
        self.set_target_box(self.type_box.currentText())
        self.gridLayout_2.addWidget(self.target_box, 1, 1, 1, 1)

        self.select_btn.setFont(font)
        self.select_btn.setObjectName("select_btn")
        self.select_btn.clicked.connect(self.complete_selection)
        self.select_btn.setText("Complete Selection")
        self.gridLayout_2.addWidget(self.select_btn, 2, 0, 1, 1)
        self.cancel_btn = QtWidgets.QPushButton(self)
        self.cancel_btn.setMinimumSize(QtCore.QSize(300, 50))
        self.cancel_btn.setMaximumSize(QtCore.QSize(300, 16777215))

        self.cancel_btn.setFont(font)
        self.cancel_btn.setObjectName("cancel_btn")
        self.cancel_btn.clicked.connect(self.cancel)
        self.cancel_btn.setText("Cancel")
        self.gridLayout_2.addWidget(self.cancel_btn, 2, 1, 1, 1)

    def set_target_box(self, selection):
        self.target_box.clear()
        if selection == "Gender":
            self.target_box.addItems(["Man", "Woman"])
        elif selection == "Ethnicity":
            self.target_box.addItems(["White", "Black", "Asian"])
        else:
            self.target_box.addItems(["Happy", "Sad", "Angry", "Surprise", "Neutral"])

    def complete_selection(self):
        self.parent.set_target(self.type_box.currentText())
        self.parent.set_impersonation_type(self.target_box.currentText())
        self.close()
        
    def cancel(self):
        self.close()
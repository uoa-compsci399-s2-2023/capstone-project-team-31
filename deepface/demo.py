from rawClasses import *
import matplotlib.pyplot as plt

e = attributeModel("gender")

contents = getImageContents("img.jpg") # returns a list of faces in the image
currentImage = contents[0] #run demo on first detected face

print("Raw Model output - ", e.raw_unprocessed_predict(currentImage))

labeledOutput = e.predict(currentImage)
print("Labeled Output")
print(labeledOutput)


attack = e.lastProcessedIntoAttack(labeledOutput["max_index"], eps = 0.03)

initialOutput = e.lastProcessed

sideBySide(initialOutput, labeledOutput, attack[0], attack[1])
from rawClasses import *

e = attributeModel("emotion")

contents = getImageContents("img.jpg") # returns a list of faces in the image
currentImage = contents[0] #run demo on first detected face

print("Raw Model output - ", e.raw_unprocessed_predict(currentImage))

labeledOutput = e.predict(currentImage)
print("Labeled Output")
print(labeledOutput)

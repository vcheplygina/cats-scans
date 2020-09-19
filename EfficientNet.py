from keras_efficientnets import EfficientNetB0

model = EfficientNetB0(classes=1000, include_top=True, weights='imagenet')
model.summary()
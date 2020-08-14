# -*- coding: utf-8 -*-
"""
@author: Felipe
"""
#Parte 1 - Construir el modelo de RNC

#Importar las librerias y paquetes
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Inicializar la RNC
classifier= Sequential()

#Paso 1 - Convolucion
classifier.add(Conv2D(filters= 32,kernel_size=(3,3),
                      input_shape=(64,64,3),activation="relu"))

#Paso 2 - MaxPooling- Detectar rasgos o patrones de una imagen
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Paso 3 - Flatteing o Aplanado 
classifier.add(Flatten())

#Paso 4 - Full Connection 
classifier.add(Dense(units= 128,activation="relu"))
classifier.add(Dense(units= 1,activation="sigmoid"))

#Construir RNC
classifier.compile(optimizer ="adam",loss="binary_crossentropy",metrics=["accuracy"])

#Parte 2 - Ajustar la CNN a las imagenes para entrenar
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_dataset = train_datagen.flow_from_directory(r'C:\Users\Felipe\Desktop\CursoDeepLearningPython\dataset\training_set',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')
testing_dataset = test_datagen.flow_from_directory(r'C:\Users\Felipe\Desktop\CursoDeepLearningPython\dataset\test_set',
                                                        target_size=(64, 64),
                                                        batch_size=32,
                                                        class_mode='binary')
classifier.fit_generator(training_dataset,
                            steps_per_epoch=8000,
                            epochs=25,
                            validation_data=testing_dataset,
                            validation_steps=2000)

from tensorflow.keras.applications import VGG16, Xception, ResNet152V2, InceptionResNetV2
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

def load_model(i):

    if i == 0:
        vgg = VGG16(weights="imagenet", include_top=False,
    	input_tensor=Input(shape=(224, 224, 3)))
        # freeze all VGG layers so they will *not* be updated during the
        # training process
        vgg.trainable = False
        # flatten the max-pooling output of VGG
        flatten = vgg.output
        flatten = Flatten()(flatten)
        # construct a fully-connected layer header to output the predicted
        # bounding box coordinates
        bboxHead = Dense(128, activation="relu")(flatten)
        bboxHead = Dense(64, activation="relu")(bboxHead)
        bboxHead = Dense(32, activation="relu")(bboxHead)
        bboxHead = Dense(8, activation="sigmoid")(bboxHead)
        # construct the model we will fine-tune for bounding box regression
        model = Model(inputs=vgg.input, outputs=bboxHead)
        keyword = 'vgg'

        return model, keyword

    if i == 1:
        xception = Xception(weights="imagenet", include_top=False)
        input_tensor=Input(shape=(224, 224, 3)) # does performance improve at 299, 299, 3
        xception.trainable = False
        flatten = xception.output
        flatten = Flatten()(flatten)
        # construct a fully-connected layer header to output the predicted
        # bounding box coordinates
        bboxHead = Dense(128, activation="relu")(flatten)
        bboxHead = Dense(64, activation="relu")(bboxHead)
        bboxHead = Dense(32, activation="relu")(bboxHead)
        bboxHead = Dense(4, activation="sigmoid")(bboxHead)
        # construct the model we will fine-tune for bounding box regression
        model = Model(inputs=xception.input, outputs=bboxHead)
        keyword = 'xception'

    if i == 2:
        resnet152 = ResNet152V2(weights="imagenet", include_top=False)
        input_tensor=Input(shape=(224, 224, 3))
        resnet152.trainable = False
        flatten = resnet152.output
        flatten = Flatten()(flatten)
        # construct a fully-connected layer header to output the predicted
        # bounding box coordinates
        bboxHead = Dense(128, activation="relu")(flatten)
        bboxHead = Dense(64, activation="relu")(bboxHead)
        bboxHead = Dense(32, activation="relu")(bboxHead)
        bboxHead = Dense(4, activation="sigmoid")(bboxHead)
        # construct the model we will fine-tune for bounding box regression
        model = Model(inputs=resnet152.input, outputs=bboxHead)
        keyword = 'resnet152'

    if i == 3:
        inception_resnet = InceptionResNetV2(weights="imagenet", include_top=False)
        input_tensor=Input(shape=(224, 224, 3)) # does performance improve at 299, 299, 3
        inception_resnet.trainable = False
        flatten = inception_resnet.output
        flatten = Flatten()(flatten)
        # construct a fully-connected layer header to output the predicted
        # bounding box coordinates
        bboxHead = Dense(128, activation="relu")(flatten)
        bboxHead = Dense(64, activation="relu")(bboxHead)
        bboxHead = Dense(32, activation="relu")(bboxHead)
        bboxHead = Dense(4, activation="sigmoid")(bboxHead)
        # construct the model we will fine-tune for bounding box regression
        model = Model(inputs=inception_resnet.input, outputs=bboxHead)
        keyword = 'resnet152'

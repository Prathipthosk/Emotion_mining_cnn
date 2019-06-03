from keras.layers import Activation, Convolution2D, Dropout, Conv2D
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D, Concatenate, concatenate
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras import layers
from keras.regularizers import l2

def my_FeatCNN(input_shape,classes):
    padding = 'valid'
    img_input = Input(shape=input_shape)

    #start model
    conv_1 = Conv2D(64, (5, 5), strides=(2, 2), padding=padding, activation='relu', name='conv_1')(img_input)
    maxpool_1 = MaxPooling2D((2, 2), strides=(2,2))(conv_1)
    x = BatchNormalization()(maxpool_1)
    
    #Feat-ex1
    conv_2a = Conv2D(96, (1, 1), strides=(1,1), activation='relu', padding=padding, name='conv_2a')(x)
    conv_2b = Conv2D(208, (3, 3), strides=(1,1), activation='relu', padding=padding, name='conv_2b')(conv_2a)
    maxpool_2a = MaxPooling2D((3,3), strides=(1,1), padding=padding, name='maxpool_2a')(x)
    conv_2c = Conv2D(64, (1, 1), strides=(1,1), name='conv_2c')(maxpool_2a)
    concat_1 = concatenate(inputs=[conv_2b,conv_2c], axis=3,name='concat2')
    maxpool_2b = MaxPooling2D((3,3), strides=(2,2), padding=padding, name='maxpool_2b')(concat_1)

    #Feat-ex2
    # conv_3a = Conv2D(96, (1, 1), strides=(1,1), activation='relu', padding=padding, name='conv_3a')(maxpool_2b)
    #conv_3b = Conv2D(208, (3, 3), strides=(1,1), activation='relu', padding=padding, name='conv_3b')(conv_3a)
    #maxpool_3a = MaxPooling2D((3,3), strides=(1,1), padding=padding, name='maxpool_3a')(maxpool_2b)
    #conv_3c = Conv2D(64, (1, 1), strides=(1,1), name='conv_3c')(maxpool_3a)
    #concat_3 = concatenate(inputs=[conv_3b,conv_3c],axis=3,name='concat3')
    #maxpool_3b = MaxPooling2D((3,3), strides=(2,2), padding=padding, name='maxpool_3b')(concat_3)
    
    #Final Layers
    net = Flatten()(maxpool_2b)
    net = Dense(classes, activation='softmax', name='predictions')(net)
    
    # Create model.
    model = Model(img_input, net, name='deXpression')
    return model


if __name__ == "__main__":
    num_classes = 7
   
    model = my_FeatCNN((48, 48, 1), num_classes)
    model.summary()

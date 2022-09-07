import tensorflow
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D, MaxPooling2D, Dense
from tensorflow.keras.layers import Input, Dropout, ZeroPadding2D, Reshape, Concatenate, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras import models
import tensorflow.keras.backend as K
from kapre.time_frequency import Melspectrogram, Spectrogram
from kapre.utils import Normalization2D

from models.attention_layer import channel_attention

def resnet_layer(inputs,num_filters=16,kernel_size=3,strides=1,learn_bn = True,wd=1e-4,use_relu=True):
    x = inputs
    x = Conv2D(num_filters,kernel_size=kernel_size,strides=strides,padding='valid',kernel_initializer='he_normal',
                  kernel_regularizer=l2(wd),use_bias=False)(x)
    x = BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    if use_relu:
        x = Activation('relu')(x)
    return x

def conv_layer1(inputs, num_channels=6, num_filters=14, learn_bn=True, wd=1e-4, use_relu=True):
    kernel_size1 = [5, 5]
    kernel_size2 = [3, 3]
    strides1 = [2, 2]
    strides2 = [1, 1]
    x = inputs
    x = BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    x = ZeroPadding2D(padding=(2, 2), data_format='channels_last')(x)
    x = Conv2D(num_filters*num_channels, kernel_size=kernel_size1, strides=strides1,
               padding='valid', kernel_initializer='he_normal',
               kernel_regularizer=l2(wd), use_bias=False)(x)
    x = BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    if use_relu:
        x = Activation('relu')(x)

    x = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(x)
    x = Conv2D(num_filters * num_channels, kernel_size=kernel_size2, strides=strides2,
               padding='valid', kernel_initializer='he_normal',
               kernel_regularizer=l2(wd), use_bias=False)(x)
    x = BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    if use_relu:
        x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2),padding='valid')(x)
    return x


def conv_layer2(inputs, num_channels=6, num_filters=28, learn_bn=True, wd=1e-4, use_relu=True):
    kernel_size = [3, 3]
    strides = [1, 1]
    x = inputs
    x = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(x)
    x = Conv2D(num_filters*num_channels, kernel_size=kernel_size, strides=strides,
               padding='valid', kernel_initializer='he_normal',
               kernel_regularizer=l2(wd), use_bias=False)(x)
    x = BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    if use_relu:
        x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(x)
    x = Conv2D(num_filters * num_channels, kernel_size=kernel_size, strides=strides,
               padding='valid', kernel_initializer='he_normal',
               kernel_regularizer=l2(wd), use_bias=False)(x)
    x = BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    if use_relu:
        x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='valid')(x)
    return x

def conv_layer3(inputs, num_channels=6, num_filters=56, learn_bn=True, wd=1e-4, use_relu=True):
    kernel_size = [3, 3]
    strides = [1, 1]
    x = inputs
    # 1
    x = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(x)
    x = Conv2D(num_filters*num_channels, kernel_size=kernel_size, strides=strides,
               padding='valid', kernel_initializer='he_normal',
               kernel_regularizer=l2(wd), use_bias=False)(x)
    x = BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    if use_relu:
        x = Activation('relu')(x)
    x = Dropout(0.3)(x)    
    #x = Dropout(0.3)(x,training=True)
    #2
    x = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(x)
    x = Conv2D(num_filters*num_channels, kernel_size=kernel_size, strides=strides,
               padding='valid', kernel_initializer='he_normal',
               kernel_regularizer=l2(wd), use_bias=False)(x)
    x = BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    if use_relu:
        x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='valid')(x)
    return x


def model_fcnn(input_shape=None, num_filters=[24, 48, 96], wd=1e-3):
    inputs = Input(shape=(48000,))
    x = Reshape((1,-1))(inputs)

    # Audio feature extraction layer
    m = Melspectrogram(n_dft=1024, n_hop=128, input_shape=(1,),
                       padding='same', sr=16000, n_mels=128,
                       fmin=40.0, fmax=8000, power_melgram=1.0,
                       return_decibel_melgram=True, trainable_fb=False,
                       trainable_kernel=False,
                       name='mel_stft')
    m.trainable = False

    x = m(x)
    x = Normalization2D(int_axis=0, name='mel_stft_norm')(x)

    ConvPath1 = conv_layer1(inputs=x,
                            num_channels=input_shape[-1],
                            num_filters=num_filters[0],
                            learn_bn=True,
                            wd=wd,
                            use_relu=True)
    ConvPath2 = conv_layer2(inputs=ConvPath1,
                            num_channels=input_shape[-1],
                            num_filters=num_filters[1],
                            learn_bn=True,
                            wd=wd,
                            use_relu=True)
    ConvPath3 = conv_layer3(inputs=ConvPath2,
                            num_channels=input_shape[-1],
                            num_filters=num_filters[2],
                            learn_bn=True,
                            wd=wd,
                            use_relu=True)
    # OutputPath = resnet_layer(inputs=ConvPath3,
    #                           num_filters=1,
    #                           strides=1,
    #                           kernel_size=1,
    #                           learn_bn=False,
    #                           wd=wd,
    #                           use_relu=True)
    
    OutputPath = ConvPath3
    OutputPath = BatchNormalization(center=False, scale=False)(OutputPath)
    OutputPath = channel_attention(OutputPath, ratio=2)
    OutputPath = GlobalAveragePooling2D()(OutputPath)
    OutputPath = Dense(1)(OutputPath)

    OutputPath = Activation('relu')(OutputPath)

    model = Model(inputs=inputs, outputs=OutputPath)
    return model


def model_fcnn_pre(num_classes, input_shape, num_filters, wd):
    fcnn = model_fcnn(12, input_shape=[128, None, 1], num_filters=[24, 48, 96], wd=0)
    print('loading GSC pre-trained weight')
    weights_path = 'weight_limit100/gsc97-0.9231.hdf5'
    fcnn.load_weights(weights_path)
    #model.add(Dense(num_classes, input_shape=(12,)))
    model = models.Sequential()
    model.add(fcnn)
    model.add(Dense(units=num_classes, activation='softmax'))
    #fcnn.trainable = False
    return model

def model_fcnn_music(num_classes, input_shape, num_filters, wd):

    inputs = Input(shape=(50208,))
    x = Reshape((1,-1))(inputs)

    # Audio feature extraction layer
    m = Melspectrogram(n_dft=1024, n_hop=128, input_shape=(1,),
                       padding='same', sr=16000, n_mels=128,
                       fmin=40.0, fmax=8000, power_melgram=1.0,
                       return_decibel_melgram=True, trainable_fb=False,
                       trainable_kernel=False,
                       name='mel_stft')
    m.trainable = False

    x = m(x)
    x = Normalization2D(int_axis=0, name='mel_stft_norm')(x)

    #fcnn = model_fcnn_mic(12, input_shape=[128, None, 1], num_filters=[24, 48, 96], wd=0)

    ConvPath1 = conv_layer1(inputs=x,
                            num_channels=input_shape[-1],
                            num_filters=num_filters[0],
                            learn_bn=True,
                            wd=wd,
                            use_relu=True)
    ConvPath2 = conv_layer2(inputs=ConvPath1,
                            num_channels=input_shape[-1],
                            num_filters=num_filters[1],
                            learn_bn=True,
                            wd=wd,
                            use_relu=True)
    ConvPath3 = conv_layer3(inputs=ConvPath2,
                            num_channels=input_shape[-1],
                            num_filters=num_filters[2],
                            learn_bn=True,
                            wd=wd,
                            use_relu=True)
    OutputPath = resnet_layer(inputs=ConvPath3,
                              num_filters=num_classes,
                              strides=1,
                              kernel_size=1,
                              learn_bn=False,
                              wd=wd,
                              use_relu=True)

    OutputPath = BatchNormalization(center=False, scale=False)(OutputPath)
    OutputPath = channel_attention(OutputPath, ratio=2)
    OutputPath = GlobalAveragePooling2D()(OutputPath)
    OutputPath = Activation('softmax',name="music_output")(OutputPath)

    model = Model(inputs=inputs, outputs=OutputPath)
    return model

def model_fcnn_music_emb(num_classes, input_shape, num_filters, wd):

    inputs = Input(shape=(48000,))
    inputs_2 = Input(shape=(8,23,12))

    x = Reshape((1,-1))(inputs)

    # Audio feature extraction layer
    m = Melspectrogram(n_dft=1024, n_hop=128, input_shape=(1,),
                       padding='same', sr=16000, n_mels=128,
                       fmin=40.0, fmax=8000, power_melgram=1.0,
                       return_decibel_melgram=True, trainable_fb=False,
                       trainable_kernel=False,
                       name='mel_stft')
    m.trainable = False

    x = m(x)
    x = Normalization2D(int_axis=0, name='mel_stft_norm')(x)

    #fcnn = model_fcnn_mic(12, input_shape=[128, None, 1], num_filters=[24, 48, 96], wd=0)

    ConvPath1 = conv_layer1(inputs=x,
                            num_channels=input_shape[-1],
                            num_filters=num_filters[0],
                            learn_bn=True,
                            wd=wd,
                            use_relu=True)
    ConvPath2 = conv_layer2(inputs=ConvPath1,
                            num_channels=input_shape[-1],
                            num_filters=num_filters[1],
                            learn_bn=True,
                            wd=wd,
                            use_relu=True)
    ConvPath3 = conv_layer3(inputs=ConvPath2,
                            num_channels=input_shape[-1],
                            num_filters=num_filters[2],
                            learn_bn=True,
                            wd=wd,
                            use_relu=True)
    OutputPath = resnet_layer(inputs=ConvPath3,
                              num_filters=num_classes,
                              strides=1,
                              kernel_size=1,
                              learn_bn=False,
                              wd=wd,
                              use_relu=True)

    OutputPath = BatchNormalization(center=False, scale=False)(OutputPath)
    OutputPath = OutputPath + inputs_2[:,:,:,:7]
    OutputPath = channel_attention(OutputPath, ratio=2)
    #OutputPath = Concatenate(axis=-1)([OutputPath, inputs_2])
    #OutputPath = OutputPath + inputs_2[:,:,:,:7]
    OutputPath = GlobalAveragePooling2D()(OutputPath)
    OutputPath = Activation('softmax',name="music_output")(OutputPath)

    model = Model(inputs=[inputs,inputs_2], outputs=OutputPath)
    return model

def model_fcnn_mic(num_classes, input_shape=None, num_filters=[24, 48, 96], wd=1e-3):
    inputs = Input(shape=(48000,))
    x = Reshape((1,-1))(inputs)

    # Audio feature extraction layer
    m = Melspectrogram(n_dft=1024, n_hop=128, input_shape=(1,),
                       padding='same', sr=16000, n_mels=128,
                       fmin=40.0, fmax=8000, power_melgram=1.0,
                       return_decibel_melgram=True, trainable_fb=False,
                       trainable_kernel=False,
                       name='mel_stft')
    m.trainable = False

    x = m(x)
    x = Normalization2D(int_axis=0, name='mel_stft_norm')(x)

    ConvPath1 = conv_layer1(inputs=x,
                            num_channels=input_shape[-1],
                            num_filters=num_filters[0],
                            learn_bn=True,
                            wd=wd,
                            use_relu=True)
    ConvPath2 = conv_layer2(inputs=ConvPath1,
                            num_channels=input_shape[-1],
                            num_filters=num_filters[1],
                            learn_bn=True,
                            wd=wd,
                            use_relu=True)
    ConvPath3 = conv_layer3(inputs=ConvPath2,
                            num_channels=input_shape[-1],
                            num_filters=num_filters[2],
                            learn_bn=True,
                            wd=wd,
                            use_relu=True)
    OutputPath_1 = resnet_layer(inputs=ConvPath3,
                              num_filters=num_classes,
                              strides=1,
                              kernel_size=1,
                              learn_bn=False,
                              wd=wd,
                              use_relu=True)

    OutputPath = BatchNormalization(center=False, scale=False)(OutputPath_1)
    OutputPath = channel_attention(OutputPath, ratio=2)
    OutputPath = GlobalAveragePooling2D()(OutputPath)
    OutputPath = Activation('softmax',name="mic_output")(OutputPath)

    model = Model(inputs=inputs, outputs=OutputPath)
    return model

def model_mic_music(model_mic, model_music):
    inputs = Input(shape=(48000,))

    model_mic_output = model_mic(inputs)
    #for idx in range(len(model_mic.layers)):
    #    print(model_mic.get_layer(index = idx).name)
    #model_mic_emb = model_mic.get_layer('multiply').output
    model_mic_emb_md = Model(inputs=model_mic.inputs, outputs=model_mic.get_layer(name="multiply").output,)
    model_mic_emb = model_mic_emb_md(inputs)
    model_mic_emb = Reshape((2208,))(model_mic_emb)
    #print(model_mic_emb.shape)
    #print(model_mic_output.shape)
    concatenated = Concatenate(axis=1)([model_mic_emb, inputs])
    #print(concatenated.shape)
    model_music_output = model_music(concatenated)
    #model_music_output = model_music([inputs, model_mic_emb])


    model = Model(inputs=inputs, outputs=[model_music_output, model_mic_output])
    #tensorflow.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_dtype=True)
    #model.compile(loss=losses.MSE)
    #model.summary()
    return model
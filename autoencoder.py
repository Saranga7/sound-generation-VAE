import os
import pickle
from tensorflow.keras import Model
from tensorflow.keras.layers import Input,Dense,Lambda, Reshape,Flatten,Conv2D,ReLU, BatchNormalization, Conv2DTranspose, Activation
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

import tensorflow as tf
tf.compat.v1.disable_eager_execution

import numpy as np


class VAE:
#mirrored CNN architecture used
    def __init__(self,input_shape,conv_filters,conv_kernels,conv_strides,latent_space_dim):
        self.input_shape=input_shape
        self.conv_filters=conv_filters
        self.conv_kernels=conv_kernels
        self.conv_strides=conv_strides
        self.latent_space_dim=latent_space_dim
        self.recons_loss_weight=1000000


        self.encoder=None
        self.decoder=None
        self.model=None
        self._model_input=None
        self._shape_before_bottleneck=None

        self._num_conv_layers=len(conv_filters)

        self._build()

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def compile(self,lr=0.0003):
        optimizer=Adam(learning_rate=lr)
       # mse_loss=MeanSquaredError()
        self.model.compile(optimizer=optimizer,loss=self._combined_loss,metrics=[self._recons_loss,self._kl_loss])

    def train(self,X_train,EPOCHS,batch_size=64):
        self.model.fit(X_train,X_train,
        batch_size=batch_size,
        epochs=EPOCHS,
        shuffle=True)

    def reconstruct(self, images):
        latent_rep=self.encoder.predict(images)
        reconstructed_img=self.decoder.predict(latent_rep)
        return reconstructed_img,latent_rep


    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()

    def save(self,save_folder="."):
        self._create_folder_if_not_exist(save_folder)
        self._save_parameters(save_folder)
        self._save_weights(save_folder)


    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    @classmethod
    def load(cls,save_folder="."):
        parameters_path=os.path.join(save_folder,"parameters.pkl")
        weights_path=os.path.join(save_folder,"weights.h5")

        with open(parameters_path,"rb") as f:
            parameters=pickle.load(f)
        ae=VAE(*parameters)
        weights_path=os.path.join(save_folder,"weights.h5")
        ae.load_weights(weights_path)

        return ae

    #-------------------------loss--------------------------------------


    def _combined_loss(self,y_target,y_predicted):
        recons_loss=self._recons_loss(y_target,y_predicted)
        kl_loss=self._kl_loss(y_target,y_predicted)
        combined=self.recons_loss_weight*recons_loss + kl_loss

        return combined

        
    def _recons_loss(self,y_target,y_predicted):
        error=y_target-y_predicted
        recons_loss=K.mean(K.square(error),axis=[1,2,3])
        return recons_loss

    def _kl_loss(self, y_target, y_predicted):
        kl_loss=-0.5*K.sum(1+self.log_variance-K.square(self.mu)-K.exp(self.log_variance),axis=1)
        return kl_loss

    #----------------------------------------------------------------------    


    def _create_folder_if_not_exist(self,folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

    def _save_parameters(self,save_folder):
        parameters=[self.input_shape,
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides,
            self.latent_space_dim
        ]
        save_path=os.path.join(save_folder,"parameters.pkl")
        with open(save_path,"wb") as f:
            pickle.dump(parameters,f)

    def _save_weights(self,save_folder):
        save_path=os.path.join(save_folder,"weights.h5")
        self.model.save_weights(save_path)


#------------------------------ENCODER---------------------------------------------------------

    def _build_encoder(self):
        encoder_input=self._add_encoder_input()
        conv_layers=self._add_conv_layers(encoder_input)
        bottleneck=self._add_bottleneck(conv_layers)

        self._model_input=encoder_input

        self.encoder=Model(encoder_input,bottleneck,name="encoder")

    def _add_encoder_input(self):
        return Input(shape=self.input_shape,name="encoder_input")

    def _add_conv_layers(self,encoder_input):
        #Creates all conv blocks in the encoder
        x=encoder_input

        for i in range(self._num_conv_layers):
            x=self._add_conv_layer(i,x)
        return x
    
    def _add_conv_layer(self,layer_index,x):
        #Adds convolutional block to a graph of layers
        layer_num=layer_index+1
        conv_layer=Conv2D(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"encoder_conv_layer_{layer_num}"
        )

        x=conv_layer(x)
        x=ReLU(name=f"encoder_relu_{layer_num}")(x)
        x=BatchNormalization(name=f"encoder_batchnorm_{layer_num}")(x)

        return x
        
    def _add_bottleneck(self,x):
        #flatten data and add bottleneck with Gaussian sampling
        
        #retain shape before flattening
        self._shape_before_bottleneck=K.int_shape(x)[1:] #4 dimensional, ignoring batch size
        
        x=Flatten()(x)
        self.mu=Dense(self.latent_space_dim,name="mean")(x)
        self.log_variance=Dense(self.latent_space_dim, name="log_variance")(x)

        def sample_point_norm_distri(args):
            mu, log_variance=args
            epsilon=K.random_normal(shape=K.shape(self.mu),mean=0.,stddev=1.)
            sampled_point=mu+K.exp(log_variance/2)+epsilon
            return sampled_point

        x=Lambda(sample_point_norm_distri,name="encoder_output")([self.mu,self.log_variance])

        return x

#----------------------------------DECODER-------------------------------------------------------------------
    def _build_decoder(self):
        decoder_input=self._add_decoder_input()
        dense_layer=self._add_dense_layer(decoder_input)
        reshape_layer=self._add_reshape_layer(dense_layer)
        conv_trans_layers=self._add_conv_trans_layers(reshape_layer)
        decoder_output=self._add_decoder_output(conv_trans_layers)

        self.decoder=Model(decoder_input,decoder_output,name="decoder")


    def _add_decoder_input(self):
        return Input(shape=self.latent_space_dim,name="decoder_input")

    def _add_dense_layer(self, decoder_input):
        num_neurons=np.prod(self._shape_before_bottleneck)
        dense_layer=Dense(num_neurons, name="decoder_dense")(decoder_input)
        return dense_layer

    def _add_reshape_layer(self,dense_layer):
        return Reshape(self._shape_before_bottleneck)(dense_layer)
       
    def _add_conv_trans_layers(self, x):
        #loop through all the conv layers in reverse and stop at the first layer
        for i in reversed(range(1,self._num_conv_layers)):
            x=self._add_conv_trans_layer(x,i)

        return x

    def _add_conv_trans_layer(self, x, layer_index):
        layer_num=self._num_conv_layers-layer_index
        conv_trans_layer=Conv2DTranspose(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"decoder_conv_trans_layer_{layer_num}"
        )
        x=conv_trans_layer(x)
        x=ReLU(name=f"decoder_relu_{layer_num}")(x)
        x=BatchNormalization(name=f"decoder_batchnorm_{layer_num}")(x)
        return x

    def _add_decoder_output(self,x):
        conv_transpose_layer=Conv2DTranspose(
            filters=1, #grayscale output required, i.e 1 colour channel
            kernel_size=self.conv_kernels[0],
            strides=self.conv_strides[0],
            padding="same",
            name=f"decoder_conv_trans_layer_{self._num_conv_layers}"
        )
        x=conv_transpose_layer(x)
        output_layer=Activation("sigmoid", name="sigmoid_layer")(x)

        return output_layer

#--------------------------------------Var-AUTOENCODER----------------------------------------
    def _build_autoencoder(self):
        model_input=self._model_input
        model_output=self.decoder(self.encoder(model_input))

        self.model=Model(model_input,model_output, name="Var-Autoencoder")


if __name__=="__main__":

    ae=VAE(
        input_shape=(28,28,1),
        conv_filters=(32,64,128,64),
        conv_kernels=(3,3,3,3),
        conv_strides=(1,2,2,1),
        latent_space_dim=2)

    ae.summary()




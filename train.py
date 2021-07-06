from autoencoder import VAE
import os
import numpy as np
#from tensorflow.keras.datasets import mnist

SPECTROGRAMS_PATH=r'C:\Users\Saranga\Desktop\Music Generation\Autoencoders Music Generation\free spoken digit dataset\dataset\spectograms'
LR=0.0003
BS=8
EPOCHS=10

def load_mnist():
    (X_train,y_train),(X_test,y_test)=mnist.load_data()
    X_train=X_train.astype("float32")/255
    X_train=X_train.reshape(X_train.shape+(1,))
    X_test=X_test.astype("float32")/255
    X_test=X_test.reshape(X_test.shape+(1,))

    return X_train, y_train, X_test, y_test

def load_dataset(data_path):
    X_train = []
    for root,_,filenames in os.walk(data_path):
        for file in filenames:
            file_path=os.path.join(data_path,file)
            spectrogram=np.load(file_path) #shape: (n_bins,n_fames)
            
            X_train.append(spectrogram)
        

    X_train=np.array(X_train)
    X_train=X_train[...,np.newaxis]
    return X_train
 

def train(X_train,lr,BS,EPOCHS):

    vae=VAE(
        input_shape=(256,64,1),
        conv_filters=(512,256,128,64,32),
        conv_kernels=(3,3,3,3,3),
        conv_strides=(2,2,2,2,(2,1)),
        latent_space_dim=128)
    
    vae.summary()
    vae.compile(lr)
    vae.train(X_train,batch_size=BS,EPOCHS=EPOCHS)

    return vae
    

if __name__=="__main__":

    
    X_train=load_dataset(SPECTROGRAMS_PATH)
    vae=train(X_train,LR,BS=BS,EPOCHS=EPOCHS)
    #autoencoder.save("model")
    #autoencoder2=VAE.load("model")
    #autoencoder2.summary()

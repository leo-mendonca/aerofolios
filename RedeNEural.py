import keras
import tensorflow as tf
import keras.activations as kact
import keras.optimizers as kopt
from Definicoes import *
dtype_geral=tf.float64
keras.mixed_precision.set_global_policy("float64")


class RedeAerofolio(keras.Model):
    def __init__(self, **kwargs):
        #Entrada: [camber, posicao de max. camber, espessura, angulo de ataque, velocidade do fluido]
        entrada=keras.Input(shape=(5,), name="Entrada")
        x=keras.layers.Dense(units=100, activation=kact.tanh, use_bias=True)(entrada)
        x=keras.layers.Dense(units=100, activation=kact.tanh, use_bias=True)(x)
        # x=keras.layers.Dense(units=10, activation=kact.tanh, use_bias=True)(x)
        saida=keras.layers.Dense(units=3, activation=None, use_bias=True, name="Saida")(x)
        super(RedeAerofolio,self).__init__(entrada, saida)

def scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return lr * 0.8

if __name__=="__main__":
    import numpy as np
    import pandas as pd
    dados_dframe=pd.read_csv("Saida/Aerofolio Fino NACA4/banco_resultados.csv", index_col=0)
    dados=np.array(dados_dframe)
    x=tf.cast(dados[:,:5], dtype=dtype_geral)
    y=tf.cast(dados[:,5:], dtype=dtype_geral)
    modelo=RedeAerofolio()
    otimizador=kopt.Adam(learning_rate=0.001)
    modelo.compile(optimizer=otimizador, loss=keras.losses.MeanSquaredError())
    modelo.summary()

    callback_taxa=keras.callbacks.LearningRateScheduler(scheduler)

    modelo.fit(x,y,batch_size=256, epochs=10, validation_split=0.25, callbacks=[callback_taxa])

    ymed = y[:256]
    ypred = modelo(x[:256])
    plt.scatter(ymed[:,0],ypred[:,0])
    plt.scatter(ymed[:, 2], ypred[:, 2])
    plt.show(block=False)


    modelo.summary()

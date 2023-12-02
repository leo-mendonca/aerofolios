# import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import keras.activations as kact
import keras.optimizers as kopt
from Definicoes import *
dtype_geral=tf.float64
keras.mixed_precision.set_global_policy("float64")

class CalculaVolume(keras.layers.Layer):
    '''Calcula o volume de um aerofolio fino NACA4.
    Recebe [camber, pos. camber, espessura, angulo ataque, velocidade fluido] e retorna [volume]'''
    def __init__(self, vetor_multiplicativo=[0.,0.,0.6851,0.,0.], **kwargs):
        super(CalculaVolume,self).__init__(**kwargs)
        self.vetor_linear=tf.constant(vetor_multiplicativo, dtype=dtype_geral)

    def get_config(self):
        base_config=super().get_config()
        config= {"vetor_linear":self.vetor_linear}
        return {**base_config, **config}

    def from_config(cls, config):
        vetor_linear=config.pop("vetor_linear")
        return cls(vetor_linear, **config)
    def call(self,inputs):
        # espessura=inputs[:,2]
        return inputs[:,2:3]*0.6851

@keras.saving.register_keras_serializable()
class RedeAerofolio(keras.Model):
    def __init__(self, **kwargs):
        #Entrada: [camber, posicao de max. camber, espessura, angulo de ataque, velocidade do fluido]
        entrada=keras.Input(shape=(5,), name="Entrada")
        vol=CalculaVolume(name="CalculoVolume")(entrada)
        x=keras.layers.Dense(units=100, activation=kact.tanh, use_bias=True)(entrada)
        x=keras.layers.Dense(units=100, activation=kact.tanh, use_bias=True)(x)
        # x=keras.layers.Dense(units=10, activation=kact.tanh, use_bias=True)(x)
        saida=keras.layers.Dense(units=3, activation=None, use_bias=True, name="SaidaMecanica")(x)
        saida_completa=keras.layers.Concatenate(axis=-1, name="Concatenacao")([saida, vol])
        super(RedeAerofolio,self).__init__(entrada, saida_completa, **kwargs)
    def get_config(self):
        base_config=super().get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return lr * 0.7

if __name__=="__main__":
    import numpy as np
    import pandas as pd
    import os
    nome_modelo="AerofolioFino NACA4 V1"
    path_saida = os.path.join("Saida", nome_modelo)
    os.makedirs(path_saida, exist_ok=True)
    dados_dframe=pd.read_csv("Saida/Aerofolio Fino NACA4/banco_resultados.csv", index_col=0)
    dados=np.array(dados_dframe)
    np.random.shuffle(dados)
    x=tf.cast(dados[:,:5], dtype=dtype_geral)
    y=tf.cast(dados[:,5:], dtype=dtype_geral)

    modelo=RedeAerofolio()
    otimizador=kopt.Adam(learning_rate=0.001)
    metricas=[keras.metrics.CosineSimilarity(name="Cosseno"), keras.metrics.MeanAbsoluteError(name="Vies"), keras.metrics.LogCoshError(name="LogCosh")]
    modelo.compile(optimizer=otimizador, loss=keras.losses.MeanSquaredError(name="MSE"), metrics=metricas)
    modelo.summary()
    keras.utils.plot_model(modelo, to_file=os.path.join(path_saida,"RedeAerofolio.png"), show_shapes=True, show_layer_names=True, show_layer_activations=True, show_trainable=True, dpi=300)

    callback_taxa=keras.callbacks.LearningRateScheduler(scheduler)
    callback_log=keras.callbacks.CSVLogger(os.path.join(path_saida,"Log.csv"))
    callback_parada=keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)


    modelo.fit(x,y,batch_size=256, epochs=15, validation_split=0.25, callbacks=[callback_taxa, callback_log], shuffle=True)
    avaliacao=modelo.evaluate(x,y)
    modelo.save(os.path.join(path_saida, "Modelo.keras"))
    # modelo2=keras.models.load_model(os.path.join(path_saida, "Modelo.keras"))
    # avaliacao2=modelo2.evaluate(x,y)



    ymed = y[-1024:]
    ypred = modelo(x[-1024:])
    # ypred2 = modelo2(x[-256:])
    # plt.scatter(ypred[:,0],ypred2[:,0], label="c_L")
    # plt.scatter(ypred[:,1],ypred2[:,1], label="c_D")
    # plt.scatter(ypred[:, 2], ypred2[:, 2], label="c_M")
    # ypred=modelo(x[-1024])
    plt.figure(title="c_L")
    plt.scatter(ymed[:,0],ypred[:,0], label="c_L")
    plt.scatter(ymed[:,1],ypred[:,1], label="c_D")
    plt.scatter(ymed[:, 2], ypred[:, 2], label="c_M")
    plt.legend()
    plt.savefig(os.path.join(path_saida,"Desempenho da rede - Aerofolio fino.png"), dpi=300)

    plt.show(block=False)
    plt.show(block=True)


    modelo.summary()

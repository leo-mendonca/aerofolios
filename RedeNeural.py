# import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import keras.activations as kact
import keras.optimizers as kopt
from Definicoes import *
import Salvamento
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
    def __init__(self, n_camadas, n_neuronios, lamda, **kwargs):
        #Entrada: [camber, posicao de max. camber, espessura, angulo de ataque, velocidade do fluido]

        entrada=keras.Input(shape=(5,), name="Entrada")
        # normalizacao=keras.layers.Normalization(name="Normalizacao")(entrada)
        vol=CalculaVolume(name="CalculoVolume")(entrada)
        x = keras.layers.Dense(units=n_neuronios, activation=kact.tanh, use_bias=True, kernel_regularizer=keras.regularizers.L2(lamda))(entrada)
        for i in range(n_camadas-1):
            x=keras.layers.Dense(units=n_neuronios, activation=kact.tanh, use_bias=True)(x)
        saida=keras.layers.Dense(units=3, activation=None, use_bias=True, name="SaidaMecanica")(x)
        saida_completa=keras.layers.Concatenate(axis=-1, name="Concatenacao")([saida, vol])
        super(RedeAerofolio,self).__init__(entrada, saida_completa, **kwargs)
    def get_config(self):
        base_config=super().get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def scheduler(epoch, lr, k=0.95):
    if epoch < 10:
        return lr
    else:
        return lr * k

def carrega_dados(path):
    '''Le um arquivo .csv contendo os resultados do MEF e retorna os conjuntos de treino, teste e validacao como tensores tf'''
    dados_dframe=pd.read_csv(path, sep=";", skipinitialspace=True)
    dados_dframe = Salvamento.filtragem_outliers(dados_dframe)
    dados = np.array(dados_dframe)
    x = tf.cast(dados[:, :5], dtype=dtype_geral)
    y = tf.cast(dados[:, 5:], dtype=dtype_geral)
    n=len(x)
    n_treino=n//2
    n_val=n//4
    x_treino, y_treino=x[:n_treino],y[:n_treino]
    x_val, y_val=x[n_treino:n_treino+n_val], y[n_treino:n_treino+n_val]
    x_teste, y_teste=x[n_treino+n_val:], y[n_treino+n_val:]
    return x_treino, y_treino, x_val, y_val, x_teste, y_teste

def plota_log(path):
    log=pd.read_csv(path, sep=",", skipinitialspace=True)
    plt.figure()
    plt.plot(log["epoch"].array, log["loss"].array, label="Treino")
    plt.plot(log["epoch"].array, log["val_loss"].array, label="Validacao")
    plt.legend()
    plt.xlabel("Epoca")
    plt.ylabel("Erro")
    path_diretorio=os.path.dirname(path)
    plt.savefig(os.path.join(path_diretorio,"Evolucao treinamento.png"), dpi=300)
    plt.figure()
    plt.plot(log["epoch"].array, log["lr"].array, label="Treino")
    return log

def treinar_rede(eta, decaimento, lamda, n_camadas, neuronios):
    '''Treina uma rede neural com os parametros fornecidos e salva os resultados em uma pasta'''
    nome_caso=f"Rede eta={eta} k={decaimento} lambda={lamda} camadas={n_camadas}x{neuronios}"
    nome_modelo="Rede"
    path_saida=os.path.join("Saida","Redes Neurais", nome_caso)
    os.makedirs(path_saida, exist_ok=True)
    x, y, x_val, y_val, x_teste, y_teste = carrega_dados(os.path.join("Entrada", "Dados", "dados_mef_v2.csv"))
    modelo = RedeAerofolio(n_camadas, neuronios, lamda, name=nome_modelo)
    otimizador = kopt.Adam(learning_rate=eta)
    metricas = [keras.metrics.CosineSimilarity(name="Cosseno"), keras.metrics.MeanAbsoluteError(name="Vies"), keras.metrics.LogCoshError(name="LogCosh")]
    modelo.compile(optimizer=otimizador, loss=keras.losses.MeanSquaredError(name="MSE"), metrics=metricas)
    modelo.summary()
    keras.utils.plot_model(modelo, to_file=os.path.join(path_saida, "RedeAerofolio.png"), show_shapes=True, show_layer_names=True, show_layer_activations=True, show_trainable=True, dpi=300)
    callback_taxa = keras.callbacks.LearningRateScheduler(lambda *args: scheduler(*args, k=decaimento))
    callback_log = keras.callbacks.CSVLogger(os.path.join(path_saida, "Log.csv"))
    callback_parada = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, min_delta=1E-4)
    modelo.fit(x, y, validation_data=(x_val, y_val), batch_size=64, epochs=100, callbacks=[callback_taxa, callback_log, callback_parada], shuffle=True)
    avaliacao = modelo.evaluate(x_val, y_val)
    modelo.save(os.path.join(path_saida, "Modelo.keras"))
    log=plota_log(os.path.join(path_saida,"Log.csv"))
    ymed = y_val[-1024:]
    ypred = modelo(x_val[-1024:])
    plt.figure()
    plt.scatter(ymed[:,0],ypred[:,0], label="c_L")
    plt.scatter(ymed[:,1],ypred[:,1], label="c_D")
    plt.scatter(ymed[:, 2], ypred[:, 2], label="c_M")
    plt.legend()
    plt.savefig(os.path.join(path_saida,"Desempenho da rede - Aerofolio fino.png"), dpi=300)
    return modelo, avaliacao




if __name__=="__main__":
    modelo1, avaliacao1=treinar_rede( 0.001, 0.95, 0.01,1,50)
    modelo2, avaliacao2=treinar_rede( 0.001, 0.95, 0.01,2,50)
    modelo3, avaliacao3=treinar_rede( 0.001, 0.95, 0.01,3,50)
    modelo4, avaliacao4=treinar_rede( 0.001, 0.95, 0.01,4,50)
    tabela=pd.DataFrame(columns=["perda","cosseno", "vies", "logcosh"],data=np.vstack([avaliacao1,avaliacao2,avaliacao3,avaliacao4]))
    print(tabela)




    plt.show(block=False)
    plt.show(block=True)



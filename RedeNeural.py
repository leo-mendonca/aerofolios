# import keras
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
        # vol=CalculaVolume(name="CalculoVolume")(entrada)
        x = keras.layers.Dense(units=n_neuronios, activation=kact.tanh, use_bias=True, kernel_regularizer=keras.regularizers.L2(lamda))(entrada)
        for i in range(n_camadas-1):
            x=keras.layers.Dense(units=n_neuronios, activation=kact.tanh, use_bias=True)(x)
        saida=keras.layers.Dense(units=3, activation=None, use_bias=True, name="SaidaMecanica")(x)
        # saida_completa=keras.layers.Concatenate(axis=-1, name="Concatenacao")([saida, vol])
        super(RedeAerofolio,self).__init__(entrada, saida, **kwargs)
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

def carrega_dados(path, usa_volume=False):
    '''Le um arquivo .csv contendo os resultados do MEF e retorna os conjuntos de treino, teste e validacao como tensores tf'''
    dados_dframe=pd.read_csv(path, sep=";", skipinitialspace=True)
    dados_dframe = Salvamento.filtragem_outliers(dados_dframe)
    if not usa_volume:
        try: dados_dframe.drop(columns=["V"], inplace=True)
        except KeyError: pass
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
    plt.plot(log["epoch"].array[1:], log["loss"].array[1:], label="Treino")
    plt.plot(log["epoch"].array[1:], log["val_loss"].array[1:], label="Validacao")
    plt.legend()
    plt.xlabel("Epoca")
    plt.ylabel("Erro")
    path_diretorio=os.path.dirname(path)
    plt.savefig(os.path.join(path_diretorio,"Evolucao treinamento.png"), dpi=300)
    plt.figure()
    plt.plot(log["epoch"].array, log["lr"].array, label="aprendizado")
    plt.xlabel("Epoca")
    plt.ylabel("Taxa de aprendizado")
    return log

def treinar_rede(eta, decaimento, lamda, n_camadas, neuronios, path_dados=os.path.join("Entrada", "Dados", "dados_mef_v2.csv")):
    '''Treina uma rede neural com os parametros fornecidos e salva os resultados em uma pasta'''
    nome_caso=f"Rede eta={eta} k={decaimento} lambda={lamda} camadas={n_camadas}x{neuronios}"
    nome_modelo="Rede"
    path_saida=os.path.join("Saida","Redes Neurais", nome_caso)
    os.makedirs(path_saida, exist_ok=True)
    x, y, x_val, y_val, x_teste, y_teste = carrega_dados(path_dados)
    modelo = RedeAerofolio(n_camadas, neuronios, lamda, name=nome_modelo)
    otimizador = kopt.Adam(learning_rate=eta)
    metricas = [keras.metrics.CosineSimilarity(name="Cosseno"), keras.metrics.MeanSquaredError(name="EQM"), MetricaEQMComponente(0, name="EQM_D"), MetricaEQMComponente(1, name="EQM_L"), MetricaEQMComponente(2, name="EQM_M")]
    modelo.compile(optimizer=otimizador, loss=keras.losses.MeanSquaredError(name="MSE"), metrics=metricas)
    # modelo.summary()
    keras.utils.plot_model(modelo, to_file=os.path.join(path_saida, "RedeAerofolio.png"), show_shapes=True, show_layer_names=True, show_layer_activations=True, show_trainable=True, dpi=300)
    callback_taxa = keras.callbacks.LearningRateScheduler(lambda *args: scheduler(*args, k=decaimento))
    callback_log = keras.callbacks.CSVLogger(os.path.join(path_saida, "Log.csv"))
    callback_parada = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, min_delta=1E-4)
    t0=time.process_time()
    modelo.fit(x, y, validation_data=(x_val, y_val), batch_size=64, epochs=100, callbacks=[callback_taxa, callback_log, callback_parada], shuffle=True)
    t1=time.process_time()
    tempo=t1-t0
    with open(os.path.join(path_saida, "Tempo.txt"), "w") as arquivo:
        arquivo.write(f"{tempo} s")
    avaliacao = modelo.evaluate(x_val, y_val)
    modelo.save(os.path.join(path_saida, "Modelo.keras"))
    log=plota_log(os.path.join(path_saida,"Log.csv"))
    ymed = y_val
    ypred = modelo(x_val)
    plt.figure()
    plt.scatter(ymed[:,0],ypred[:,0], label="c_D", s=5)
    plt.scatter(ymed[:,1],ypred[:,1], label="c_L",s=5)
    plt.scatter(ymed[:, 2], ypred[:, 2], label="c_M",s=5)
    plt.legend()
    plt.savefig(os.path.join(path_saida,"Desempenho da rede"), dpi=300)
    return modelo, avaliacao, tempo

@keras.saving.register_keras_serializable()
def metrica_eqm_componente(y_true, y_pred, n):
    '''Calcula o erro quadratico medio da n-esima compnente do vetor de saida'''
    return keras.losses.mean_squared_error(y_true[:,n], y_pred[:,n])

@keras.saving.register_keras_serializable()
class MetricaEQMComponente(keras.metrics.MeanMetricWrapper):
    def __init__(self, n, name="EQM",dtype=None):
        self.n=n
        super(MetricaEQMComponente, self).__init__(lambda y_true, y_pred: metrica_eqm_componente(y_true, y_pred, n),name=name,dtype=dtype)

    def get_config(self):
        '''Salva as configuracoes da camada para que ela possa ser reconstruida pelo keras'''
        base_config=super().get_config()
        config= {"n":self.n}
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        '''Reconstrói a camada a partir das configuracoes salvas'''
        n=config.pop("n")
        return cls(n, **config)


def carregar_rede(eta, k, lamda, camadas, neuronios, path_dados=os.path.join("Entrada", "Dados", "dados_mef_v2.csv")):
    '''Carrega os pesos e os resultados de uma rede neural ja devidamente treinada e analisa o desempenho dessa rede nos dados de validacao'''
    nome_caso=f"Rede eta={eta} k={k} lambda={lamda} camadas={camadas}x{neuronios}"
    path_saida = os.path.join("Saida", "Redes Neurais", nome_caso)
    try: modelo = keras.models.load_model(os.path.join(path_saida, "Modelo.keras"), custom_objects={"CalculaVolume":CalculaVolume})
    except: modelo = keras.models.load_model(os.path.join(path_saida, "Modelo.keras"))
    x, y, x_val, y_val, x_teste, y_teste = carrega_dados(path_dados)
    avaliacao = modelo.evaluate(x_val, y_val)
    with open(os.path.join(path_saida, "Tempo.txt"), "r") as arquivo:
        tempo=float(arquivo.read().rstrip(" s"))
    return modelo, avaliacao, tempo



def analisa_hiperparametros(valores_eta, valores_k, valores_lambda, camadas, neuronios, executa=True, path_dados=os.path.join("Entrada", "Dados", "dados_mef_v2.csv")):
    '''Analisa o desempenho de uma rede neural variando os hiperparametros eta, k e lambda'''
    resultados=pd.DataFrame(index=pd.MultiIndex.from_product((camadas,neuronios,valores_eta, valores_k, valores_lambda)), columns=["perda","cosseno", "vies", "logcosh", "tempo"], dtype=np.float64)
    resultados = pd.DataFrame(index=pd.MultiIndex.from_product((camadas, neuronios,valores_eta, valores_k, valores_lambda)), columns=["perda", "cosseno", "EQM", "EQM_D", "EQM_L", "EQM_M", "tempo"], dtype=np.float64)
    for eta in valores_eta:
        for k in valores_k:
            for lamda in valores_lambda:
                for n_camadas in camadas:
                    for n_neuronios in neuronios:
                        print(f"Avaliando rede com eta={eta}, k={k}, lambda={lamda}, camadas={n_camadas}x{n_neuronios}")
                        if executa:
                            modelo, avaliacao, tempo = treinar_rede(eta, k, lamda, n_camadas, n_neuronios, path_dados)
                        else:
                            modelo, avaliacao, tempo = carregar_rede(eta, k, lamda, n_camadas, n_neuronios, path_dados)
                        resultados.loc[(n_camadas, n_neuronios,eta,k,lamda)] = np.concatenate([avaliacao, [tempo]])
                        print(resultados.loc[(n_camadas, n_neuronios,eta,k,lamda)])
                        plt.close("all")
    resultados.to_csv(os.path.join("Saida", "Redes Neurais", "Comparacao hiperparametros.csv"), sep=";", index_label=["camadas","neuronios","eta", "k", "lambda"])
    print(resultados)
    return resultados

def plotar_saida_arquitetura(path_resultados, plot="arquitetura"):
    '''plota os resultados da analise de arquitetura. Avalia o erro e o tempo computacional em funcao do numero de neuronios e camadas'''
    resultados=pd.read_csv(path_resultados, sep=";", skipinitialspace=True)
    if plot=="arquitetura":
        grandeza1=resultados["camadas"]
        grandeza2=resultados["neuronios"]
        label1=u"Camadas"
        label2=u"Neurônios"
    elif plot=="aprendizado":
        grandeza1=resultados["eta"]
        grandeza2=resultados["k"]
        label1=r"Taxa de aprendizado inicial $\eta_0$"
        label2=r"Taxa de decaimento $\gamma$"
    e=resultados["EQM"]
    tempo=resultados["tempo"]
    e_log=np.log10(e)
    t_log=np.log10(tempo)
    with plt.rc_context({"text.usetex":True}):
        fig1=plt.figure()
        eixo1=fig1.add_subplot(projection="3d")
        log_locator=lambda x, pos=None: f"$10^{{{int(x)}}}$"
        eixo1.zaxis.set_major_formatter(mtick.FuncFormatter(log_locator))
        eixo1.zaxis.set_major_locator(mtick.MaxNLocator(integer=True))
        eixo1.plot_trisurf(grandeza1, grandeza2, e_log)
        eixo1.set_xlabel(label1)
        eixo1.set_ylabel(label2)
        eixo1.set_zlabel(u"Erro Quadrático Médio")

        fig2=plt.figure()
        eixo2=fig2.add_subplot(projection="3d")
        eixo2.plot_trisurf(grandeza1, grandeza2, tempo)
        eixo2.set_xlabel(label1)
        eixo2.set_ylabel(label2)
        eixo2.set_zlabel("Tempo [s]")

        ##Faz um plot heatmap das mesmas grandezas acima
        fig3, eixo3=plt.subplots()
        fig3.set_size_inches(6,4)
        eixo3.grid(False)
        eixo3.set_xlabel(label1)
        eixo3.set_ylabel(label2)
        eixo3.set_title(u"Erro Quadrático Médio")
        if plot=="arquitetura":
            x_grid, y_grid=np.meshgrid(grandeza1.unique(), grandeza2.unique())
        elif plot=="aprendizado":
            x_grid, y_grid=np.meshgrid(np.log10(grandeza1.unique()), grandeza2.unique())
        e_grid=np.array([[resultados.loc[np.logical_and(grandeza1==y, grandeza2==x),"EQM"].values[0]  for y in grandeza1.unique()]for x in grandeza2.unique()])
        mapa_erro=eixo3.pcolormesh(x_grid, y_grid, e_grid, cmap="turbo", norm=mcolors.LogNorm(), shading="nearest")
        eixo3.xaxis.set_major_formatter(mtick.FuncFormatter(log_locator))
        eixo3.xaxis.set_major_locator(mtick.MaxNLocator(integer=True))
        # eixo3.imshow(e.values.reshape(len(camadas.unique()), len(neuronios.unique())), cmap="turbo", origin="lower", norm=mcolors.LogNorm())
        fig3.colorbar(mapa_erro)
        plt.savefig(os.path.join("Saida","Redes Neurais","Mapa de calor EQM.png"), dpi=300, bbox_inches="tight", transparent=True)
        fig4, eixo4=plt.subplots()
        eixo4.grid(False)
        fig4.set_size_inches(6,4)
        eixo4.set_xlabel(label1)
        eixo4.set_ylabel(label2)
        eixo4.set_title("Tempo [s]")

        t_grid=np.array([[resultados.loc[np.logical_and(grandeza1==y, grandeza2==x),"tempo"].values[0]  for y in grandeza1.unique()]for x in grandeza2.unique()])
        mapa_tempo=eixo4.pcolormesh(x_grid, y_grid, t_grid, cmap="turbo", shading="nearest")
        eixo4.xaxis.set_major_formatter(mtick.FuncFormatter(log_locator))
        eixo4.xaxis.set_major_locator(mtick.MaxNLocator(integer=True))
        # eixo4.imshow(tempo.values.reshape(len(camadas.unique()), len(neuronios.unique())), cmap="turbo", origin="lower")
        fig4.colorbar(mapa_tempo)
        plt.savefig(os.path.join("Saida","Redes Neurais","Mapa de calor Tempo.png"), dpi=300, bbox_inches="tight", transparent=True)
    return

def plotar_saida_lambda(path_resultados):
    resultados=pd.read_csv(path_resultados, sep=";", skipinitialspace=True)
    with plt.rc_context({"text.usetex":True}):
        fig, eixo=plt.subplots()
        eixo.set_xlabel(r"Parâmetro de regularização $\lambda$")
        eixo.set_ylabel(u"Erro Quadrático Médio")
        # eixo.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: f"$10^{{{int(x)}}}$"))
        # eixo.yaxis.set_major_locator(mtick.MaxNLocator(integer=True))
        # eixo.scatter(resultados["lambda"].array, np.log10(resultados["EQM"].array))
        eixo.loglog(resultados["lambda"].array, resultados["EQM"].array, marker="o", linestyle="none")
        plt.savefig(os.path.join("Saida","Redes Neurais","Lambda erro.png"), dpi=300, bbox_inches="tight", transparent=True)
        fig2, eixo2=plt.subplots()
        eixo2.set_xlabel(r"Parâmetro de regularização $\lambda$")
        eixo2.set_ylabel("Tempo [s]")
        # eixo2.scatter(resultados["lambda"].array, resultados["tempo"].array)
        eixo2.semilogx(resultados["lambda"].array, resultados["tempo"].array, marker="o", linestyle="none")
        plt.savefig(os.path.join("Saida","Redes Neurais","Lambda tempo.png"), dpi=300, bbox_inches="tight", transparent=True)
    return

if __name__=="__main__":
    modelo,avaliacao,tempo=treinar_rede(4.6E-3,0.96,2.5E-6,4,80)
    print(avaliacao)
    print(f"tempo= {tempo} s")
    colunas=["perda", "cosseno", "EQM", "EQM_D", "EQM_L", "EQM_M"]
    [print(f"{coluna}: {avaliacao[i]}") for i,coluna in enumerate(colunas)]
    raise SystemExit
    # modelo1, avaliacao1=treinar_rede( 0.001, 0.95, 0.01,1,50)
    # modelo2, avaliacao2=treinar_rede( 0.001, 0.95, 0.01,2,50)
    # modelo3, avaliacao3=treinar_rede( 0.001, 0.95, 0.01,3,50)
    # modelo4, avaliacao4=treinar_rede( 0.001, 0.95, 0.01,4,50)
    # modelo5, avaliacao5=treinar_rede( 0.001, 0.95, 0.01,5,50)
    # modelo6, avaliacao6=treinar_rede( 0.001, 0.95, 0.01,6,50)
    # tabela=pd.DataFrame(columns=["perda","cosseno", "vies", "logcosh"],data=np.vstack([avaliacao1,avaliacao2,avaliacao3,avaliacao4,avaliacao5,avaliacao6]))
    # print(tabela)
    ##Arquitetura 6x50, lambda=1E-5
    # valores_lambda=[0.00001, 1E-6, 1E-7]
    # resultados=pd.DataFrame(index=valores_lambda,columns=["perda","cosseno", "vies", "logcosh", "tempo"], dtype=np.float64)
    # for i,lamda in enumerate(valores_lambda):
    #     t0=time.process_time()
    #     modelo, avaliacao = treinar_rede(0.001, 0.95, lamda, 6, 50)
    #     t1=time.process_time()
    #     tempo=t1-t0
    #     resultados.loc[lamda]=np.concatenate([avaliacao, [tempo]])
    #     print(resultados.loc[lamda])
    #     plt.close("all")
    # resultados.to_csv(os.path.join("Saida","Redes Neurais","Comparacao lambda.csv"), sep=";", index_label="lambda")
    # print(resultados)
    # modelo,avaliacao,tempo=treinar_rede(0.001, 0.95, 1E-5, 1, 10)
    # modelo2,avaliacao2,tempo2=carregar_rede(0.001, 0.95, 1E-5, 1, 10)
    # plotar_saida_arquitetura(os.path.join("Saida", "Redes Neurais", "Comparacao", "Comparacao arquitetura.csv"))
    plotar_saida_lambda(os.path.join("Saida", "Redes Neurais", "Comparacao", "Comparacao lambda.csv"))
    plotar_saida_arquitetura(os.path.join("Saida", "Redes Neurais", "Comparacao", "Comparacao aprendizado.csv"), plot="aprendizado")
    plt.show(block=False)
    ##Arquitetura: 8x40
    # k = [0.95,]
    k=np.linspace(0.70,0.98,15)
    # eta = [0.001,]
    eta=np.logspace(-5,-2,10)
    lamda = [2.5E-6,]
    # lamda=np.logspace(-7,-1,31)
    # lamda=[4E-7,]
    # camadas = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # neuronios = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    camadas=[4,]
    neuronios=[80,]
    resultados=analisa_hiperparametros(eta, k, lamda, camadas, neuronios, executa=True)




    # modelo,avaliacao=treinar_rede(eta, k, lamda, 6, 51)
    # resultados = pd.DataFrame(index=pd.MultiIndex.from_product((camadas, neuronios)), columns=["perda", "cosseno", "perda_D", "perda_L", "perda_M", "tempo"], dtype=np.float64)
    # for n_camadas in camadas:
    #     for n_neuronios in neuronios:
    #         t0 = time.process_time()
    #         modelo, avaliacao = treinar_rede(eta, k, lamda, n_camadas, n_neuronios)
    #         t1 = time.process_time()
    #         tempo = t1 - t0
    #         resultados.loc[(n_camadas, n_neuronios)] = np.concatenate([avaliacao, [tempo]])
    #         print(resultados.loc[(n_camadas, n_neuronios)])
    #         plt.close("all")
    # resultados.to_csv(os.path.join("Saida", "Redes Neurais", "Comparacao arquitetura.csv"), sep=";", index_label=["camadas", "neuronios"])
    # print(resultados)





    # plt.show(block=False)
    # plt.show(block=True)



import numpy as np

from Definicoes import *
import Execucao
import RedeNeural
import AerofolioFino
import TestesMEF
import keras
import pickle

def compara_metodos(m,p,t,Re, picles=False):
    '''Faz um grafico comparando os metodos de elementos finitos, rede neural e teoria de aerofolio fino.
    Calcula os coeficientes e um mesmo aerofolio em angulos diferentes e compara os resultados.'''
    path_saida=os.path.join("Saida","Comparacao metodos")
    if not picles:
        alfas=np.linspace(-15,+15,1001)*np.pi/180
        alfas_mef=np.arange(-15,16,1)*np.pi/180
        vetor_entrada=np.array([[m,p,t, alfa,Re] for alfa in alfas], dtype=np.float64)
        modelo=keras.models.load_model(os.path.join("Entrada","Rede neural","Modelo.keras"))
        saidas_rede=modelo(vetor_entrada)
        saidas_af=np.zeros((len(alfas),3), dtype=np.float64)
        for i, alfa in enumerate(alfas):
            af=AerofolioFino.AerofolioFinoNACA4([m,p,t],alfa=alfa,U0=Re)
            saidas_af[i]=(af.c_L, af.c_D, af.c_M)
        saidas_mef=np.zeros((len(alfas_mef),3),dtype=np.float64)
        for i, alfa in enumerate(alfas_mef):
            print(alfa*180/np.pi)
            alfa*=-1 ## Corrigindo o sinal de alfa #TODO implementar solucao definitiva
            af=AerofolioFino.AerofolioFinoNACA4([m,p,t],alfa=alfa,U0=Re)
            c_l,c_d,c_M=TestesMEF.teste_aerofolio(af, Re=Re, executa=False, plota_tudo=False, desenha_aerofolio=False)
            saidas_mef[i]=(c_l,c_d,c_M)
        with open(os.path.join(path_saida, "resultados.pkl"), "wb") as f:
            pickle.dump((alfas, saidas_rede, saidas_af, alfas_mef, saidas_mef), f)
    elif picles:
        with open(os.path.join(path_saida, "resultados.pkl"), "rb") as f:
            alfas, saidas_rede, saidas_af, alfas_mef, saidas_mef=pickle.load(f)
    with plt.rc_context({"text.usetex" : True}):
        fig,eixo=plt.subplots()
        fig.set_size_inches(8,5)
        eixo.plot(alfas*180/np.pi, saidas_rede[:,0], label="Rede Neural", zorder=10)
        eixo.plot(alfas*180/np.pi, saidas_af[:,0], label="Aerofolio Fino")
        eixo.plot(alfas_mef*180/np.pi, saidas_mef[:,0], label="Elementos Finitos", marker="*", linestyle="none")
        eixo.set_xlabel(r"$\alpha \ [ ^\circ ]$")
        eixo.set_ylabel(r"$C_L$")
        eixo.legend()
        plt.savefig(os.path.join(path_saida,"Comparacao_CL.png"),bbox_inches="tight",dpi=300)
        fig2,eixo2=plt.subplots()
        fig2.set_size_inches(8,5)
        eixo2.plot(alfas*180/np.pi, saidas_rede[:,1], label="Rede Neural", zorder=10)
        eixo2.plot(alfas*180/np.pi, saidas_af[:,1], label="Aerofolio Fino")
        eixo2.plot(alfas_mef*180/np.pi, saidas_mef[:,1], label="Elementos Finitos", marker="*", linestyle="none")
        eixo2.set_xlabel(r"$\alpha \ [ ^\circ ]$")
        eixo2.set_ylabel(r"$C_D$")
        eixo2.legend()
        plt.savefig(os.path.join(path_saida,"Comparacao_CD.png"),bbox_inches="tight",dpi=300)
        fig3,eixo3=plt.subplots()
        fig3.set_size_inches(8,5)
        eixo3.plot(alfas*180/np.pi, saidas_rede[:,2], label="Rede Neural", zorder=10)
        eixo3.plot(alfas*180/np.pi, saidas_af[:,2], label="Aerofolio Fino")
        eixo3.plot(alfas_mef*180/np.pi, saidas_mef[:,2], label="Elementos Finitos", marker="*", linestyle="none")
        eixo3.set_xlabel(r"$\alpha \ [ ^\circ ]$")
        eixo3.set_ylabel(r"$C_M$")
        eixo3.legend()
        plt.savefig(os.path.join(path_saida,"Comparacao_CM.png"),bbox_inches="tight",dpi=300)

def avalia_dados_teste():
    '''Avalia os dados de teste da rede neural.'''
    modelo=keras.models.load_model(os.path.join("Entrada","Rede neural","Modelo.keras"))
    x, y, x_val, y_val, x_teste, y_teste = RedeNeural.carrega_dados(os.path.join("Entrada","Dados","dados_mef_alfa_corrigido.csv"))
    entradas=x_teste
    saidas=y_teste
    t0=time.process_time()
    saidas_rede=modelo(entradas)
    t1=time.process_time()
    erros=saidas_rede-saidas
    print("Erro maximo:", np.max(np.abs(erros)))
    print("Erro medio de vies:", np.mean(erros))
    print("Erro quadratico medio:", np.mean(erros**2))
    print("EQM sustentacao:", np.mean(erros[:,0]**2))
    print("EQM arrasto:", np.mean(erros[:,1]**2))
    print("EQM momento:", np.mean(erros[:,2]**2))
    print("Tempo de execucao:", t1-t0)

    saidas_af=np.zeros((len(saidas),3), dtype=np.float64)
    x_t=np.array(x_teste)
    t2=time.process_time()
    for i in range(len(saidas)):
        aerofolio=AerofolioFino.AerofolioFinoNACA4([x_t[i,0],x_t[i,1],x_t[i,2]],alfa=x_t[i,3],U0=x_t[i,4])
        saidas_af[i]=Execucao.teoria_af_fino(aerofolio)
    t3=time.process_time()
    erros_af=saidas_af-saidas
    print("Erro maximo AF:", np.max(np.abs(erros_af)))
    print("Erro medio de vies AF:", np.mean(erros_af))
    print("Erro quadratico medio AF:", np.mean(erros_af**2))
    print("EQM sustentacao AF:", np.mean(erros_af[:,0]**2))
    print("EQM arrasto AF:", np.mean(erros_af[:,1]**2))
    print("EQM momento AF:", np.mean(erros_af[:,2]**2))
    print("Tempo de execucao AF:", t3-t2)



if __name__=="__main__":
    compara_metodos(0.04,0.40,0.12,100, picles=True)
    avalia_dados_teste()
    plt.show()
from Definicoes import np, pd
from Definicoes import os, time
import AerofolioFino
import ElementosFinitos


def teoria_af_fino(aerofolio):
    '''Obtem os coeficientes dinamicos do aerofolio pela teoria de aerofolio fino'''
    c_L = aerofolio.c_L
    c_D = aerofolio.c_D
    c_M = aerofolio.c_M
    return c_L, c_D, c_M


def calculo_mef(aerofolio):
    '''Obtem os coeficientes dinamicos do aerofolio pelo metodo dos elementos finitos'''
    c_L, c_D, c_M = ElementosFinitos.calculo_aerofolio(aerofolio)
    return c_L, c_D, c_M
def calculo_mef_grosseiro(aerofolio):
    '''Obtem os coeficientes dinamicos do aerofolio pelo metodo dos elementos finitos'''
    c_L, c_D, c_M = ElementosFinitos.calculo_aerofolio(aerofolio, grosseiro=True)
    return c_L, c_D, c_M


def gerar_banco_dados(distribuicoes, n_amostras, path_salvar=None, metodo=teoria_af_fino):
    '''Produz uma tabela de valores de entrada e saida de resultados de aerofolio fino NACA4.
    :param distribuicoes: list. Lista de distribuicoes aleatorias, uma para cada parametro. Devem receber n e retornar um vetor de n amostras aleatorias
    :param n_amostras: int. Numero de casos a gerar
    :param path_salvar: str. Local para salvar os resultados em um arquivo .csv. Se None, os resultados nao sao salvos
    '''
    nvars = len(distribuicoes)
    amostragens = []
    for i in range(nvars):
        amostragens.append(distribuicoes[i](n_amostras))
    params_entrada = np.vstack(amostragens).T
    resultados = []
    for i in range(n_amostras):
        m, p, t, alfa, U0 = params_entrada[i]
        af = AerofolioFino.AerofolioFinoNACA4(vetor_coeficientes=[m, p, t], alfa=alfa, U0=U0)
        print(f"Calculando aerofolio {af.nome}...")
        c_L, c_D, c_M = metodo(af)
        V = af.volume
        resultados.append([c_L, c_D, c_M, V])
    matriz_resultados = np.vstack(resultados)
    saida = np.hstack((params_entrada, matriz_resultados))
    dframe = pd.DataFrame(saida, columns=["M", "P", "T", "alfa", "U", "c_L", "c_D", "c_M", "V"])
    if not path_salvar is None:
        ##Se ja existir um arquivo de resultados, salvar com outro nome
        nome_tentativo=path_salvar
        n=0
        while os.path.exists(nome_tentativo):
            nome_tentativo = path_salvar[:-4] + f" {n}.csv"
            n+=1
        dframe.to_csv(nome_tentativo)
    return dframe


if __name__ == "__main__":
    def distro_p(n):
        amostra = np.random.normal(0.40, 0.10, size=n)
        amostra[amostra < 0.1] = 0.1
        amostra[amostra > 0.9] = 0.9
        return amostra

    distro_m = lambda n: np.random.uniform(-0.10, +0.10, n)

    def distro_t(n):
        amostra = np.random.normal(0.12, 0.05, n)
        amostra[amostra < 0.01] = 0.01
        return amostra

    distro_alfa = lambda n: np.random.normal(0, 5*np.pi/180, n)
    distro_U = lambda n: 1 * np.random.weibull(3, n)
    def distro_m_positivo(n):
        ##Seleciona 20% das amostras para serem nulas (aerofolio simetrico), e o restante seguira uma distribuicao uniforme
        bool_positivos=np.random.rand(n)>0.2
        positivo=np.random.uniform(0, 0.10, np.count_nonzero(bool_positivos))
        zero=np.zeros(np.count_nonzero(~bool_positivos),dtype=np.float64)
        saida=np.concatenate((positivo, zero))
        np.random.shuffle(saida)
        return saida

    distro_p_uniforme = lambda n: np.random.uniform(0.1, 0.9, n)
    distro_t_uniforme = lambda n: np.random.uniform(0.05, 0.3, n)
    distro_alfa_uniforme = lambda n: np.random.uniform(-15*np.pi/180, 15*np.pi/180, n)
    distro_U_100 = lambda n: 100 * np.random.weibull(2, n)
    distribuicoes = [distro_m, distro_p, distro_t, distro_alfa, distro_U]
    distribuicoes_V2 = [distro_m_positivo, distro_p_uniforme, distro_t_uniforme, distro_alfa_uniforme, distro_U_100]
    n_lotes=30
    for i in range(n_lotes):
        t0 = time.process_time()
        n_amostras=10
        print(f"Lote {i+1}/{n_lotes}")
        print(f"{n_amostras*i} casos ja foram executados")
        banco = gerar_banco_dados(distribuicoes_V2, n_amostras=n_amostras, path_salvar="Saida/MEF_NACA4/resultados_v2.csv", metodo=calculo_mef)
        t1=time.process_time()
        print(banco)
        print(f"Tempo de execucao: {t1-t0:.2f} s")

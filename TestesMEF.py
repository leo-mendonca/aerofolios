import AerofolioFino
import Malha
import ElementosFinitos
import RepresentacaoEscoamento
import time
import os
import pickle
from Definicoes import *
from Salvamento import carregar_resultados, cria_diretorio, salvar_resultados


def teste_cilindro(n=50):
    cilindro = AerofolioFino.Cilindro(.5, 0, 1)
    nome_malha, tag_fis = Malha.malha_aerofolio(cilindro, n_pontos_contorno=n)
    Problema = ElementosFinitos.FEA(nome_malha, tag_fis)
    ux_dirichlet = [
        (Problema.nos_cont["esquerda"], lambda x: 1.),
        (Problema.nos_cont["superior"], lambda x: 0.),
        (Problema.nos_cont["inferior"], lambda x: 0.),
        (Problema.nos_cont["af"], lambda x: 0.),
    ]
    uy_dirichlet = [
        (Problema.nos_cont["esquerda"], lambda x: 0.),
        (Problema.nos_cont["superior"], lambda x: 0.),
        (Problema.nos_cont["inferior"], lambda x: 0.),
        (Problema.nos_cont["af"], lambda x: 0.),
    ]
    p_dirichlet = [(Problema.nos_cont_o1["direita"], lambda x: 0.), ]
    resultados = Problema.escoamento_IPCS_NS(ux_dirichlet=ux_dirichlet, uy_dirichlet=uy_dirichlet, p_dirichlet=p_dirichlet, T=10, dt=0.1, Re=1, conveccao=True)
    RepresentacaoEscoamento.plotar_momento(Problema, resultados, 10)
    # RepresentacaoEscoamento.plotar_perfis(Problema, resultados, 10, lim_x=(-2,0))
    return


def numeracao_nos():
    cilindro = AerofolioFino.Cilindro(.5, 0, 1)
    nome_malha, tag_fis = Malha.malha_aerofolio(cilindro, n_pontos_contorno=10)
    Problema = ElementosFinitos.FEA(nome_malha, tag_fis)
    elem0 = Problema.elementos[0]
    x, y, z = Problema.x_nos[elem0].T
    plt.triplot(Problema.x_nos[:, 0], Problema.x_nos[:, 1], triangles=[Problema.elementos_o1[0], ], alpha=0.5)
    plt.scatter(x, y)
    for i in range(len(x)):
        plt.text(x[i], y[i], i)
    plt.show(block=False)
    return


def teste_forca(n=20, tamanho=0.1, p0=0., debug=False, executa=True, formulacao="F", T=3, dt=0.01, Re=1.):
    nome_diretorio = f"Saida/Cilindro/Cilindro n={n} h={tamanho} dt={dt} Re={Re} T={T} {formulacao}"
    if executa:
        cilindro = AerofolioFino.Cilindro(.5, 0, 1)
        nome_malha, tag_fis = Malha.malha_aerofolio(cilindro, n_pontos_contorno=n, tamanho=tamanho)
        Problema = ElementosFinitos.FEA(nome_malha, tag_fis)
        ux_dirichlet = [
            (Problema.nos_cont["esquerda"], lambda x: 1.),
            (Problema.nos_cont["superior"], lambda x: 1.),
            (Problema.nos_cont["inferior"], lambda x: 1.),
            (Problema.nos_cont["af"], lambda x: 0.),
        ]
        uy_dirichlet = [
            (Problema.nos_cont["esquerda"], lambda x: 0.),
            (Problema.nos_cont["superior"], lambda x: 0.),
            (Problema.nos_cont["inferior"], lambda x: 0.),
            (Problema.nos_cont["af"], lambda x: 0.),
        ]
        p_dirichlet = [(Problema.nos_cont_o1["direita"], lambda x: p0), ]

        resultados = Problema.escoamento_IPCS_NS(ux_dirichlet=ux_dirichlet, uy_dirichlet=uy_dirichlet, p_dirichlet=p_dirichlet, T=T, dt=dt, Re=Re, u0=1., p0=p0, formulacao=formulacao)
        nome_diretorio = cria_diretorio(nome_diretorio)
        nome_arquivo = os.path.join(nome_diretorio, f" n={n} h={tamanho} dt={dt} Re={Re} T={T} {formulacao}.zip")
        salvar_resultados(nome_malha, tag_fis, resultados, nome_arquivo)
        RepresentacaoEscoamento.plotar_momento(Problema, resultados, T)
        u = resultados[T]["u"]
        p = resultados[T]["p"]
    else:

        # with open("Picles/resultados_forca.pkl", "rb") as f:
        #     Problema, resultados = pickle.load(f)
        nome_arquivo = os.path.join(nome_diretorio, f" n={n} h={tamanho} dt={dt} Re={Re} T={T} {formulacao}.zip")
        Problema, u, p, nome_malha = carregar_resultados(nome_arquivo)

    forca, x, tensao = Problema.calcula_forcas(p, u, debug=debug, viscosidade=True, Re=Re)
    forca_p, x, tensao_p = Problema.calcula_forcas(p, u, debug=debug, viscosidade=False, Re=Re)
    F = np.sum(forca, axis=0)
    F_p = np.sum(forca_p, axis=0)
    rho = 1.
    U0 = 1.
    D = 1.
    c_d = F[0] / (0.5 * rho * U0 ** 2 * D)
    c_l = F[1] / (0.5 * rho * U0 ** 2 * D)
    x_rel = x - np.array([0.5, 0])
    M = np.sum(np.cross(x_rel, forca), axis=0)
    c_M = M / (0.5 * rho * U0 ** 2 * D ** 2)
    pressao_a = Problema.interpola(np.array([0., 0.]), p, ordem=1)
    pressao_b = Problema.interpola(np.array([1., 0.]), p, ordem=1)
    c_p_a = pressao_a / (0.5 * rho * U0 ** 2)
    c_p_b = pressao_b / (0.5 * rho * U0 ** 2)
    c_d_p = F_p[0] / (0.5)
    c_l_p = F_p[1] / (0.5)
    print(f"Coeficiente de arrasto devido a pressao: {c_d_p}")
    print(f"Coeficiente de arrasto devido ao atrito: {c_d - c_d_p}")
    print(f"Coeficiente de arrasto total: {c_d}")
    print(f"Coeficiente de sustentacao devido a pressao: {c_l_p}")
    print(f"Coeficiente de sustentacao total: {c_l}")
    print(f"Coeficiente de Momento: {c_M}")
    print(f"Coeficiente de pressao de estagnacao: {c_p_a}")
    print(f"Coeficiente de pressao de saida: {c_p_b}")
    coeficientes = c_d_p, c_d - c_d_p, c_d, c_l, c_M, c_p_a, c_p_b
    vetor_F = forca / 200
    vetor_tensao = tensao / 200
    plt.figure()
    theta = np.arange(0, 2 * np.pi, 0.01)
    plt.plot(.5 + 0.5 * np.cos(theta), 0.5 * np.sin(theta), 'b-', alpha=0.3, linewidth=2)
    for i in range(len(x)):
        plt.plot([x[i, 0], x[i, 0] + vetor_tensao[i, 0]], [x[i, 1], x[i, 1] + vetor_tensao[i, 1]], 'k-')
    (x, y), mapa_p = RepresentacaoEscoamento.mapa_de_cor(Problema, p, ordem=1, resolucao=0.05, titulo=u"Pressão")
    ##Salvando os resultados
    resolucao = 0.05
    x = np.arange(Problema.x_min, Problema.x_max + resolucao, resolucao)
    y = np.arange(Problema.y_min, Problema.y_max + resolucao, resolucao)
    localizacao = Problema.localiza_grade(x, y)
    (x, y), mapa_p = RepresentacaoEscoamento.mapa_de_cor(Problema, p, ordem=1, resolucao=None, x_grade=x, y_grade=y, local_grade=localizacao, titulo=u"Pressão", path_salvar=os.path.join(nome_diretorio, "P.png"))
    (x, y), mapa_u = RepresentacaoEscoamento.mapa_de_cor(Problema, u[:, 0], ordem=2, resolucao=None, x_grade=x, y_grade=y, local_grade=localizacao, titulo=u"Velocidade horizontal", path_salvar=os.path.join(nome_diretorio, "U.png"))
    (x, y), mapa_u = RepresentacaoEscoamento.mapa_de_cor(Problema, u[:, 1], ordem=2, resolucao=None, x_grade=x, y_grade=y, local_grade=localizacao, titulo=u"Velocidade vertical", path_salvar=os.path.join(nome_diretorio, "V.png"))
    iniciais = np.linspace([Problema.x_min, Problema.y_min + 0.1], [Problema.x_min, Problema.y_max - 0.1], 20)
    linhas = RepresentacaoEscoamento.linhas_de_corrente(Problema, u, pontos_iniciais=iniciais, resolucao=tamanho / 10, path_salvar=os.path.join(nome_diretorio, "Correntes.png"))
    plt.show(block=False)
    return forca, x, coeficientes


def teste_poiseuille(tamanho=0.1, p0=0, Re=1., dt=0.05, T=3., executa=True, formulacao="F"):
    if executa:
        nome_malha, tag_fis = Malha.malha_retangular("teste 5-1", tamanho, (5, 1))
        Problema = ElementosFinitos.FEA(nome_malha, tag_fis)
        ux_dirichlet = [
            (Problema.nos_cont["esquerda"], lambda x: 1.),
            (Problema.nos_cont["superior"], lambda x: 0.),
            (Problema.nos_cont["inferior"], lambda x: 0.),
        ]
        uy_dirichlet = [
            (Problema.nos_cont["esquerda"], lambda x: 0.),
            (Problema.nos_cont["superior"], lambda x: 0.),
            (Problema.nos_cont["inferior"], lambda x: 0.),
        ]
        p_dirichlet = [(Problema.nos_cont_o1["direita"], lambda x: p0),
                       # (Problema.nos_cont_o1["esquerda"], lambda x: 1.),
                       ]
        regiao_analitica = lambda x: np.logical_and(x[:, 0] >= 2, x[:, 0] < 4.9)
        solucao_analitica = lambda x: np.vstack([6 * x[:, 1] * (1 - x[:, 1]), np.zeros(len(x))]).T
        resultados = Problema.escoamento_IPCS_NS(ux_dirichlet=ux_dirichlet, uy_dirichlet=uy_dirichlet, p_dirichlet=p_dirichlet, T=T, dt=dt, Re=Re, solucao_analitica=solucao_analitica, regiao_analitica=regiao_analitica, formulacao=formulacao)
        # with open(os.path.join("Picles", "resultados Poiseuille.pkl"), "wb") as f :
        #     pickle.dump((Problema, resultados), f)
        salvar_resultados(nome_malha, tag_fis, resultados, os.path.join("Saida", "Poiseuille", f"Poiseuille h={tamanho} dt={dt} Re={Re} {formulacao}.zip"))
        RepresentacaoEscoamento.plotar_perfis(Problema, resultados, T)
        RepresentacaoEscoamento.plotar_momento(Problema, resultados, T)
        u = resultados[T]["u"]
        p = resultados[T]["p"]
    else:
        # with open(os.path.join("Picles", "resultados Poiseuille.pkl"), "rb") as f :
        #     Problema, resultados = pickle.load(f)
        Problema, u, p, nome_malha = carregar_resultados(os.path.join("Saida", "Poiseuille", f"Poiseuille h={tamanho} dt={dt} Re={Re} {formulacao}.zip"))

    t0 = time.process_time()
    t1 = time.process_time()
    print(f"Perfis plotados em {t1 - t0:.4f} s")
    t2 = time.process_time()
    resolucao = tamanho / 3
    x = np.arange(Problema.x_min, Problema.x_max + resolucao, resolucao)
    y = np.arange(Problema.y_min, Problema.y_max + resolucao, resolucao)
    localizacao = Problema.localiza_grade(x, y)
    path_salvar = os.path.join("Saida", "Poiseuille", f"Poiseuille h={tamanho} dt={dt} Re={Re} T={T} {formulacao}")
    (x1, y1), mapa_u = RepresentacaoEscoamento.mapa_de_cor(Problema, u[:, 0], ordem=2, resolucao=None, x_grade=x, y_grade=y, local_grade=localizacao, titulo=u"Velocidade horizontal", path_salvar=path_salvar + " u.png")
    (x1, y1), mapa_v = RepresentacaoEscoamento.mapa_de_cor(Problema, u[:, 1], ordem=2, resolucao=None, x_grade=x, y_grade=y, local_grade=localizacao, titulo=u"Velocidade vertical", path_salvar=path_salvar + " v.png")
    (x1, y1), mapa_p = RepresentacaoEscoamento.mapa_de_cor(Problema, p, ordem=1, resolucao=None, x_grade=x, y_grade=y, local_grade=localizacao, titulo=u"Pressão", path_salvar=path_salvar + " p.png")
    t3 = time.process_time()
    print(f"Mapa de cor calculado em {t3 - t2:.4f} s")
    t4 = time.process_time()
    pontos_inicio = np.linspace([0., 0.1], [0., 0.9], 9)
    correntes = RepresentacaoEscoamento.linhas_de_corrente(Problema, u, pontos_iniciais=pontos_inicio, resolucao=resolucao, path_salvar=path_salvar + " correntes.png")
    t5 = time.process_time()
    print(f"Linhas de corrente calculadas em {t5 - t4:.4f} s")

    # plotar_momento(Problema, resultados, 3)
    plt.show(block=False)


def teste_cavidade(tamanho=0.01, p0=0, dt=0.01, T=3, Re=1, executa=True, formulacao="F", debug=False):
    nome_diretorio = os.path.join("Saida", "Cavidade", f"Cavidade h={tamanho} dt={dt} Re={Re} T={T} {formulacao}")
    if executa:
        nome_malha, tag_fis = Malha.malha_quadrada("cavidade", tamanho)
        Problema = ElementosFinitos.FEA(nome_malha, tag_fis)
        ux_dirichlet = [
            (Problema.nos_cont["esquerda"], lambda x: 0.),
            (Problema.nos_cont["superior"], lambda x: 1.),
            (Problema.nos_cont["inferior"], lambda x: 0.),
            (Problema.nos_cont["direita"], lambda x: 0.),
        ]
        uy_dirichlet = [
            (Problema.nos_cont["esquerda"], lambda x: 0.),
            (Problema.nos_cont["superior"], lambda x: 0.),
            (Problema.nos_cont["inferior"], lambda x: 0.),
            (Problema.nos_cont["direita"], lambda x: 0.),
        ]
        vertice_pressao = np.where(np.logical_and(Problema.x_nos[:, 0] == 1, Problema.x_nos[:, 1] == 0))[0]
        p_dirichlet = [(vertice_pressao, lambda x: p0), ]
        if debug:
            u0 = 1
        else:
            u0 = 0
        resultados = Problema.escoamento_IPCS_NS(ux_dirichlet=ux_dirichlet, uy_dirichlet=uy_dirichlet, p_dirichlet=p_dirichlet, T=T, dt=dt, Re=Re, formulacao=formulacao, debug=debug, u0=u0)
        nome_diretorio = cria_diretorio(nome_diretorio)
        nome_arquivo = os.path.join(nome_diretorio, f" cavidade h={tamanho} dt={dt} Re={Re} T={T} {formulacao}.zip")
        salvar_resultados(nome_malha, tag_fis, resultados, nome_arquivo)
        RepresentacaoEscoamento.plotar_momento(Problema, resultados, T)
        u = resultados[T]["u"]
        p = resultados[T]["p"]
        ###Avaliando como as grandezas variaram de m passo para outro
        t_ant = np.sort(list(resultados.keys()))[-2]
        u_ant = resultados[t_ant]["u"]
        p_ant = resultados[t_ant]["p"]
        print(f"Derivada absoluta media da velocidade : {np.mean(np.abs(u - u_ant)) / dt}")
        print(f"Derivada absoluta media da pressao : {np.mean(np.abs(p - p_ant)) / dt}")
        # with open(os.path.join("Picles", f"resultados cavidade.pkl h={tamanho} dt={dt} Re={Re} {formulacao}"), "wb") as f :
        #     pickle.dump((Problema, resultados), f)
    else:
        nome_arquivo = os.path.join(nome_diretorio, f" cavidade h={tamanho} dt={dt} Re={Re} T={T} {formulacao}.zip")
        Problema, u, p, nome_malha = carregar_resultados(nome_arquivo)
        # with open(os.path.join("Picles", f"resultados cavidade.pkl h={tamanho} dt={dt} Re={Re} {formulacao}"), "rb") as f :
        #     Problema, resultados = pickle.load(f)
    resolucao = tamanho / 3
    x = np.arange(Problema.x_min, Problema.x_max + resolucao, resolucao)
    y = np.arange(Problema.y_min, Problema.y_max + resolucao, resolucao)
    localizacao = Problema.localiza_grade(x, y)

    (x1, y1), mapa_u = RepresentacaoEscoamento.mapa_de_cor(Problema, u[:, 0], ordem=2, resolucao=None, x_grade=x, y_grade=y, local_grade=localizacao, titulo=u"Velocidade horizontal", path_salvar=os.path.join(nome_diretorio, "U.png"))
    (x1, y1), mapa_v = RepresentacaoEscoamento.mapa_de_cor(Problema, u[:, 1], ordem=2, resolucao=None, x_grade=x, y_grade=y, local_grade=localizacao, titulo=u"Velocidade vertical", path_salvar=os.path.join(nome_diretorio, "V.png"))
    (x1, y1), mapa_p = RepresentacaoEscoamento.mapa_de_cor(Problema, p, ordem=1, resolucao=None, x_grade=x, y_grade=y, local_grade=localizacao, titulo=u"Pressão", path_salvar=os.path.join(nome_diretorio, "P.png"))
    op_conveccao_x = lambda x, l, u: Problema.conveccao_localizado(x, u, u[:, 0], l, ordem=2)
    op_conveccao_y = lambda x, l, u: Problema.conveccao_localizado(x, u, u[:, 1], l, ordem=2)
    (x1, y1), mapa_conveccao_x = RepresentacaoEscoamento.mapa_de_cor(Problema, u, ordem=2, resolucao=None, x_grade=x, y_grade=y, local_grade=localizacao, titulo=u"Convecção da velocidade horizontal", path_salvar=os.path.join(nome_diretorio, "Conveccao U.png"), operacao=op_conveccao_x)
    (x1, y1), mapa_conveccao_y = RepresentacaoEscoamento.mapa_de_cor(Problema, u, ordem=2, resolucao=None, x_grade=x, y_grade=y, local_grade=localizacao, titulo=u"Convecção da velocidade vertical", path_salvar=os.path.join(nome_diretorio, "Conveccao V.png"), operacao=op_conveccao_y)

    iniciais = np.linspace([0.5, 0.1], [0.5, 0.9], 10)
    ##Plotando as linhas de corrente para um lado e para o outro
    fig, eixo = plt.subplots()
    correntes = RepresentacaoEscoamento.linhas_de_corrente(Problema, u, pontos_iniciais=iniciais, resolucao=resolucao, eixo=eixo)
    correntes_inversas = RepresentacaoEscoamento.linhas_de_corrente(Problema, -u, pontos_iniciais=iniciais, resolucao=tamanho / 10, eixo=eixo)
    plt.savefig(os.path.join(nome_diretorio, "Correntes.png"), dpi=300, bbox_inches="tight")


def compara_cavidade_ref(h, dt, T, formulacao="A", plota=True):
    '''Compara os resultados de um caso estacionario com os resultados de referencia.
    Recebe como entrada um caso ja devidamente calculado'''
    arquivo_referencia = "Entrada/Referencia/Cavidade solucao referencia.txt"
    valores_Re = (0.01, 10, 100, 400, 1000, "inf")
    dframe_erros = pd.DataFrame(index=valores_Re, columns=["u_med", "u_rms", "u_max", "v_med", "v_rms", "v_max"], dtype=np.float64)
    roda_cores = {0.01: "b", 10: "g", 100: "r", 400: "c", 1000: "m", "inf": "y"}
    path_salvar = os.path.join("Saida", "Cavidade", "Comparacao", f"Comparacao h={h} dt={dt} T={T} {formulacao}")
    if plota:
        fig_u, eixo_u = plt.subplots()
        fig_v, eixo_v = plt.subplots()
        eixo_u.set_title(f"Velocidade horizontal em x=0.5")
        eixo_v.set_title(f"Velocidade vertical em y=0.5")
        eixo_u.set_xlabel("u")
        eixo_v.set_ylabel("v")
        eixo_u.set_ylabel("y")
        eixo_v.set_xlabel("x")
        eixo_u.set_xlim(-1, 1)
        eixo_u.set_ylim(0, 1)
        eixo_v.set_xlim(0, 1)
        eixo_v.set_ylim(-1, 1)
    for Re in valores_Re:
        try:
            arquivo_resultados = os.path.join("Saida", "Cavidade", f"Cavidade h={h} dt={dt} Re={Re} T={T} {formulacao}", f" cavidade h={h} dt={dt} Re={Re} T={T} {formulacao}.zip")
            Problema, u, p, nome_malha = carregar_resultados(arquivo_resultados)
            dframe_ref = pd.read_csv(arquivo_referencia)
            if Re == "inf":
                vel_ref = dframe_ref.loc[11:, f"Re=0.01"]
            else:
                vel_ref = dframe_ref.loc[11:, f"Re={Re}"]
            u_ref, v_ref = vel_ref.values.reshape((2, len(vel_ref) // 2))
            pontos_u = np.linspace([0.5, 0.0625], [0.5, 0.9375], 15)
            pontos_v = np.linspace([0.0625, 0.5], [0.9375, 0.5], 15)
            u_calc = np.array([Problema.interpola(ponto, u, ordem=2) for ponto in pontos_u])[:, 0]
            v_calc = np.array([Problema.interpola(ponto, u, ordem=2) for ponto in pontos_v])[:, 1]
            erro_u = u_calc - u_ref
            u_med = np.average(erro_u)
            u_rms = np.sqrt(np.average(erro_u ** 2))
            u_max = np.max(np.abs(erro_u))
            erro_v = v_calc - v_ref
            v_med = np.average(erro_v)
            v_rms = np.sqrt(np.average(erro_v ** 2))
            v_max = np.max(np.abs(erro_v))
            dframe_erros.loc[Re] = u_med, u_rms, u_max, v_med, v_rms, v_max
            if plota:
                eixo_u.scatter(u_ref, pontos_u[:, 1], marker='*', color=roda_cores[Re])
                pontos_u2 = np.linspace([0.5, 0.0625], [0.5, 0.9375], 301)
                u_calc2 = np.array([Problema.interpola(ponto, u, ordem=2) for ponto in pontos_u2])[:, 0]
                eixo_u.plot(u_calc2, pontos_u2[:, 1], color=roda_cores[Re], label=f"Re={Re}")
                eixo_v.scatter(pontos_v[:, 0], v_ref, marker='*', color=roda_cores[Re])
                pontos_v2 = np.linspace([0.0625, 0.5], [0.9375, 0.5], 301)
                v_calc2 = np.array([Problema.interpola(ponto, u, ordem=2) for ponto in pontos_v2])[:, 1]
                eixo_v.plot(pontos_v2[:, 0], v_calc2, color=roda_cores[Re], label=f"Re={Re}")
        except  FileNotFoundError:
            pass
    if plota:
        eixo_u.legend()
        eixo_v.legend()
        fig_u.savefig(path_salvar + " u.png", dpi=300, bbox_inches="tight")
        fig_v.savefig(path_salvar + " v.png", dpi=300, bbox_inches="tight")
    dframe_erros.to_csv(path_salvar + " erros.csv")
    return dframe_erros





def validacao_npontos_af(n_min=5, n_max=500, Re=1, dt=0.01, T=30, h=0.5, formulacao="F", aerofolio=AerofolioFino.NACA4412, resolucao=0.05, executa=True, plota=True):
    '''Calcula a solucao do aerofolio para diferentes tamanhos de malha e compara entre si os resultados'''
    # Criar uma grade de pontos que so cubra a regiao em torno do aerofolio, com resolucao fina
    x_min, x_max, y_min, y_max = -1, 2, -1.5, 1.5
    x_grade = np.arange(x_min, x_max, resolucao)
    y_grade = np.arange(y_min, y_max, resolucao)
    valores_n = np.logspace(np.log10(n_min), np.log10(n_max), 10)
    coefs_arrasto = np.zeros(len(valores_n))
    coefs_sustentacao = np.zeros(len(valores_n))
    coefs_momento = np.zeros(len(valores_n))
    tempos=np.zeros(len(valores_n))
    vetores_u = []
    vetores_p = []
    shape_mapa=(len(valores_n),len(x_grade),len(y_grade))
    mapas_u = np.zeros(shape_mapa, dtype=np.float64)
    mapas_v = np.zeros(shape_mapa, dtype=np.float64)
    mapas_p = np.zeros(shape_mapa, dtype=np.float64)

    if executa: nome_diretorio=cria_diretorio(os.path.join("Saida", "Aerofolio", f"Validacao n pontos {aerofolio.nome} h={h} Re={Re} dt={dt} T={T} {formulacao}"))
    else: nome_diretorio= os.path.join("Saida", "Aerofolio", f"Validacao n pontos {aerofolio.nome} h={h} Re={Re} dt={dt} T={T} {formulacao}")
    for i, n in enumerate(valores_n):
        nome_malha, tag_fis = Malha.malha_aerofolio(aerofolio, n_pontos_contorno=int(n), tamanho=0.01)
        Problema = ElementosFinitos.FEA(nome_malha, tag_fis)
        ux_dirichlet = [
            (Problema.nos_cont["esquerda"], lambda x: 1.),
            (Problema.nos_cont["superior"], lambda x: 1.),
            (Problema.nos_cont["inferior"], lambda x: 1.),
            (Problema.nos_cont["af"], lambda x: 0.),
        ]
        uy_dirichlet = [
            (Problema.nos_cont["esquerda"], lambda x: 0.),
            (Problema.nos_cont["superior"], lambda x: 0.),
            (Problema.nos_cont["inferior"], lambda x: 0.),
            (Problema.nos_cont["af"], lambda x: 0.),
        ]
        p_dirichlet = [(Problema.nos_cont_o1["direita"], lambda x: 0), ]

        if executa:
            t1=time.process_time()
            resultados = Problema.escoamento_IPCS_NS(ux_dirichlet=ux_dirichlet, uy_dirichlet=uy_dirichlet, p_dirichlet=p_dirichlet, T=T, dt=dt, Re=Re, u0=1., formulacao=formulacao)
            t2=time.process_time()
            u = resultados[T]["u"]
            p = resultados[T]["p"]
            salvar_resultados(nome_malha, tag_fis, resultados, os.path.join(nome_diretorio, f"n={n}.zip"))
            tempos[i]=t2-t1
        else:
            Problema, u, p, nome_malha = carregar_resultados(os.path.join(nome_diretorio, f"n={n}.zip"))
            tempos[i]=0
        vetores_u.append(u)
        vetores_p.append(p)
        localizacao = Problema.localiza_grade(x_grade, y_grade)
        (x1, y1), mapa_u = RepresentacaoEscoamento.mapa_de_cor(Problema, u[:, 0], ordem=2, resolucao=None, x_grade=x_grade, y_grade=y_grade, local_grade=localizacao, plota=False)
        (x1, y1), mapa_v = RepresentacaoEscoamento.mapa_de_cor(Problema, u[:, 1], ordem=2, resolucao=None, x_grade=x_grade, y_grade=y_grade, local_grade=localizacao, plota=False)
        (x1, y1), mapa_p = RepresentacaoEscoamento.mapa_de_cor(Problema, p, ordem=1, resolucao=None, x_grade=x_grade, y_grade=y_grade, local_grade=localizacao, plota=False)
        mapas_u[i]=mapa_u
        mapas_v[i]=mapa_v
        mapas_p[i]=mapa_p
        c_d,c_l,c_M, outros=ElementosFinitos.coeficientes_aerodinamicos(Problema, u, p, Re=Re, x_centro=np.array([0.25,0.]))
        coefs_arrasto[i]=c_d
        coefs_sustentacao[i]=c_l
        coefs_momento[i]=c_M
    with open(os.path.join(nome_diretorio, "resultados.pkl"), "wb") as f:
        pickle.dump((valores_n, coefs_arrasto, coefs_sustentacao, coefs_momento, vetores_u, vetores_p, mapas_u, mapas_v, mapas_p, tempos), f)
    erros_p=mapas_p-mapas_p[-1]
    eqm_p=np.sqrt(np.average(erros_p**2,axis=(1,2)))
    e_max_p = np.max(np.abs(erros_p), axis=(1, 2))
    vies_p=np.average(erros_p,axis=(1,2))
    erros_u=mapas_u-mapas_u[-1]
    eqm_u=np.sqrt(np.average(erros_u**2,axis=(1,2)))
    e_max_u = np.max(np.abs(erros_u), axis=(1, 2))
    vies_u=np.average(erros_u,axis=(1,2))
    erros_v=mapas_v-mapas_v[-1]
    eqm_v=np.sqrt(np.average(erros_v**2,axis=(1,2)))
    e_max_v = np.max(np.abs(erros_v), axis=(1, 2))
    vies_v=np.average(erros_v,axis=(1,2))
    with open(os.path.join(nome_diretorio, "erros.pkl"), "wb") as f:
        pickle.dump((eqm_p, e_max_p, vies_p, eqm_u, e_max_u, vies_u, eqm_v, e_max_v, vies_v), f)
    cols=["c_d","c_l","c_M","eqm_p", "e_max_p", "vies_p", "eqm_u", "e_max_u", "vies_u", "eqm_v", "e_max_v", "vies_v","t"]
    indice=valores_n
    array_dataframe=np.array([c_d,c_l,c_M,eqm_p, e_max_p, vies_p, eqm_u, e_max_u, vies_u, eqm_v, e_max_v, vies_v, tempos]).T
    dframe_erros=pd.DataFrame(index=indice,columns=cols,dtype=np.float64, data=array_dataframe)
    dframe_erros.to_csv(os.path.join(nome_diretorio, "Resultados.csv"))
    if plota:
        RepresentacaoEscoamento.plotar_dataframe_analise_n(dframe_erros, path_salvar=nome_diretorio)
    return







if __name__ == "__main__":
    # teste_poiseuille(tamanho=0.1, p0=0,  executa=True, dt=0.01, T=2, Re=1, formulacao="A")
    # teste_cavidade(tamanho=0.02,  dt=0.01, T=30, Re=10, formulacao="E", executa=True, debug=False)
    # teste_cavidade(tamanho=0.05, dt=0.01,T=20,Re=0.01,executa=False,formulacao="E")
    # plt.show(block=True)
    # teste_cavidade(tamanho=0.01, p0=0,  executa=True, dt=0.01, T=1.1, Re=1, formulacao="A")
    # teste_cavidade(tamanho=0.05, dt=0.01, T=5, Re=1, executa=True, formulacao="A")
    # teste_cavidade(tamanho=0.05, dt=0.01, T=5, Re=1, executa=False, formulacao="A")
    # plt.show(block=True)
    # teste_poiseuille(0.1, 0, 1, 0.01, 2, True, "E")
    # plt.show(block=True)
    # teste_cavidade(tamanho=0.1, p0=0, executa=True, dt=0.01, T=30, Re=1, formulacao="F", debug=False)
    # for Re in (0.1, 1, 5, 10):
    #     teste_forca(n=500, tamanho=0.5, debug=False, executa=True, formulacao="F", T=20, dt=0.01, Re=Re)
    #     plt.close("all")
    # for Re in (0.1, 1, 5, 10):
    #     teste_forca(n=500, tamanho=0.5, debug=False, executa=False, formulacao="F", T=20, dt=0.01, Re=Re)
    #     plt.close("all")
    validacao_npontos_af(n_min=5, n_max=50, Re=1, dt=0.01, T=30, h=1., formulacao="F", aerofolio=AerofolioFino.NACA4412, resolucao=0.05, executa=True, plota=True)
    plt.show(block=True)
    # for Re in (100,400,0.01,10,1000):
    #     teste_cavidade(tamanho=0.02, p0=0, executa=True, dt=0.01, T=30, Re=Re, formulacao="F", debug=False)
    #     plt.close("all")
    # erros = compara_cavidade_ref(h=0.02, dt=0.01, T=30, formulacao="F", plota=True)
    # plt.show(block=True)
    # # for Re in (0.01,10,100,400,1000):
    # #     teste_cavidade(tamanho=0.03, p0=0, executa=True, dt=0.01, T=10, Re=Re, formulacao="A", debug=True)
    # #     plt.close("all")
    # erros=compara_cavidade_ref(h=0.05, dt=0.01, T=20, formulacao="E", plota=True)
    # plt.show(block=True)
    # for Re in (400,1000):
    #     teste_cavidade(tamanho=0.01, p0=0, executa=True, dt=0.01, T=1, Re=Re, formulacao="A")
    #     plt.close("all")
    # # teste_forca(n=50, tamanho=0.3, debug=False, executa=True)
    # plt.show(block=True)
    # plt.close("all")
    # # teste_forca(n=50, tamanho=0.3, debug=False, executa=False)
    # plt.show(block=False)
    # plt.show()

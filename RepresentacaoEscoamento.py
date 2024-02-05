import ElementosFinitos
from Definicoes import plt, np, os


def plotar_momento(Problema, resultados, t, plotar_auxiliares=True) :
    plt.figure()
    plt.suptitle(f"Velocidade horizontal - ux   t= {t} s")
    plt.triplot(Problema.x_nos[:, 0], Problema.x_nos[:, 1], Problema.elementos_o1, alpha=0.5)
    plt.scatter(Problema.x_nos[:, 0], Problema.x_nos[:, 1], c=resultados[t]["u"][:, 0])
    plt.colorbar()
    plt.figure()
    plt.suptitle(f"Velocidade vertical - uy   t= {t} s")
    plt.triplot(Problema.x_nos[:, 0], Problema.x_nos[:, 1], Problema.elementos_o1, alpha=0.5)
    plt.scatter(Problema.x_nos[:, 0], Problema.x_nos[:, 1], c=resultados[t]["u"][:, 1])
    plt.colorbar()
    if plotar_auxiliares :
        plt.figure()
        plt.suptitle(f"Velocidade horizontal - u*x   t= {t} s")
        plt.triplot(Problema.x_nos[:, 0], Problema.x_nos[:, 1], Problema.elementos_o1, alpha=0.5)
        plt.scatter(Problema.x_nos[:, 0], Problema.x_nos[:, 1], c=resultados[t]["u*"][:, 0])
        plt.colorbar()
        plt.figure()
        plt.suptitle(f"Pressao ficticia - p*   t= {t} s")
        plt.triplot(Problema.x_nos[:, 0], Problema.x_nos[:, 1], Problema.elementos_o1, alpha=0.5)
        plt.scatter(Problema.x_nos_o1[:, 0], Problema.x_nos_o1[:, 1], c=resultados[t]["p*"])
        plt.colorbar()
    plt.figure()
    plt.suptitle(f"Pressao - p   t= {t} s")
    plt.triplot(Problema.x_nos[:, 0], Problema.x_nos[:, 1], Problema.elementos_o1, alpha=0.5)
    plt.scatter(Problema.x_nos_o1[:, 0], Problema.x_nos_o1[:, 1], c=resultados[t]["p"])
    plt.colorbar()


def plotar_perfil(Problema, u, x=4, eixo=None, ordem=2) :
    r = np.linspace([x, 0, 0], [x, 1, 0], 1001)
    u_interp = np.array([Problema.interpola(p, u, ordem=ordem) for p in r])
    ux = u_interp[:, 0]
    uy = u_interp[:, 1]
    if eixo is None :
        plt.figure()
        plt.suptitle(f"Perfil de velocidade horizontal - x={x}")
    plt.plot(ux, r[:, 1], label=f"ux({x},y)")


def plotar_perfis(Problema, u, lim_x=(0, 5), referencia=None) :
    fig, eixo = plt.subplots()
    for x in np.arange(lim_x[0], lim_x[1] + 0.000001, 2) :
        plotar_perfil(Problema, u, x, eixo)
    if not referencia is None :
        r, u_ref = referencia
        eixo.scatter(u_ref[:, 0], r[:, 1], marker="*", color="black", label="Referencia")
    eixo.set_xlabel("u")
    eixo.set_ylabel("y")
    eixo.legend()
    return


def plotar_pressao(Problema, p, lim_x=(0, 10), y_med=0.5, referencia=None) :
    '''Plota a pressao na linha media de um escoamento de poiseuille'''
    fig, eixo = plt.subplots()
    x_linha = np.linspace(lim_x[0], lim_x[1], 1001)
    p_interp = np.array([Problema.interpola(np.array([x, y_med]), p) for x in x_linha])
    eixo.plot(x_linha, p_interp, label=u"Pressão")
    if not referencia is None :
        r, p_ref = referencia
        x_ref = r[:, 0]
        eixo.scatter(x_ref, p_ref, marker="*", color="black", label="Referencia")
    eixo.set_xlabel("x")
    eixo.set_ylabel("p")
    eixo.legend()
    return


def mapa_de_cor(Problema, variavel, ordem, resolucao=0.01, areas_excluidas=[], x_grade=None, y_grade=None, local_grade=None, plota=True, titulo="", path_salvar=None, aspecto=(6, 4), operacao=None) :
    '''
    Interpola a variavel em uma grade estruturada NxN e plota o mapa de cor.
    :param Problema:
    :param variavel: ux, uy ou p
    :param ordem: ordem da funcao de interpolacao da variavel
    :param resolucao: resolucao espacial, que se supoe ser igual para x e y
    :param areas_excluidas: lista de funcoes que retornam True para pontos que devem ser excluidos do mapa de cor
    :param operacao: operacao a ser aplicada no mapa de cor, em vez de simplesmente interpolar (e.g. calcular a conveccao)
    :param x_grade: np.array (N,) coordenada x da grade estruturada em que se calcula o mapa de cor para o mapa de cor
    :return:
    '''

    if not (x_grade is None or y_grade is None or local_grade is None) :
        assert local_grade.shape == (len(x_grade), len(y_grade))
        if operacao is None :
            operacao = lambda r, l, var : Problema.interpola_localizado(r, var, l, ordem=ordem)
        x, y = x_grade, y_grade
        pontos = np.dstack(np.meshgrid(x_grade, y_grade, indexing="ij"))
        mapa = np.zeros((len(x_grade), len(y_grade)), dtype=np.float64)
        for i in range(len(x_grade)) :
            for j in range(len(y_grade)) :
                r = pontos[i, j]
                if not any([f(r) for f in areas_excluidas]) :
                    mapa[i, j] = operacao(r, local_grade[i, j], variavel)
                else :
                    mapa[i, j] = np.nan
    else :
        x = np.arange(Problema.x_min, Problema.x_max + resolucao, resolucao)
        y = np.arange(Problema.y_min, Problema.y_max + resolucao, resolucao)
        mapa = np.zeros((len(x), len(y)), dtype=np.float64)
        for i in range(len(x)) :
            for j in range(len(y)) :
                r = np.array([x[i], y[j]])
                if not any([f(r) for f in areas_excluidas]) :
                    try :
                        mapa[i, j] = Problema.interpola(r, variavel, ordem=ordem)
                    except ElementosFinitos.ElementoNaoEncontrado :
                        mapa[i, j] = np.nan
                else :
                    mapa[i, j] = np.nan
    if plota :
        fig, eixo = plt.subplots()
        plt.title(titulo)
        plot_mapa = plt.pcolormesh(x, y, mapa.T, cmap="turbo")
        # plt.triplot(Problema.x_nos[:, 0], Problema.x_nos[:, 1], Problema.elementos_o1, alpha=0.1, color="gray")
        fig.set_size_inches(aspecto)
        eixo.set_aspect("equal")
        eixocbar = fig.add_axes([eixo.get_position().x1 + 0.01, eixo.get_position().y0, 0.02, eixo.get_position().height])
        plt.colorbar(plot_mapa, cax=eixocbar)
    if not path_salvar is None :
        plt.savefig(path_salvar, dpi=300, bbox_inches="tight")
    return (x, y), mapa


def linhas_de_corrente(Problema, u, pontos_iniciais, resolucao=0.01, areas_excluidas=[], plota=True, eixo=None, path_salvar=None) :
    '''
    :param Problema:
    :param u: vetor da velocidade nos nos da malha
    :param pontos_iniciais: lista de pontos iniciais para as linhas de corrente
    :param resolucao: resolucao espacial, que se supoe ser igual para x e y
    :param areas_excluidas: lista de funcoes que retornam True para pontos que devem ser excluidos do mapa de cor
    :return:
    '''
    L = max(Problema.x_max - Problema.x_min, Problema.y_max - Problema.y_min)
    areas_excluidas = areas_excluidas + [lambda p : p[0] < Problema.x_min, lambda p : p[0] > Problema.x_max, lambda p : p[1] < Problema.y_min, lambda p : p[1] > Problema.y_max]
    linhas = []
    for inicio in pontos_iniciais :
        linhas.append([])
        p = inicio * 1
        linhas[-1].append(p * 1)
        c = 0
        cmax = L / resolucao * 2
        while not any([f(p) for f in areas_excluidas]) :
            c += 1
            try :
                vel = Problema.interpola(p, u, ordem=2)
            except ElementosFinitos.ElementoNaoEncontrado :
                break
            passo = vel / np.linalg.norm(vel) * resolucao
            p += passo
            linhas[-1].append(p * 1)
            if np.isclose(p, inicio, atol=resolucao / 2).all() :
                break
            if c >= cmax :
                break
        linhas[-1] = np.array(linhas[-1], dtype=np.float64)
    if plota :
        if eixo is None :
            fig, eixo = plt.subplots()
        fig = eixo.get_figure()
        eixo.set_title("Linhas de corrente")
        for linha in linhas :
            eixo.plot(linha[:, 0], linha[:, 1], color="black")
        eixo.set_xlim(Problema.x_min, Problema.x_max)
        eixo.set_ylim(Problema.y_min, Problema.y_max)
        eixo.set_aspect("equal")
        if not path_salvar is None :
            plt.savefig(path_salvar, dpi=300, bbox_inches="tight")

    return linhas


def plotar_dataframe_analise(dataframe, parametro, path_salvar=None) :
    fig1, eixo1 = plt.subplots()
    plt.suptitle(u"Variação dos coeficientes dinâmicos")
    eixo1.plot(dataframe["c_l"], label="c_L")
    eixo1.plot(dataframe["c_d"], label="c_D")
    eixo1.plot(dataframe["c_M"], label="c_M")
    eixo1.set_xlabel(parametro)
    eixo1.set_ylabel("Coeficiente adimensional")
    eixo1.legend()
    if not path_salvar is None :
        plt.savefig(os.path.join(path_salvar, "Coeficientes.png"), dpi=300, bbox_inches="tight")
    fig2, eixo2 = plt.subplots()
    plt.suptitle(u"Erro da velocidade vertical")
    eixo2.set_xlabel(parametro)
    eixo2.set_ylabel("Erro")
    eixo2.plot(dataframe["eqm_v"], label=u"Erro quadrático médio")
    eixo2.plot(dataframe["e_max_v"], label=u"Erro máximo")
    eixo2.plot(dataframe["vies_v"], label=u"Erro médio (viés)")
    eixo2.legend()
    if not path_salvar is None :
        plt.savefig(os.path.join(path_salvar, "Erro V.png"), dpi=300, bbox_inches="tight")
    fig3, eixo3 = plt.subplots()
    plt.suptitle(u"Erro da velocidade horizontal")
    eixo3.set_xlabel(parametro)
    eixo3.set_ylabel("Erro")
    eixo3.plot(dataframe["eqm_u"], label=u"Erro quadrático médio")
    eixo3.plot(dataframe["e_max_u"], label=u"Erro máximo")
    eixo3.plot(dataframe["vies_u"], label=u"Erro médio (viés)")
    eixo3.legend()
    if not path_salvar is None :
        plt.savefig(os.path.join(path_salvar, "Erro U.png"), dpi=300, bbox_inches="tight")
    fig4, eixo4 = plt.subplots()
    plt.suptitle(u"Erro da pressão")
    eixo4.set_xlabel(parametro)
    eixo4.set_ylabel("Erro")
    eixo4.plot(dataframe["eqm_p"], label=u"Erro quadrático médio")
    eixo4.plot(dataframe["e_max_p"], label=u"Erro máximo")
    eixo4.plot(dataframe["vies_p"], label=u"Erro médio (viés)")
    eixo4.legend()
    if not path_salvar is None :
        plt.savefig(os.path.join(path_salvar, "Erro P.png"), dpi=300, bbox_inches="tight")
    fig4, eixo4 = plt.subplots()
    plt.suptitle(u"Tempo de cálculo")
    eixo4.set_xlabel(parametro)
    eixo4.set_ylabel("Tempo [s]")
    eixo4.plot(dataframe["t"])
    if not path_salvar is None :
        plt.savefig(os.path.join(path_salvar, "Tempo.png"), dpi=300, bbox_inches="tight")
    return

def desenhar_aerofolio_svg(aerofolio, path_salvar):
    '''
    Desenha um aerofolio em um arquivo svg
    :param aerofolio:
    :param path_salvar:
    :return:
    '''
    fig, eixo = plt.subplots()
    eixo.grid(False)
    fig.set_size_inches(7.5, 5)
    eixo.set_xlim(-0.05, 1.05)
    eixo.set_ylim(-0.2, 0.2)
    eixo.set_aspect("equal")
    aerofolio.desenhar(eixo)
    eixo.set_axis_off()
    fig.savefig(path_salvar, bbox_inches="tight", format="svg", transparent=True)
    return


if __name__=="__main__":
    from AerofolioFino import AerofolioFinoNACA4
    af=AerofolioFinoNACA4((0.04, 0.40, 0.12), 15*np.pi/180, 100)
    desenhar_aerofolio_svg(af, os.path.join("Saida","Aerofolio Fino NACA4", "Figuras", af.nome + ".svg"))
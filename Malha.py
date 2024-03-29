from Definicoes import gmsh, np, plt
from Definicoes import os

geo = gmsh.model.geo  # definindo um alias para o modulo de geometria do gmsh
n_pontos_contorno_padrao = 100
tamanho_padrao = 0.1


def malha_quadrada(nome_modelo, tamanho, ordem=2) :
    '''Gera uma malha quadrada no gmsh'''
    return malha_retangular(nome_modelo, tamanho, (1, 1), ordem)

def malha_retangular(nome_modelo, tamanho, formato, ordem=2) :
    contornos = {"esquerda": 1, "direita": 2, "superior": 3, "inferior": 4}
    Lx,Ly=formato
    tag_fis = {}
    ##Inicializando o gmsh
    gmsh.initialize()
    gmsh.model.add(nome_modelo)  # adiciona um modelo
    gmsh.model.set_current(nome_modelo)  # define o modelo atual
    geo.addPoint(0, 0, 0, tamanho, tag=1)  # ponto inferior esquerdo
    geo.addPoint(0, Ly, 0, tamanho, tag=2)  # ponto superior esquerdo
    geo.addPoint(Lx, 0, 0, tamanho, tag=3)  # ponto inferior direito
    geo.addPoint(Lx, Ly, 0, tamanho, tag=4)  # ponto superior direito
    geo.addLine(2, 1, tag=contornos["esquerda"])  # linha esquerda
    geo.addLine(4, 2, tag=contornos["superior"])  # linha superior
    geo.addLine(3, 4, tag=contornos["direita"])  # linha direita
    geo.addLine(1, 3, tag=contornos["inferior"])  # linha inferior
    geo.add_curve_loop([contornos["direita"], contornos["superior"], contornos["esquerda"], contornos["inferior"]], tag=1)  # superficie externa
    geo.add_plane_surface([1], tag=1)  # superficie do escoamento
    tag_fis["esquerda"] = geo.add_physical_group(1, [contornos["esquerda"]])
    tag_fis["direita"] = geo.add_physical_group(1, [contornos["direita"]])
    tag_fis["superior"] = geo.add_physical_group(1, [contornos["superior"]])
    tag_fis["inferior"] = geo.add_physical_group(1, [contornos["inferior"]])
    tag_fis["escoamento"] = geo.add_physical_group(2, [1])
    ###Sincronizar as modificacoes geometricas e gerar a malha
    geo.synchronize()  # necessario!
    gmsh.option.set_number("Mesh.ElementOrder", ordem)  # Define a ordem dos elementos
    gmsh.model.mesh.generate(2)  # gera a malha
    nome_arquivo = os.path.join("Malha", f"{nome_modelo}.msh")
    gmsh.write(nome_arquivo)  # salva o arquivo da malha
    ##Encerrando o gmsh
    gmsh.finalize()
    return nome_arquivo, tag_fis

def malha_degrau(nome_modelo="Degrau", tamanho=0.2, tamanho_fino=None, S=1, L=20) :
    '''Gera uma malha correspondendo ao problema do degrau invertido'''
    contornos={"entrada":1,"topo_degrau":2,"parede_degrau":3,"inferior":4,"saida":5,"superior":6}
    tag_fis = {}
    ordem=2
    if tamanho_fino is None: tamanho_fino=tamanho
    ##Inicializando o gmsh
    gmsh.initialize()
    gmsh.model.add(nome_modelo)  # adiciona um modelo
    gmsh.model.set_current(nome_modelo)  # define o modelo atual
    geo.add_point(-1,0,0,tamanho, tag=1)  # ponto inferior da entrada
    geo.add_point(0,0,0,tamanho_fino, tag=2)  # vertice do degrau
    geo.add_point(0,-S,0,tamanho_fino, tag=3)  # base do degrau
    geo.add_point(L,-S,0,tamanho, tag=4)  # ponto inferior da saida
    geo.add_point(L,1,0,tamanho, tag=5)  # ponto superior da saida
    geo.add_point(-1,1,0,tamanho, tag=6)  # ponto superior da entrada
    geo.add_line(6,1,tag=contornos["entrada"])  # linha de entrada
    geo.add_line(1,2,tag=contornos["topo_degrau"])  # linha do topo do degrau
    geo.add_line(2,3,tag=contornos["parede_degrau"])  # linha da parede do degrau
    geo.add_line(3,4,tag=contornos["inferior"])  # linha inferior
    geo.add_line(4,5,tag=contornos["saida"])  # linha de saida
    geo.add_line(5,6,tag=contornos["superior"])  # linha superior
    geo.add_curve_loop([contornos["entrada"],contornos["topo_degrau"],contornos["parede_degrau"],contornos["inferior"],contornos["saida"],contornos["superior"]],tag=1)  # superficie externa
    geo.add_plane_surface([1],tag=1)  # superficie do escoamento
    tag_fis["entrada"]=geo.add_physical_group(1,[contornos["entrada"]])
    tag_fis["parede"]=geo.add_physical_group(1,[contornos["topo_degrau"],contornos["parede_degrau"],contornos["inferior"],contornos["superior"]])
    tag_fis["saida"]=geo.add_physical_group(1,[contornos["saida"]])
    tag_fis["escoamento"] = geo.add_physical_group(2, [1])
    ###Sincronizar as modificacoes geometricas e gerar a malha
    geo.synchronize()  # necessario!
    gmsh.option.set_number("Mesh.ElementOrder", ordem)  # Define a ordem dos elementos
    gmsh.model.mesh.generate(2)  # gera a malha
    nome_arquivo = os.path.join("Malha", f"{nome_modelo}.msh")
    gmsh.write(nome_arquivo)  # salva o arquivo da malha
    ##Encerrando o gmsh
    gmsh.finalize()
    return nome_arquivo, tag_fis


def malha_aerofolio(aerofolio, nome_modelo="modelo", n_pontos_contorno=n_pontos_contorno_padrao, ordem=2, tamanho=tamanho_padrao, folga=10, verbosidade=5) :
    '''Gera uma malha no gmsh correspondendo a regiao em torno do aerofolio'''
    contornos = {"esquerda" : 1, "direita" : 2, "superior" : 3, "inferior" : 4, }
    # n_pontos_contorno = 1000
    tag_fis = {}  # tags dos grupos fisicos
    # af_tamanho = 1 / n_pontos_contorno
    # tamanho = 10 * af_tamanho
    x_min,x_max=-folga,1+folga
    y_min,y_max=-folga,+folga
    ##Inicializando o gmsh
    gmsh.initialize()
    gmsh.option.set_number("General.Verbosity",verbosidade) ##Define o nivel de verbosidade do gmsh
    gmsh.model.add(nome_modelo)  # adiciona um modelo
    gmsh.model.set_current(nome_modelo)  # define o modelo atual
    geo.addPoint(x_min, y_min, 0, tamanho, tag=1)  # ponto inferior esquerdo
    geo.addPoint(x_min,y_max, 0, tamanho, tag=2)  # ponto superior esquerdo
    geo.addPoint(x_max, y_min, 0, tamanho, tag=3)  # ponto inferior direito
    geo.addPoint(x_max, y_max, 0, tamanho, tag=4)  # ponto superior direito
    geo.add_line(1, 2, tag=contornos["esquerda"])
    geo.add_line(3, 4, tag=contornos["direita"])
    geo.add_line(1, 3, tag=contornos["inferior"])
    geo.add_line(2, 4, tag=contornos["superior"])

    ###versao a vera com aerofolio
    tamanho_af=tamanho
    ponto_inicial = geo.add_point(aerofolio.x_med(0), aerofolio.y_med(0), 0, tamanho_af)
    ponto_final = geo.add_point(aerofolio.x_med(1), aerofolio.y_med(1), 0, tamanho_af)
    pontos_sup = [ponto_inicial, ]
    pontos_inf = [ponto_inicial, ]
    af_sup = []
    af_inf = []
    for i in range(1, n_pontos_contorno) :
        ##eta eh igual ao x da linha base do aerofolio
        eta = i / n_pontos_contorno
        pontos_sup.append(geo.add_point(aerofolio.x_sup(eta), aerofolio.y_sup(eta), 0, tamanho_af))
        pontos_inf.append(geo.add_point(aerofolio.x_inf(eta), aerofolio.y_inf(eta), 0, tamanho_af))
    pontos_sup.append(ponto_final) ##O ponto final eh o mesmo para as duas linhas
    pontos_inf.append(ponto_final)
    for i in range(n_pontos_contorno) :
        af_sup.append(geo.add_line(pontos_sup[i], pontos_sup[i + 1]))
        af_inf.append(geo.add_line(pontos_inf[i], pontos_inf[i + 1]))
    af_inf_inverso = [-item for item in af_inf[: :-1]]
    contornos["af_superior"] = af_sup
    contornos["af_inferior"] = af_inf

    ###Definindo as superficies para simulacao
    geo.add_curve_loop(af_sup + af_inf_inverso, tag=2)  # superficie do aerofolio
    geo.add_curve_loop([-1, 4, 2, -3], tag=1)  # superficie externa
    geo.add_plane_surface([1, 2], tag=1)  # superficie do escoamento

    ##Criando grupos fisicos correspondendo a cada elemento da simulacao
    tag_fis["af"] = geo.add_physical_group(1, af_sup + af_inf)
    tag_fis["esquerda"] = geo.add_physical_group(1, [contornos["esquerda"]])
    tag_fis["direita"] = geo.add_physical_group(1, [contornos["direita"]])
    tag_fis["superior"] = geo.add_physical_group(1, [contornos["superior"]])
    tag_fis["inferior"] = geo.add_physical_group(1, [contornos["inferior"]])
    tag_fis["escoamento"] = geo.add_physical_group(2, [1])  # grupo fisico 2d correspondendo a todo o escoamento

    ###Sincronizar as modificacoes geometricas e gerar a malha
    geo.synchronize()  # necessario!
    gmsh.option.set_number("Mesh.ElementOrder", ordem)  # Define a ordem dos elementos
    gmsh.model.mesh.generate(2)  # gera a malha
    nome_arquivo = os.path.join("Malha", f"{nome_modelo}.msh")
    gmsh.write(nome_arquivo)  # salva o arquivo da malha
    ##Encerrando o gmsh
    gmsh.finalize()
    return nome_arquivo, tag_fis


def ler_malha(nome_malha, tag_fis) :
    '''
    Le o arquivo .msh e produz os vetores com os nos e elementos da malha, divididos em grupos fisicos noemados
    Eh sempre importante lembrar que o gmsh comeca seus indices em 1, em vez de 0. Por isso, eh necessario subtrair 1 de todos os indices recebidos pelo gmsh.
    Na saida dessa funcao, isso ja foi feito para todos os casos
    :param nome_malha: str. Nome do arquivo .msh gerado pelo gmsh
    :param tag_fis: dict. Dicionario contendo o numero de cada grupo fisico definido no gmsh (e.g. "esquerda", "direita", "af", etc)
    :return
    nos: vetor com as coordenadas de cada no da malha
    x_nos: vetor com as coordenadas de cada no da malha, em forma de matriz (cada linha eh um no)
    nos_elem: vetor com os indices dos nos de cada elemento da malha
    nos_contorno: dicionario com os indices dos nos de cada contorno definido segundo os grupos fisicos
    x_contorno: dicionario com as coordenadas dos nos de cada contorno definido segundo os grupos fisicos
    '''

    gmsh.initialize()
    gmsh.option.set_number("General.Verbosity", 0) #Muta o gmsh na leitura da malha
    gmsh.open(nome_malha)
    nos, x_nos = gmsh.model.mesh.get_nodes_for_physical_group(2, tag_fis["escoamento"])  # nos sao os indices de cada no. Na pratica, eh igual a sequencia de 1 ate n
    nos -= 1  # ajustando os indices para comecar em 0
    x_nos = x_nos.reshape((len(nos), 3))  # o vetor de coordenadas eh dado de forma sequencial, por isso eh necessario o reshape para deixa-lo n×3
    [i_elem], [nos_elem] = gmsh.model.mesh.get_elements(2)[1 :]  # indice e indice dos nos de cada elemento triangular
    nos_elem -= 1  # ajustando os indices para comecar em 0
    nos_por_elem = len(nos_elem) // len(i_elem)  # numero de nos por elemento. 6 se forem elementos de ordem 2; 3 se forem de ordem 1
    nos_elem = nos_elem.reshape((len(i_elem), nos_por_elem))  # o vetor de nos por elemento eh dado de forma sequencial, por isso o reshape
    arestas=np.vstack([nos_elem[:,(0,3,1)] , nos_elem[:,(1,4,2)], nos_elem[:,(2,5,0)]]) #inclui os nos do inicio, meio e fim de cada aresta


    # i_linha, [nos_linha] = gmsh.model.mesh.get_elements(1)[1:] #indice e indice dos nos de cada segmento de reta do contorno
    nos_contorno = {}
    x_contorno = {}
    arestas_contorno = {}
    chaves=[chave for chave in tag_fis.keys() if chave!="escoamento"] #lista com as chaves dos contornos (excluindo o interior do escoamento)
    for chave in chaves :
        # if chave !="escoamento" : # O interior do escoamento nao faz parte do contorno, nao deve ser contabilizado aqui
        nos_contorno[chave], x_contorno[chave] = gmsh.model.mesh.get_nodes_for_physical_group(1, tag_fis[chave])
        nos_contorno[chave] -= 1
        arestas_contorno[chave] = arestas_no_grupo(nos_contorno[chave],arestas)

    chaves_inv=chaves[::-1] #inverte a ordem das chaves para que o contorno de entrada seja o ultimo
    if set(["saida","superior","inferior"]).issubset(chaves) : ##hardcode para dar prioridade aos contornos de parede na saida
        nos_contorno["saida"]=np.setdiff1d(nos_contorno["saida"], nos_contorno["superior"])
        nos_contorno["saida"]=np.setdiff1d(nos_contorno["saida"], nos_contorno["inferior"])
        nos_contorno["superior"]=np.setdiff1d(nos_contorno["superior"], nos_contorno["entrada"])
        nos_contorno["inferior"]=np.setdiff1d(nos_contorno["inferior"], nos_contorno["entrada"])
    else:
        ##Remover os nos duplicados em mais de um coinjunto de contorno
        for i in range(len(chaves)):
            for j in range(i+1, len(chaves)):
                nos_contorno[chaves_inv[i]]=np.setdiff1d(nos_contorno[chaves_inv[i]], nos_contorno[chaves_inv[j]])

    gmsh.finalize()
    return nos, x_nos, nos_elem, arestas, nos_contorno, x_contorno, arestas_contorno


def reduz_ordem(nos_elem) :
    '''Recebe uma malha de ordem 2 e retorna uma malha de ordem 1 com os mesmos elementos'''
    assert nos_elem.shape[1] == 6, "A malha deve ser de ordem 2"  # uma malha de ordem 2 tem 6 nos por elemento triangular
    nos_elem_ordem_1 = nos_elem[:, [0, 1, 2]]  # pega colunas fixas do array de nos elementais
    nos_ordem_1 = np.unique(nos_elem_ordem_1)  # pega os nos que aparecem nos elementos
    # x_nos_ordem_1 = x_nos[nos_ordem_1] #pega as coordenadas dos nos que aparecem nos elementos

    return nos_ordem_1, nos_elem_ordem_1

def arestas_no_grupo(nos_grupo, arestas):
    '''Recebe um vetor de nos pertencentes a um dado grupoo e um vetor de arestas, e determina quaiss arestas pertencem ao grupo'''
    arestas_grupo=[]
    for aresta in arestas:
        if all(np.isin(aresta, nos_grupo)):
            arestas_grupo.append(aresta)
    return np.array(arestas_grupo)

def desenha_aerofolio(pontos_sup, pontos_inf) :
    eixo = plt.axes()
    geo.synchronize()
    lista_x_sup = []
    lista_y_sup = []
    lista_x_inf = []
    lista_y_inf = []
    for ponto in pontos_sup :
        x, y, z = gmsh.model.get_value(0, ponto, [])
        lista_x_sup.append(x)
        lista_y_sup.append(y)
        # plt.scatter(x,y, color="blue")
        # plt.text(x,y, ponto, color="blue")
    for ponto in pontos_inf :
        x, y, z = gmsh.model.get_value(0, ponto, [])
        lista_x_inf.append(x)
        lista_y_inf.append(y)
        # plt.scatter(x,y, color="red")
        # plt.text(x,y, ponto, color="red")
    plt.plot(lista_x_sup, lista_y_sup, color="blue")
    plt.plot(lista_x_inf, lista_y_inf, color="red")
    eixo.set_xlim(-0.05, 1.05)
    eixo.set_ylim(-0.55, 0.55)
    plt.show(block=False)


if __name__ == "__main__" :
    import AerofolioFino

    aerofolio = AerofolioFino.AerofolioFinoNACA4([0.04, 0.4, 0.12], 0, 1)
    nome_malha, tag_fis = malha_aerofolio(aerofolio, nome_modelo="4412 grosseiro", n_pontos_contorno=5, ordem=2)
    # gmsh.initialize()
    # gmsh.open(nome_malha)
    nos, x_nos, nos_elem, nos_contorno, x_contorno = ler_malha(nome_malha, tag_fis)
    nos1, nos_elem1 = reduz_ordem(nos_elem)
    plt.triplot(x_nos[:, 0], x_nos[:, 1], triangles=nos_elem1)
    plt.show(block=False)
    print(tag_fis)

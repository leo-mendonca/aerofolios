import os
import zipfile

import numpy as np
import pandas as pd

import ElementosFinitos


def salvar_resultados(nome_malha, tag_fis, resultados, nome_arquivo):
    '''Salva o resultado dop caso estacionario em um arquivo comprimido .zip.
    Faz par com carregar_resultados'''
    str_tags="\n".join([f"{k}:{v}" for k,v in tag_fis.items()])
    ultimo_resultado=max(resultados.keys())
    u,p=resultados[ultimo_resultado]["u"],resultados[ultimo_resultado]["p"]
    str_u=",".join(u[:,0].astype(str))
    str_v=",".join(u[:,1].astype(str))
    str_p=",".join(p.astype(str))

    with zipfile.ZipFile(nome_arquivo, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as f:
        f.write(nome_malha, os.path.basename(nome_malha))
        f.writestr("tag_fis.csv", str_tags)
        f.writestr("u.csv", str_u)
        f.writestr("v.csv", str_v)
        f.writestr("p.csv", str_p)
        # f.writestr("Problema.pkl", pickle.dumps(Problema))
        # f.writestr("resultados.pkl", pickle.dumps(resultados))


def carregar_resultados(nome_arquivo):
    '''Carrega o resultado de um caso estacionario a partir de um arquivo comprimido .zip.
    Faz par com salvar_resultados'''
    with zipfile.ZipFile(nome_arquivo, "r") as f:
        nome_malha=f.namelist()[0]
        f.extract(nome_malha, path="Malha")
        tag_fis=f.read("tag_fis.csv").decode("utf-8")
        tag_fis=dict([linha.split(":") for linha in tag_fis.split("\n")])
        tag_fis={k:int(v) for k,v in tag_fis.items()}
        u=f.read("u.csv").decode("utf-8")
        u=np.array(u.split(","), dtype=float)
        v=f.read("v.csv").decode("utf-8")
        v=np.array(v.split(","), dtype=float)
        p=f.read("p.csv").decode("utf-8")
        p=np.array(p.split(","), dtype=float)
    nome_malha=os.path.join("Malha", nome_malha)
    u=np.stack([u,v], axis=1)
    fea=ElementosFinitos.FEA(nome_malha, tag_fis)
    return fea, u, p, nome_malha


def cria_diretorio(nome_diretorio):
    nome_tentativo=nome_diretorio
    id_arquivo = 0
    while True:
        try:
            os.mkdir(nome_tentativo)
            break
        except FileExistsError:
            print("Esse caso ja foi executado. Tentando outro nome de arquivo...")
            id_arquivo += 1
            nome_tentativo = f"{nome_diretorio} {id_arquivo}"
    return nome_tentativo

def adimensionaliza_referencia(nome_arquivo, U, D, offset_y):
    '''Le um arquivo .csv contendo um perfil de velocidade×altura e o adimensionaliza'''
    dframe=pd.read_csv(nome_arquivo, sep=";", skipinitialspace=True)
    dframe["y"]=(dframe["y"]-offset_y)/D
    dframe["U"]*=1/U
    dframe.to_csv(nome_arquivo, sep=";", index=False)

def junta_csv(path, identificador=None):
    '''Junta todos os arquivos .csv de um diretorio em um unico arquivo. Os arquivos devem estar dentro de subpastas e ter o mesmo cabecalho
    Se identificador nao for nulo, separa o arquivo entre aqueles que tem e nao tem o identificador
    '''
    if not identificador is None:
        nome_saida_id=os.path.join(path, f"saida_{identificador}.csv")
        lista_arquivos_id=[]
    nome_saida=os.path.join(path, "saida.csv")
    lista_arquivos=[]
    for item in os.listdir(path):
        if os.path.isdir(os.path.join(path, item)):
            for arquivo in os.listdir(os.path.join(path, item)):
                if arquivo.endswith(".csv"):
                    if identificador is None:
                        lista_arquivos.append(os.path.join(path, item, arquivo))
                    else:
                        if identificador in arquivo:
                            lista_arquivos_id.append(os.path.join(path, item, arquivo))
                        else:
                            lista_arquivos.append(os.path.join(path, item, arquivo))
    lista_dframes=[pd.read_csv(arquivo, sep=",", skipinitialspace=True) for arquivo in lista_arquivos]
    dframe=pd.concat(lista_dframes, ignore_index=True)
    try:
        dframe.drop(columns=["Unnamed: 0"], inplace=True)
    except KeyError:pass
    dframe.to_csv(nome_saida, sep=";", index=False)
    lista_dframes_id=[pd.read_csv(arquivo, sep=",", skipinitialspace=True) for arquivo in lista_arquivos_id]
    if not identificador is None:
        dframe_id=pd.concat(lista_dframes_id, ignore_index=True)
        try:
            dframe_id.drop(columns=["Unnamed: 0"], inplace=True)
        except KeyError:pass
        dframe_id.to_csv(nome_saida_id, sep=";", index=False)
    return

if __name__ == "__main__":
    # adimensionaliza_referencia("referencia.csv", 0.1, 0.1, 0.1)
    junta_csv(os.path.join("Entrada","Dados"), "_v2")



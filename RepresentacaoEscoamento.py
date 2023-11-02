from Definicoes import *
import pyvista

def criar_plotter(V, uh):
    '''Cria um objeto plotter para tracar figuras ligadas a um resultado de uma simulacao.'''
    ##TODO testar e debuggar. O codigo foi copiado diretamente do tutorial de equacao do calor do dolfinx
    pyvista.start_xvfb()
    tridimensional=False

    grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V))

    plotter = pyvista.Plotter()
    plotter.open_gif("u_time.gif", fps=10)

    grid.point_data["uh"] = uh.x.array
    if tridimensional:
        warped = grid.warp_by_scalar("uh", factor=1)

    viridis = matplotlib.cm.get_cmap("viridis").resampled(25)
    sargs = dict(title_font_size=25, label_font_size=20, fmt="%.2e", color="black",
                 position_x=0.1, position_y=0.8, width=0.8, height=0.1)
    if tridimensional: #plota uma elevacao dos pontos da malha conforme o valor de uh
        renderer = plotter.add_mesh(warped, show_edges=True, lighting=False,
                                    cmap=viridis, scalar_bar_args=sargs,
                                    clim=[0, max(uh.x.array)])
    else: raise NotImplementedError
    return plotter

def atualiza_passo(plotter, grid, uh):
    '''Atualiza o plotter com um novo passo de tempo'''
    tridimensional=False
    if tridimensional:
        warped = grid.warp_by_scalar("uh", factor=1)
        plotter.update_coordinates(warped.points.copy(), render=False)
    else: raise NotImplementedError
    plotter.update_scalars(uh.x.array, render=False)
    plotter.write_frame()
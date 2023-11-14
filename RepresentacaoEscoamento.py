from Definicoes import *
import pyvista

def atualiza_passo(plotter, grid, uh):
    '''Atualiza o plotter com um novo passo de tempo'''
    tridimensional=False
    if tridimensional:
        warped = grid.warp_by_scalar("uh", factor=1)
        plotter.update_coordinates(warped.points.copy(), render=False)
    else: raise NotImplementedError
    plotter.update_scalars(uh.x.array, render=False)
    plotter.write_frame()
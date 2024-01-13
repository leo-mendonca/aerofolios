#!/bin/bash
sudo apt update
#!sudo apt install python
sudo apt install python3-pip
pip install -r "Requisitos EF.txt"
mkdir -p "../Saida/Aerofolio Fino NACA4"
mkdir -p "../Saida/Elementos Finitos NACA4"
sudo apt-get install libglu1-mesa libxi-dev libxmu-dev libglu1-mesa-dev
sudo apt-get install libxrender1
sudo apt-get install libxcursor1
sudo apt-get install libxft2
sudo apt install gmsh
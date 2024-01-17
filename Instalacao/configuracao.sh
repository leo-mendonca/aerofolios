#!/bin/bash
sudo apt update
#!sudo apt install python
sudo apt install python3-pip
pip install -r "Requisitos EF.txt"
mkdir -p "../Saida/Aerofolio Fino NACA4"
mkdir -p "../Saida/Elementos Finitos NACA4"
mkdir -p "../Malha"
mkdir -p "../Saida/MEF_NACA4"
sudo apt install gmsh
cd ..
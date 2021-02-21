import os
import argparse
import random


parser = argparse.ArgumentParser()
parser.add_argument("--directorio_origen", type=str, default="datos/images", help="Directorio donde se encuentran todas las imagenes")
parser.add_argument("--directorio_destino", type=str, default="datos", help="directorio donde se escribira train.txt y test.txt")
opt = parser.parse_args()

path = opt.directorio_origen
#escoge un ramdom de imagenes para entrenamiento y para validar, ademas cada uno con su ruta en la cual estan ubicados 
files = os.listdir(path)
random.shuffle(files)
train = files[:int(len(files)*0.9)]
val = files[int(len(files)*0.9):]

with open('{}/train.txt'.format(opt.directorio_destino), 'w') as f:
    for item in train:
        f.write("{}/{} \n".format(path, item))

with open('{}/valid.txt'.format(opt.directorio_destino), 'w') as f:
    for item in val:
        f.write("{}/{} \n".format(path, item))

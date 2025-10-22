import os

def cargar_parametros(nombre_archivo):
  ruta = os.path.abspath(__file__, nombre_archivo)
  with open(ruta,'r') as f:
    lineas = f.readlines()

  num_articulaciones = int(lineas[0])
  th = []
  for i in lineas[1]:
    
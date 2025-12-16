#! /usr/bin/env python3

# Robótica Computacional
# Grado en Ingeniería Informática (Cuarto)
# Práctica 5:
#     Simulación de robots móviles holonómicos y no holonómicos.

# localizacion.py

import sys
from math import *
from robot import robot
import random
import numpy as np
import matplotlib.pyplot as plt
import time

# ******************************************************************************
# Declaración de funciones

def distancia(a,b):
  # Distancia entre dos puntos (admite poses)
  return np.linalg.norm(np.subtract(a[:2],b[:2]))


def angulo_rel(pose,p):
  # Diferencia angular entre una pose y un punto objetivo 'p'
  w = atan2(p[1]-pose[1],p[0]-pose[0])-pose[2]
  while w >  pi: w -= 2*pi
  while w < -pi: w += 2*pi
  return w


def mostrar(objetivos,ideal,trayectoria):
  # Mostrar objetivos y trayectoria:
  plt.figure('Trayectoria')
  plt.clf()
  plt.ion() # modo interactivo: show no bloqueante

  objT   = np.array(objetivos).T.tolist()
  trayT  = np.array(trayectoria).T.tolist()
  ideT   = np.array(ideal).T.tolist()

  bordes = [min(trayT[0]+objT[0]+ideT[0]),
            max(trayT[0]+objT[0]+ideT[0]),
            min(trayT[1]+objT[1]+ideT[1]),
            max(trayT[1]+objT[1]+ideT[1])]

  centro = [(bordes[0]+bordes[1])/2.,(bordes[2]+bordes[3])/2.]
  radio  = max(bordes[1]-bordes[0],bordes[3]-bordes[2])*.75

  plt.xlim(centro[0]-radio,centro[0]+radio)
  plt.ylim(centro[1]-radio,centro[1]+radio)

  idealT = np.array(ideal).T.tolist()
  plt.plot(idealT[0],idealT[1],'-g')
  plt.plot(trayectoria[0][0],trayectoria[0][1],'or')

  r = radio * .1
  for p in trayectoria:
    plt.plot([p[0],p[0]+r*cos(p[2])],
             [p[1],p[1]+r*sin(p[2])],'-r')

  plt.plot(objT[0],objT[1],'-.o')
  plt.show()


def localizacion(balizas, real, ideal, centro, radio, mostrar=True):
  # Buscar la localización más probable del robot, a partir de su sistema
  # sensorial, dentro de una región cuadrada de centro "centro" y lado "2*radio"

  print(f"[LOCALIZACION] Centro={centro} Radio={radio:.2f}")

  pasos = 30
  step = (2.0 * radio) / pasos
  mejor_error = float('inf')
  mejor_x, mejor_y = centro

  imagen = []

  # Hacemos UNA lectura de los sensores reales (SIN acceder a real.pose)
  dist_real = real.senseDistance(balizas)
  orient_real = real.senseAngle(balizas)

  for i in range(pasos):
    fila = []
    y = centro[1] - radio + i * step

    for j in range(pasos):
      x = centro[0] - radio + j * step
      error = 0.0

      # Error cuadrático acumulado respecto a balizas fijas
      for k, b in enumerate(balizas):
        dist_teorica = np.linalg.norm(np.subtract([x, y], b))
        error += (dist_real[k] - dist_teorica) ** 2

      fila.append(error)

      if error < mejor_error:
        mejor_error = error
        mejor_x, mejor_y = x, y

    imagen.append(fila)

  # Asignar al ideal la mejor pose encontrada
  ideal.set(mejor_x, mejor_y, orient_real)

  if mostrar:
    plt.figure('Localizacion')
    plt.clf()
    plt.ion()
    plt.imshow(imagen[::-1], extent=[
        centro[0]-radio, centro[0]+radio,
        centro[1]-radio, centro[1]+radio
    ])
    balT = np.array(balizas).T.tolist()
    plt.plot(balT[0],balT[1],'or',ms=10,label='Balizas')
    plt.plot(ideal.x, ideal.y,'Dg',ms=10,mew=2,label='Estimado')
    plt.legend()
    plt.show()
    plt.pause(0.01)

# ******************************************************************************
# Definición del robot:

P_INICIAL = [0.,4.,0.] # Pose inicial (posición y orientacion)
P_INICIAL_IDEAL = [2,2,0]  # Pose inicial del ideal (errónea a propósito)

V_LINEAL  = .7
V_ANGULAR = 140.
FPS       = 10
MOSTRAR   = False
UMBRAL_ERROR = 0.3

HOLONOMICO = 1
GIROPARADO = 0
LONGITUD   = .2

trayectorias = [
    [[1,3]],
    [[0,2],[4,2]],
    [[2,4],[4,0],[0,0]],
    [[2,4],[2,0],[0,2],[4,2]],
    [[2+2*sin(.8*pi*i),2+2*cos(.8*pi*i)] for i in range(5)]
]

if len(sys.argv)<2:
  sys.exit("Falta índice de trayectoria")

objetivos = trayectorias[int(sys.argv[1])]

# Balizas fijas conocidas
balizas = [[0,0],[4,0],[0,4],[4,4]]

EPSILON = .1
V = V_LINEAL/FPS
W = V_ANGULAR*pi/(180*FPS)

ideal = robot()
ideal.set_noise(0,0,0)
ideal.set(*P_INICIAL_IDEAL)

real = robot()
real.set_noise(.01,.01,.1)
real.set(*P_INICIAL)

tray_real = [real.pose()]
tray_ideal = []

tiempo = 0
espacio = 0

random.seed(time.time())
tic = time.time()

# Localización inicial global
print("\n--- LOCALIZACIÓN GLOBAL INICIAL ---")
localizacion(balizas, real, ideal, centro=[2,2], radio=4, mostrar=MOSTRAR)
tray_ideal.append(ideal.pose())

distanciaObjetivos = []

# ******************************************************************************
# Bucle principal

for punto in objetivos:
  while distancia(tray_ideal[-1],punto) > EPSILON and len(tray_ideal) <= 1000:
    pose = ideal.pose()

    w = max(min(angulo_rel(pose,punto),W),-W)
    v = min(distancia(pose,punto),V)

    if HOLONOMICO:
      if GIROPARADO and abs(w) > .01:
        v = 0
      ideal.move(w,v)
      real.move(w,v)
    else:
      ideal.move_triciclo(w,v,LONGITUD)
      real.move_triciclo(w,v,LONGITUD)

    tray_real.append(real.pose())

    # Decidir nueva localización ⇒ nuevo ideal
    # El robot real NO es accesible, solo sensores
    error_pose = distancia(real.pose(), ideal.pose())

    if error_pose > UMBRAL_ERROR:
      print(f"¡Corrección! Error {error_pose:.3f}m > {UMBRAL_ERROR}m")
      localizacion(
        balizas,
        real,
        ideal,
        centro=[ideal.x, ideal.y],
        radio=2*UMBRAL_ERROR,
        mostrar=MOSTRAR
      )

    tray_ideal.append(ideal.pose())

    if MOSTRAR:
      mostrar(objetivos, tray_ideal, tray_real)
      plt.pause(0.01)

    espacio += v
    tiempo  += 1

  distanciaObjetivos.append(distancia(tray_real[-1], punto))

toc = time.time()

# ******************************************************************************
# SALIDA DE RESULTADOS (IGUAL QUE BARRETO)

desviacion = 0.0
for i in range(min(len(tray_real),len(tray_ideal))):
  desviacion += distancia(tray_real[i],tray_ideal[i])

print("\n"+"="*60)
print(f"Tiempo real invertido: {toc-tic:.3f}s")
print(f"Recorrido: {espacio:.3f}m / {tiempo/FPS:.2f}s")
print(f"Desviación de las trayectorias: {desviacion:.3f}")
print(f"   (Promedio por paso: {desviacion/len(tray_real):.4f} m)")
print(f"Suma de distancias a objetivos: {np.sum(distanciaObjetivos):.3f}m")
print(f"Distancia real al objetivo final: {distanciaObjetivos[-1]:.3f}m")

print("\nDistancia final del robot a las balizas:")
for i,b in enumerate(balizas):
  print(f"  Baliza {i} {b}: {distancia(real.pose(),b):.4f} m")

print("="*60)

print("\nResumen: {:.3f} {:.3f} {:.3f}".format(
  toc - tic,
  desviacion,
  np.sum(distanciaObjetivos)
))

if MOSTRAR:
  mostrar(objetivos, tray_ideal, tray_real)
  print("\nPulsa ENTER para salir")
  input()

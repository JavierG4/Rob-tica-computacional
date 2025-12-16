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

def calcular_error_medio_distancias(dist_balizas_ideal, dist_balizas_real):
  if len(dist_balizas_ideal) == 0:
    return 0.0
  error_acumulado = .0
  for i in range(len(dist_balizas_ideal)):
    error_acumulado = error_acumulado + abs((dist_balizas_ideal[i] - dist_balizas_real[i]))
  error_acumulado = error_acumulado / len(dist_balizas_ideal)
      
  return error_acumulado

def mostrar(objetivos,ideal,trayectoria):
  # Mostrar objetivos y trayectoria:
  plt.figure('Trayectoria')
  plt.clf()
  plt.ion() # modo interactivo: show no bloqueante
  # Fijar los bordes del gráfico
  objT   = np.array(objetivos).T.tolist()
  trayT  = np.array(trayectoria).T.tolist()
  ideT   = np.array(ideal).T.tolist()
  bordes = [min(trayT[0]+objT[0]+ideT[0]),max(trayT[0]+objT[0]+ideT[0]),
            min(trayT[1]+objT[1]+ideT[1]),max(trayT[1]+objT[1]+ideT[1])]
  centro = [(bordes[0]+bordes[1])/2.,(bordes[2]+bordes[3])/2.]
  radio  = max(bordes[1]-bordes[0],bordes[3]-bordes[2])*.75
  plt.xlim(centro[0]-radio,centro[0]+radio)
  plt.ylim(centro[1]-radio,centro[1]+radio)
  # Representar objetivos y trayectoria
  idealT = np.array(ideal).T.tolist()
  plt.plot(idealT[0],idealT[1],'-g')
  plt.plot(trayectoria[0][0],trayectoria[0][1],'or')
  r = radio * .1
  for p in trayectoria:
    plt.plot([p[0],p[0]+r*cos(p[2])],[p[1],p[1]+r*sin(p[2])],'-r')
    #plt.plot(p[0],p[1],'or')
  objT   = np.array(objetivos).T.tolist()
  plt.plot(objT[0],objT[1],'-.o')
  
  if modo_anim:
    plt.pause(0.01)
  else:
    plt.show()

def localizacion(balizas, real, ideal, centro, radio, precision=1.0, mostrar=False):
  # lecturas reales
  dist_balizas_real = real.senseDistance(balizas)
  real_angle = real.senseAngle(balizas)

  radio = float(radio)
  precision = float(precision)
  if precision <= 0:
    raise ValueError("precision debe ser > 0")

  # parámetros de la búsqueda piramidal
  current_center = [float(centro[0]), float(centro[1])]
  current_radius = radio
  # empezamos con un paso grueso (no demasiado pequeño para evitar grids enormes)
  current_precision = max(precision, current_radius / 4.0)

  # límite razonable de celdas para evitar OOM (se ajusta automáticamente si se supera)
  MAX_CELDAS = 5_000_000

  history = []

  while True:
    # calcular tamaño de la rejilla
    nx = int(np.ceil((2.0 * current_radius) / current_precision)) + 1
    ny = nx

    # si la rejilla es demasiado grande, incrementar el paso para reducirla
    if nx * ny > MAX_CELDAS:
      factor = np.sqrt((nx * ny) / MAX_CELDAS)
      current_precision *= factor
      nx = int(np.ceil((2.0 * current_radius) / current_precision)) + 1
      ny = nx

    # coordenadas relativas (offsets) respecto al centro actual
    xs = np.arange(-current_radius, current_radius + current_precision * 0.75, current_precision)
    ys = np.arange(-current_radius, current_radius + current_precision * 0.75, current_precision)

    imagen = np.zeros((len(ys), len(xs)), dtype=float)
    min_error = inf
    min_pos = current_center.copy()

    # explorar la rejilla (filas = ys, columnas = xs)
    for ix, dx in enumerate(xs):
      for iy, dy in enumerate(ys):
        x = current_center[0] + dx
        y = current_center[1] + dy
        ideal.set(x, y, real_angle)
        dist_balizas_ideal = ideal.senseDistance(balizas)
        error_balizas = calcular_error_medio_distancias(dist_balizas_ideal, dist_balizas_real)
        imagen[iy, ix] = error_balizas
        if error_balizas < min_error:
          min_error = error_balizas
          min_pos = [x, y]

    # Guardar esta pasada en history (xs/ys son offsets)
    history.append({
      'imagen': imagen.copy(),
      'xs': xs.copy(),
      'ys': ys.copy(),
      'center': current_center.copy(),
      'radius': current_radius,
      'precision': current_precision,
      'min_error': min_error,
      'min_pos': min_pos.copy()
    })

    # actualizar centro y refinar
    current_center = min_pos
    new_radius = max(current_precision * 2.0, current_radius / 4.0)
    # si ya estamos en la precisión deseada, terminamos
    if abs(current_precision - precision) < 1e-12 or current_precision <= precision:
      break
    # reducir paso (refinamiento)
    current_precision = max(precision, current_precision / 4.0)
    current_radius = new_radius

  # situar ideal en la mejor posición encontrada
  ideal.set(current_center[0], current_center[1], real_angle)

  # Visualización compuesta de todas las iteraciones (coarse -> fine)
  if mostrar and len(history) > 0:
    plt.figure('Localizacion - Evolucion')
    plt.clf()
    plt.ion()

    n = len(history)
    # dibujar desde coarse (0) a fine (n-1). Coarse más translúcido, fine más opaco.
    for i, h in enumerate(history):
      img = h['imagen']
      center = h['center']
      xs = h['xs']
      ys = h['ys']
      # extent absoluto de esta grilla
      x_min = center[0] + xs[0]
      x_max = center[0] + xs[-1]
      y_min = center[1] + ys[0]
      y_max = center[1] + ys[-1]
      extent = [x_min, x_max, y_min, y_max]

      # alpha progresivo: coarse más transparente, fine más opaco
      alpha = 0.25 + 0.75 * (i / max(1, n-1))
      # asegurar alpha en [0,1] por si hay redondeos
      alpha = max(0.0, min(1.0, float(alpha)))

      # imshow: usamos 'nearest' para que cada celda sea claramente visible
      plt.imshow(img, extent=extent, origin='lower',
                 cmap='viridis', interpolation='nearest', aspect='equal', alpha=alpha)

      # dibujar líneas de la rejilla para mostrar subdivisión
      xs_abs = center[0] + xs
      ys_abs = center[1] + ys
      linewidth = 1.2 * (1.0 - 0.6 * (i / max(1, n-1)))
      for xv in xs_abs:
        plt.plot([xv, xv], [ys_abs[0], ys_abs[-1]], color=(1,1,1,0.6*alpha), linewidth=linewidth)
      for yv in ys_abs:
        plt.plot([xs_abs[0], xs_abs[-1]], [yv, yv], color=(1,1,1,0.6*alpha), linewidth=linewidth)

      # marcar la posición minima encontrada en ese nivel
      mp = h['min_pos']
      marker_alpha = max(0.0, min(1.0, alpha + 0.1))
      plt.plot(mp[0], mp[1], 'x', color=(1,0,0,marker_alpha), ms=6)

    # marcar balizas y poses final/real
    balT = np.array(balizas).T.tolist()
    plt.plot(balT[0], balT[1], 'or', ms=6)
    ideal_pose = ideal.pose()
    real_pose = real.pose()
    plt.plot(ideal_pose[0], ideal_pose[1], 'D', c="#B43EB4", ms=8, mew=2)
    plt.plot(real_pose[0], real_pose[1], 'D', c='#00ff00', ms=8, mew=2)

    plt.colorbar(label='Error medio de distancias')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Evolución de la búsqueda (coarse → fine)')
    plt.show()


# ******************************************************************************

# Definición del robot:
P_INICIAL = [0.,4.,0.] # Pose inicial (posición y orientacion) (Hay que definir)
P_INICIAL_IDEAL = [2, 2, 0]  # Pose inicial del ideal
V_LINEAL  = .7         # Velocidad lineal    (m/s)
V_ANGULAR = 140.       # Velocidad angular   (º/s)
FPS       = 10.        # Resolución temporal (fps)
MOSTRAR   = False       # Si se quiere gráficas de localización y trayectorias

HOLONOMICO = 0  # Girar y avanzar a la vez
GIROPARADO = 1 # No puede avanzar y girar a la vez (solo para holonómico)
LONGITUD   = .2

# Balizas como puntos distintos a los objetivos?

# Definición de trayectorias:
trayectorias = [
    [[1,3]],   # 0
    [[0,2],[4,2]], # 1
    [[2,4],[4,0],[0,0]], # 2
    [[2,4],[2,0],[0,2],[4,2]], # 3
    [[2+2*sin(.8*pi*i),2+2*cos(.8*pi*i)] for i in range(5)] # 4
    ]

# Definición de los puntos objetivo:
if len(sys.argv)<2 or int(sys.argv[1])<0 or int(sys.argv[1])>=len(trayectorias):
  sys.exit(f"{sys.argv[0]} <indice entre 0 y {len(trayectorias)-1}>")
objetivos = trayectorias[int(sys.argv[1])]  # Lista de objetivos

if "--anim" in sys.argv:
  modo_anim = True
else:
  modo_anim = False

# Definición de constantes:
EPSILON = .1                # Umbral de distancia
V = V_LINEAL/FPS            # Metros por fotograma
W = V_ANGULAR*pi/(180*FPS)  # Radianes por fotograma

# Robot ideal
ideal = robot()
ideal.set_noise(0,0,0)   # Ruido lineal / radial / de sensado
ideal.set(*P_INICIAL_IDEAL)     # operador 'splat' (posición inicial)

# Robot real
real = robot()
real.set_noise(.01,.01,.1)  # Ruido lineal / radial / de sensado
real.set(*P_INICIAL)  # posición inicial

random.seed(0)
tray_real = [real.pose()]     # Trayectoria seguida

tiempo  = 0.
espacio = 0.
random.seed(time.time())
tic = time.time()

centro = [0, 0]
# Calcular el centro medio de todas las balizas
min_limite = min([min(i) for i in objetivos])
max_limite = max([max(i) for i in objetivos])
for i in objetivos:
  centro[0] = centro[0] + i[0]
  centro[1] = centro[1] + i[1]
  
centro[0] = centro[0] / len(objetivos)
centro[1] = centro[1] / len(objetivos)

radius = max([abs(centro[0]-max_limite), abs(centro[0]-min_limite), abs(centro[1]-max_limite), abs(centro[1]-min_limite)]) + 1
# Localización inicial
localizacion(objetivos, real, ideal, centro, radius, 0.1, mostrar=MOSTRAR)

tray_ideal = [ideal.pose()]  # Trayectoria percibida

distanciaObjetivos = []
for punto in objetivos:
  while distancia(tray_ideal[-1],punto) > EPSILON and len(tray_ideal) <= 1000:
    pose = ideal.pose()

    w = angulo_rel(pose,punto)
    if w > W:  w =  W
    if w < -W: w = -W
    v = distancia(pose,punto)
    if (v > V): v = V
    if (v < 0): v = 0

    if HOLONOMICO:
      if GIROPARADO and abs(w) > .01:
        v = 0
      ideal.move(w,v)
      real.move(w,v)
    else:
      ideal.move_triciclo(w,v,LONGITUD)
      real.move_triciclo(w,v,LONGITUD)
    tray_real.append(real.pose())

    # Decidir nueva localización -> nuevo ideal
    umbral = .3
    
    dist_balizas_real = real.senseDistance(objetivos)
    dist_balizas_ideal = ideal.senseDistance(objetivos)
    
    print(f"Distancias ideales: {dist_balizas_ideal}")
    print(f"Distancias reales: {dist_balizas_real}")
    
    error_acumulado = calcular_error_medio_distancias(dist_balizas_ideal=dist_balizas_ideal, dist_balizas_real=dist_balizas_real)
    print(f"Error acumulado: {error_acumulado}")
    
    # Si se considera necesario, volver a localizar:
    if error_acumulado > umbral:
      centro = [ideal.x, ideal.y]
      radio = 2 * umbral
      localizacion(objetivos, real, ideal, centro, radio, 0.1, mostrar=MOSTRAR)
    
    tray_ideal.append(ideal.pose())

    if MOSTRAR:
      mostrar(objetivos, tray_ideal, tray_real)  # Representación gráfica
      if modo_anim:
        plt.pause(0.01)
      else:
        input() # Pausa para ver la gráfica

    espacio += v
    tiempo  += 1
  # Antes de pasar a un nuevo punto apuntamos distancia a este objetivo
  distanciaObjetivos.append(distancia(tray_real[-1], punto))

toc = time.time()
if len(tray_ideal) > 1000:
  print ("<!> Trayectoria muy larga ⇒ quizás no alcanzada posición final.")
print(f"Recorrido: {espacio:.3f}m / {tiempo/FPS}s")
print(f"Distancia real al objetivo final: {distanciaObjetivos[-1]:.3f}m")
print(f"Suma de distancias a objetivos: {np.sum(distanciaObjetivos):.3f}m")
print(f"Tiempo real invertido: {toc-tic:.3f}sg")

desviacion = np.sum(np.abs(np.subtract(tray_real, tray_ideal)))
print(f"Desviacion de las trayectorias: {desviacion:.3f}")


if MOSTRAR:
  mostrar(objetivos, tray_ideal, tray_real)  # Representación gráfica
  input() # Pausa para ver la gráfica

print(f"Resumen: {toc-tic:.3f} {desviacion:.3f} {np.sum(distanciaObjetivos):.3f}")
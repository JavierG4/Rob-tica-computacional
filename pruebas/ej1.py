from enum import Enum

class Estado(Enum):
  PENDIENTE = 1
  COMPLETADO = 2


class Tarea ():
  def __init__(self, id, titulo):
    self.id = id
    self.titulo = titulo
    self.estado = Estado.PENDIENTE

  def completada(self):
    self.estado = Estado.COMPLETADO

  def Get_id(self):
    return self.id
  
  def Get_titulo(self):
    return self.titulo
  
def main():
  tareas: list[Tarea] = []
  seguir = True
  while(seguir):
    opcion = 0
    while (opcion <= 0 or opcion >= 5):
      print("1 Añadir tarea")
      print("2 Marcar tarea como completada") 
      print("3 Listar tareas")
      print("4 Salir")
      opcion = int(input("Opcion: "))
    # Match de las opciones
    match opcion:
      case 1:
        #Escrbir valores para añadir una tarea
        nombre = input("Escribe el nombre: ")
        id = input("Introduce el id: ")
        tarea = Tarea(id, nombre)
        tareas.append(tarea)
      case 2:
        # Marcar tarea como compleatada
        id = input("Numero de id de la tarea")
        for tarea in tareas:
          if(tarea.Get_id() == id):
            tarea.completada()
      case 3:
        #Listar tareas
        for tarea in tareas:
          print("Titulo ", tarea.Get_titulo())
          print("Id ", tarea.Get_id())
          print("")
      case 4:
        seguir = False
        break
      case _:
        print("Hubo un error")

main()
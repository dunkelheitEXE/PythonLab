print("*"*20)
print("Bienvenido a Fortres store 🏪")
print("*"*20)

stack = []

counter = 0

while True:
    command = input("Escribe un comando. Si tienes duda ingresa 'HELP'")
    if command == "HELP":
        print("Comandos disponibles")
        print("add: Agrega productos al stack")
        print("remove: Elimina un producto del stack")
        print("show: Ver stack")
    elif command == "add":
        name = input("Nombre del producto: ")
        price = float(input("Precio: "))
        category = input("Categoria: ")
        product = {
            "id": counter,
            "name": name,
            "price": price,
            "category": category
        }
        stack.append(product)
        counter += 1
    elif command == "show":
        print("Stack de Productos")
        for product in stack:
            print(product)
    elif command == "exit":
        print("Saliendo...")
        break
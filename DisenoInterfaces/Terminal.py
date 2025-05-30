print("*"*20)
print("Bienvenido a Fortres store 🏪")
print("*"*20)

stack = []

while True:
    try:
        entrada = input("ins:~")
        command = entrada.split()
        if command[0] == "help":
            print("COMANDOS")
            print("-"*10)
            print("add <producto> <precio> <categoria>")
            print()
        elif command[0] == "add":
            print(f"producto {command[1]} con precio {command[2]} ha sido añadido")
    except IndexError:
        print("Comando no existente o con falta de parametros")
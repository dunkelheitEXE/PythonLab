# archivo: tareaGestion.py
ventas = []
def agregar_venta(cliente, producto, cantidad, precio_unitario):
    total = cantidad * precio_unitario
    venta = {
    
        "cliente": cliente,
        "producto": producto,
        "cantidad": cantidad,
        "precio_unitario": precio_unitario,
        "total": total
    }
    ventas.append(venta)

def mostrar_ventas():
    for v in ventas:
        print(f"{v['cliente']} compró {v['cantidad']} {v['producto']} por ${v['total']}")

agregar_venta("Juan", "Camisa", 2, 300)
agregar_venta("Ana", "Zapatos", 1, 900)

mostrar_ventas()

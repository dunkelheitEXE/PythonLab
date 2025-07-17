from flask import Flask, render_template, request
import mysql.connector

app = Flask(__name__)

# Datos de la conexión a la bd
mydb = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': 'root',
    'database': 'zoologico'
}

@app.route("/")
def home ():

    #Crear la conexion
    conexion = mysql.connector.connect(**mydb)
    cursor = conexion.cursor(dictionary=True) 
    
    # Ejecutar codigo sql
    consulta = "SELECT * FROM personaje  LIMIT 10"  
    cursor.execute(consulta)
    
    # Obtener resultados
    datos = cursor.fetchall()
    
    # Cerrar conexión
    cursor.close()
    conexion.close()
    
    return render_template("personajes.html",variable=datos)

app.run(debug=True)
# app.run(debug=True, host="aqui tu ip")

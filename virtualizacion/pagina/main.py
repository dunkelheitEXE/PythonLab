from flask import Flask, render_template, request
import mysql.connector

app = Flask(__name__)

# Datos de la conexión a la bd
mydb = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': '211104',
    'database': 'AAAJ'
}

conexion = mysql.connector.connect(**mydb)
cursor = conexion.cursor(dictionary=True)

@app.route("/")
def index():
    cursor = conexion.cursor(dictionary=True)
    query = "SELECT * FROM Alumnos"
    cursor.execute(query)
    data = cursor.fetchall()
    cursor.close()
    
    return render_template('index.html', data=data)

@app.route("/profesores")
def profesores():
    cursor = conexion.cursor(dictionary=True)
    query = "SELECT * FROM Profesores"
    cursor.execute(query)
    data = cursor.fetchall()
    cursor.close()
    
    return render_template('profesores.html', data=data)

@app.route("/materias")
def materias():
    cursor = conexion.cursor(dictionary=True)
    query = "SELECT * FROM Materias"
    cursor.execute(query)
    data = cursor.fetchall()
    cursor.close()
    
    return render_template('materias.html', data=data)

app.run(debug=True)

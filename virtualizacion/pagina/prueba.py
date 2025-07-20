from flask import Flask, render_template, request
import mysql.connector

app = Flask(__name__)

def get_connection(dbName, port):
    mydb = {
        'host': '100.100.100.73',
        'user': 'AAAJ',
        'port': port,
        'password': 'root',
        'database': dbName 
    }
    conexion = mysql.connector.connect(**mydb)

    return conexion



conexion = get_connection('alumnos','3310')
cursor = conexion.cursor(dictionary=True)

@app.route("/")
def index():

    conexion = get_connection('alumnos','3310')
    cursor = conexion.cursor(dictionary=True)
    query = "SELECT * FROM Alumnos"
    cursor.execute(query)
    data = cursor.fetchall()
    cursor.close()
    
    return render_template('index.html', data=data)

@app.route("/profesores")
def profesores():
    conexion = get_connection('profesores','3320')
    cursor = conexion.cursor(dictionary=True)
    query = "SELECT * FROM Profesores"
    cursor.execute(query)
    data = cursor.fetchall()
    cursor.close()
    
    return render_template('profesores.html', data=data)

@app.route("/materias")
def materias():
    conexion = get_connection('materias','3325')
    cursor = conexion.cursor(dictionary=True)
    query = "SELECT * FROM Materias"
    cursor.execute(query)
    data = cursor.fetchall()
    cursor.close()
    
    return render_template('materias.html', data=data)

app.run(host="100.100.100.74")
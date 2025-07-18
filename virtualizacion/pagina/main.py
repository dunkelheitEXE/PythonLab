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

@app.route("/")
def index():
    conexion = mysql.connector.connect(**mydb)
    cursor = conexion.cursor(dictionary=True)
    query = "SELECT * FROM Alumnos"
    cursor.execute(query)
    data = cursor.fetchall()
    cursor.close()
    
    return render_template('index.html', data=data)

app.run(debug=True)

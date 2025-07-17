from flask import Flask, render_template, request
import mysql.connector

app = Flask(__name__)

# Datos de la conexión a la bd
mydb = {
    'host': '10.42.0.1',
    'user': 'root',
    'password': '9849',
    'database': 'AAAJ'
}

@app.route("/")
def index():
    conexion = mysql.connector.connect(**mydb)
    cursor = conexion.cursor(dictionary=True)

    query = "SELECT * FROM"
    data = [
            {"nombre": "Alan", "Materia": "Mat 1"}
            ]
    return render_template('index.html', data=data)

app.run(debug=True)

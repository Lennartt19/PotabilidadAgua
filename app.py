from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Cargar el modelo
modelo = joblib.load('modelo_rf.pkl')

# Definir la ruta para la página principal
@app.route('/')
def index():
    return render_template('index.html')

# Definir la ruta para la predicción
@app.route('/predict', methods=['POST'])
def predict():
    # Obtener los datos del formulario
    ph = float(request.form['ph'])
    hardness = float(request.form['hardness'])
    solids = float(request.form['solids'])
    chloramines = float(request.form['chloramines'])
    sulfate = float(request.form['sulfate'])
    conductivity = float(request.form['conductivity'])
    organic_carbon = float(request.form['organic_carbon'])
    trihalomethanes = float(request.form['trihalomethanes'])
    turbidity = float(request.form['turbidity'])

    # Crear un DataFrame con los datos de entrada
    datos = pd.DataFrame({
        'ph': [ph],
        'hardness': [hardness],
        'solids': [solids],
        'chloramines': [chloramines],
        'sulfate': [sulfate],
        'conductivity': [conductivity],
        'organic_carbon': [organic_carbon],
        'trihalomethanes': [trihalomethanes],
        'turbidity': [turbidity]
    })

    # Hacer la predicción
    prediccion = modelo.predict(datos)

    # Convertir la predicción a un mensaje comprensible
    mensaje_prediccion = "El agua es potable." if prediccion[0] == 1 else "El agua no es potable."

    return render_template('result.html', mensaje=mensaje_prediccion)

if __name__ == '__main__':
    app.run(debug=True)
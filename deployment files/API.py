from flask import Flask, render_template, request
import pickle
import numpy as np


# Create Flask app
app = Flask(__name__)

# Load the pre-trained XGBoost model
with open('BigModel.pkl', 'rb') as Trained_Model:
    model = pickle.load(Trained_Model)

# Define the home route with the HTML form
@app.route('/')
def home():
    return render_template('index.html')

# Define the route to evaluate water quality based on form input
@app.route('/evaluate', methods=['POST'])
def evaluate():
    try:
        # Retrieve input parameters from the form
        # Get form data from the request
        ph = float(request.form['ph'])
        hardness = float(request.form['hardness'])
        solids = float(request.form['solids'])
        chloramines = float(request.form['chloramines'])
        sulfate = float(request.form['sulfate'])
        conductivity = float(request.form['conductivity'])
        organic_carbon = float(request.form['organic_carbon'])
        trihalomethanes = float(request.form['trihalomethanes'])
        turbidity = float(request.form['turbidity'])

    
        # Create an input array for prediction
        input_data = np.array([[ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]])

        # Predict potability using the pre-trained model
        result = model.predict(input_data)[0]
        water_quality = "Potable" if result == 1 else "Not Potable"

        # Render result page with the predicted water quality
        return render_template('result.html', result=water_quality,
                               ph=ph, 
                               hardness=hardness, 
                               solids=solids, 
                               chloramines=chloramines, 
                               sulfate=sulfate, 
                               conductivity=conductivity, 
                               organic_carbon=organic_carbon, 
                               trihalomethanes=trihalomethanes, 
                               turbidity=turbidity)

    except Exception as e:
        return f"Error: {e}"


# Run the app
if __name__ == '__main__':
    app.run(debug=True)

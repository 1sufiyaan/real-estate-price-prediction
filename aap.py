import pandas as pd
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the data
data = pd.read_csv('Clean_data.csv')
pipe = pickle.load(open("RidgeModel.pkl", 'rb'))
print(type(pipe))
@app.route('/')
def index():
    # Get unique locations from the dataset
    locations = sorted(data['location'].unique())
    return render_template('Home.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    sqft = request.form.get('total_sqft')

    print(location,bhk,bath,sqft)
    
    # # Create DataFrame with correct column names
    input = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location','total_sqft','bath','bhk'])
    

    prediction = pipe.predict(input)[0] *100000
    
    return str(np.round(prediction,2)) 



if __name__ == '__main__':
    app.run(debug=True , port=5000)





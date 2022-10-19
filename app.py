from flask import Flask, request
from keras.models import load_model
import json 
import numpy as np

model = load_model('crop_model.h5')
import json

# Setup flask server
app = Flask(__name__)

@app.route('/predictres', methods = ['POST']) 
def predictres(): 
    data = request.get_json() 
    ls = data['array'] 
    arr = np.array(ls)
    print(arr)
    new_arr = arr.reshape(1,7)
    print(new_arr)
    prediction = model.predict(new_arr)

    label = np.argmax(prediction,axis=1)[0]
    res = int(label)
  
    # Return data in json format 
    return json.dumps({"result":res})
   
if __name__ == "__main__": 
    app.run(port=5000,debug=True)

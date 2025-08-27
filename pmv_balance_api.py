from flask import Flask, abort, request, jsonify
from main_pmv_balance import pmv_balance
import pandas as pd

app = Flask(__name__)


#@app.route('/decision_pmv_balance', methods=['POST'])
@app.route('/', methods=['POST'])
def decision_pmv_balance():
    register = request.get_json()
    #data = pd.read_json(register)
    data = pd.DataFrame([register])
    decision = pmv_balance(data)
    #print(decision)
    #運算失敗會回傳400
    if decision['StatusCode'][0] == 200:
        return jsonify({
                        "StatusCode" : "200",
                        "ac_temp": decision['ac_temp'][0],
                        'ac_mode': decision['ac_mode'][0],
                        'ac_fan': decision['ac_fan'][0],
                        'dehumidifier_state': decision['dehumidifier'][0],
                        'dehumidifier_hum': decision['dehumidifier_hum'][0],
                        'fan_state': decision['fan_state'][0]
                        })
    else:
        return jsonify({"StatusCode" : str(decision['StatusCode'][0])})

if __name__ =='__main__':
    #app.run(host='0.0.0.0', port=5000, debug=False)
    app.run(port=5000, debug=True)
    #app.run(port=5000,debug=False)
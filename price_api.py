from flask import Flask, abort, request, jsonify
from main_price import Price
import pandas as pd
import threading
from check_user_comfort import background_check_loop, chek_user_questionnaire_status
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)


@app.route('/decision_price', methods=['POST'])
def decision_price():
    register = request.get_json()
    data = pd.DataFrame(register)
    decision, predicted, woa_result, total_result = Price(data)
    #運算失敗會回傳400
    if decision['StatusCode'][0] == 200:
        room_id = total_result["room_id"][0]
        decision_remark = total_result["decision_remark"][0]
        best_temp = woa_result["best_temp"][0]
        best_humd = woa_result["best_humd"][0]
        chek_user_questionnaire_status(room_id, decision_remark, best_temp, best_humd)
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
    #app.run(host='0.0.0.0', port=5050, debug=False)
    t = threading.Thread(target=background_check_loop, daemon=True)
    t.start()
    app.run(host='0.0.0.0', port=5000, debug=True)


    
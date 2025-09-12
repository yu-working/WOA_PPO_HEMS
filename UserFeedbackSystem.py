# user_feedback.py
import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from typing import Dict, Optional

class UserFeedbackSystem:
    def __init__(self, save_path: str = 'user_feedback_log.csv'):
        self.csv_path = save_path
        self.model = None

        if os.path.exists(self.csv_path):
            self.feedback_data = pd.read_csv(self.csv_path)
        else:
            self.feedback_data = pd.DataFrame()

    def record_feedback(self,
                        system_name: str,
                        current_state: Dict[str, float],
                        system_state: Dict[str, int],
                        user_state: Dict[str, int]):
        """
        如果 user_state 與 system_state 有差異，記錄使用者回饋
        """
        if system_state != user_state:
            print(f"使用者調整了 {system_name} 給出的建議，紀錄偏好中...")

            feedback_record = {
                'system_name': system_name,
                **current_state,
                **user_state
            }

            new_entry = pd.DataFrame([feedback_record])
            self.feedback_data = pd.concat([self.feedback_data, new_entry], ignore_index=True)

            # 寫入 CSV
            self.feedback_data.to_csv(self.csv_path, index=False)

            # 即時更新模型
            self.train_model()

    def train_model(self):
        """根據目前的 feedback_data 訓練偏好模型"""
        if self.feedback_data.shape[0] < 5:
            print("資料筆數不足，略過訓練。")
            return

        features = ['current_temp', 'current_humidity']
        targets = ['dehumidifier', 'dehumidifier_hum',
                   'ac_temp', 'ac_fan', 'ac_mode', 'fan_state']

        self.model = {}
        for target in targets:
            clf = DecisionTreeClassifier()
            clf.fit(self.feedback_data[features], self.feedback_data[target])
            self.model[target] = clf

    def predict_user_preference(self,
                                 current_temp: float,
                                 current_humidity: float) -> Optional[Dict[str, int]]:
        """使用偏好模型預測使用者喜好設備設定"""
        if not self.model:
            return None

        input_df = pd.DataFrame([{
            'current_temp': current_temp,
            'current_humidity': current_humidity
        }])

        prediction = {}
        for target, model in self.model.items():
            prediction[target] = int(model.predict(input_df)[0])
        if not prediction:
            prediction = {}
        return prediction
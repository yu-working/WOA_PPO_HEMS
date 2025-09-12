import numpy as np
from typing import List, Tuple
import random
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import time
from pythermalcomfort.models import pmv_ppd
import math
from UserFeedbackSystem import UserFeedbackSystem
import os

DEFAULT_PMV_UP = float(os.getenv("DEFAULT_PMV_UP", 0.2))
DEFAULT_PMV_DOWN = float(os.getenv("DEFAULT_PMV_DOWN", -0.2))


# %%
class WhaleOptimizationHEMS:
    def __init__(
        self,
        n_whales: int = 30,
        max_iter: int = 100,
        temp_bounds: Tuple[float, float] = (26.0, 35.0),
        humidity_bounds: Tuple[float, float] = (30.0, 70.0),
        b: float = 1.0,
        a_decrease_factor: float = 2,
        pmv_up: float = 0.15,
        pmv_down: float = -0.15,
        device: list = ["dehumidifier", "ac", "fan"],
    ):
        """
        初始化鯨魚優化演算法用於家庭能源管理系統

        參數:
            n_whales: 搜索代理(鯨魚)的數量
            max_iter: 最大迭代次數
            temp_bounds: 溫度範圍(最小值, 最大值)
            humidity_bounds: 濕度範圍(最小值, 最大值)
            b: 螺旋路徑參數
            a_decrease_factor: 'a'參數的遞減因子
        """
        self.n_whales = n_whales
        self.max_iter = max_iter
        self.temp_bounds = temp_bounds
        self.humidity_bounds = humidity_bounds
        self.n_dims = 2  # 溫度和濕度兩個維度
        self.b = b
        self.a_decrease_factor = a_decrease_factor
        self.pmv_up = pmv_up
        self.pmv_down = pmv_down
        self.device = device
        if isinstance(self.pmv_up, pd.Series):
            self.pmv_up = DEFAULT_PMV_UP
        if isinstance(self.pmv_down, pd.Series):
            self.pmv_down = DEFAULT_PMV_DOWN

    def decision_tree_train(self):
        # 範例訓練資料 (features: 現在溫度、現在濕度、目標溫度、目標濕度)
        # target: [除濕機啟閉, 除濕機濕度, 冷氣溫度, 冷氣風扇, 冷氣模式, 電風扇啟閉]
        df = pd.read_csv("./data/balance_decision_tree_data.csv")
        # 特徵與目標
        X = df[["current_temp", "current_humidity", "target_temp", "target_humidity"]]
        y = df[
            [
                "dehumidifier",
                "dehumidifier_hum",
                "ac_temp",
                "ac_fan",
                "ac_mode",
                "fan_state",
            ]
        ]

        # 用決策樹分類器訓練，每個輸出一個樹
        trees = {}
        for col in y.columns:
            clf = DecisionTreeClassifier()
            clf.fit(X, y[col])
            trees[col] = clf
        return trees

    def predict_control(
        self, trees, current_temp, current_humidity, target_temp, target_humidity
    ):
        input_data = pd.DataFrame(
            [
                {
                    "current_temp": current_temp,
                    "current_humidity": current_humidity,
                    "target_temp": target_temp,
                    "target_humidity": target_humidity,
                }
            ]
        )
        output = {}
        for col, tree in trees.items():
            output[col] = int(tree.predict(input_data)[0])
        if int(output["dehumidifier"] == 1) & int(output["dehumidifier_hum"] == 0):
            output["dehumidifier"] = 0
        if int(output["ac_mode"] == 0):
            output["ac_temp"] = 0

        return output

    def initialize_population(self) -> np.ndarray:
        """初始化鯨魚群體在範圍內的隨機位置"""

        # 初始化溫度
        population_temp = []
        for i in range(self.n_whales):
            population_temp.append(
                random.randint(self.temp_bounds[0], self.temp_bounds[1])
            )
        population_temp = np.array(population_temp)

        # 初始化濕度
        population_humd = []
        for i in range(self.n_whales):
            population_humd.append(
                random.randint(self.humidity_bounds[0], self.humidity_bounds[1])
            )
        population_humd = np.array(population_humd)

        population = np.column_stack([population_temp, population_humd])
        return population

    def calculate_power(
        self,
        dehumidifier_humidity,
        indoor_humidity,
        dehumidifier_on,
        fan_on,
        ac_temperature,
        indoor_temp,
        ac_mode,
        ac_fan_speed,
    ):
        # 計算總功耗
        dehumidifier_power = 120  # 除濕機功率
        fan_power = 60  # 電扇功率
        ac_base_power = 1350  # 冷氣基礎功率
        ac_mode_weight = {0: 0.5, 1: 1, 2: 0.9}  # 冷氣運轉模式係數 0:送風 1:冷氣 2:舒眠
        ac_fan_weight = {0: 0.9, 1: 1, 2: 0.95}  # 風速係數 0:低速 1:高速 2:自動

        # 計算除濕機功耗
        dehumidifier_consumption = (
            dehumidifier_power * max(0, (indoor_humidity - dehumidifier_humidity) / 100)
            if dehumidifier_on
            else 0
        )

        # 計算電扇功耗
        fan_consumption = fan_power if fan_on else 0

        # 計算冷氣功耗
        if ac_mode_weight == 1:
            ac_consumption = (
                ac_base_power
                * ac_mode_weight[ac_mode]
                * (1 + 0.1 * max(0, indoor_temp - ac_temperature))
                * ac_fan_weight[ac_fan_speed]
            )
        else:
            ac_consumption = (
                ac_base_power * ac_mode_weight[ac_mode] * ac_fan_weight[ac_fan_speed]
            )

        return dehumidifier_consumption + fan_consumption + ac_consumption

    def fitness_function(self, temp: float, humidity: float, trees) -> float:
        """
        計算適應度值，基於舒適度和能源消耗

        參數:
            temp: 溫度值
            humidity: 濕度值
            outdoor_temp: 室外溫度
            outdoor_hum: 室外濕度

        返回:
            適應度值(越低越好)
        """
        # 舒適度參數(理想值)
        ideal_temp_high = 30.0
        ideal_humidity_high = 60.0
        ideal_temp_low = 24.0
        ideal_humidity_low = 40.0

        # 計算濕度權重
        if humidity < ideal_humidity_low:
            w_humidity = 1 - (1 / abs(humidity - ideal_humidity_low))
        elif humidity > ideal_humidity_low:
            w_humidity = 1 - (1 / abs(humidity - ideal_humidity_low))
        elif (humidity > ideal_humidity_high) or (humidity < ideal_humidity_low):
            w_humidity = 1
        else:
            w_humidity = 0.5

        # 計算溫度權重

        if temp < ideal_temp_low:
            w_temp = 1 - (1 / abs(temp - ideal_temp_low))
        elif temp > ideal_temp_low:
            w_temp = 1 - (1 / abs(temp - ideal_temp_low))
        elif (temp > ideal_temp_high) or (temp < ideal_temp_low):
            w_temp = 1
        else:
            w_temp = 0.5

        # 計算舒適度偏差
        if temp < ideal_temp_low:
            temp_deviation = abs(temp - ideal_temp_low)
        elif temp > ideal_temp_high:
            temp_deviation = abs(temp - ideal_temp_high)
        else:
            temp_deviation = 1

        if humidity < ideal_humidity_low:
            humidity_deviation = abs(humidity - ideal_humidity_low)
        elif humidity > ideal_humidity_high:
            humidity_deviation = abs(humidity - ideal_humidity_high)
        else:
            humidity_deviation = 1

        # 預測設備狀態

        # device_state = self.predict_control(trees, temp, humidity, temp, humidity)
        # device_state = pd.DataFrame(device_state, columns=['dehumidifier', 'dehumidifier_hum', 'ac_temp', 'ac_fan', 'ac_mode', 'fan_state'], index=[0])

        if "dehumidifier" in self.device:  # 未持有除濕機時不運算
            pass
        else:
            device_state.loc[device_state.index[0], "dehumidifier"] = 0  # 除濕機開關
            device_state.loc[device_state.index[0], "dehumidifier_hum"] = (
                0  # 除濕機設定濕度 40-70%
            )

        # 計算濕度變化的潛熱
        # m = ((217*(humidity/100)*6.112*math.exp(17.62*temp/(temp+243.12)))/(temp+273.15)-(217*(device_state.iloc[0, 1]/100)*6.112*math.exp(17.62*temp/(temp+243.12))))

        # 計算PMV值
        pmv = pmv_ppd(
            tdb=temp, tr=temp, vr=0.25, rh=humidity, met=1, clo=0.5, limit_inputs=False
        )
        # PMV值超出舒適範圍時給予懲罰

        if float(self.pmv_down) <= pmv["pmv"] <= float(self.pmv_up):
            pass
        else:
            pmv["pmv"] = 100000  # 較大的懲罰

        try:
            tpmv = 3
            if abs(abs(float(self.pmv_up)) - abs(pmv["pmv"])) < abs(
                abs(float(self.pmv_up)) - abs(tpmv)
            ):
                tpmv = pmv["pmv"]
                user_pmv = abs(abs(float(self.pmv_up)) - abs(pmv["pmv"]))
            else:
                user_pmv = 100000  # 較大的懲罰
        except:
            tpmv = 3
            if abs(abs(float(self.pmv_up)) - abs(pmv["pmv"])) < abs(
                abs(float(self.pmv_up)) - abs(tpmv)
            ):
                tpmv = pmv["pmv"]
                user_pmv = abs(float(self.pmv_up) - abs(pmv["pmv"]))
            else:
                user_pmv = 100000  # 較大的懲罰

        # 計算最終適應度值
        # fitness = abs((temp_deviation*1.005*w_temp+humidity_deviation*(m)*2260*w_humidity)) * abs(pmv['pmv']) * user_pmv
        # fitness = abs((temp_deviation*1.005*w_temp+humidity_deviation*(m)*2260*w_humidity)) * user_pmv
        # fitness = abs(pmv['pmv']) # 配合使用者pmv上界 嘗試靠近使用者舒適區間上界
        fitness = user_pmv  # 配合使用者pmv上界 嘗試靠近使用者舒適區間上界
        return fitness

    def apply_bounds(self, position: np.ndarray) -> np.ndarray:
        """確保位置在定義的範圍內"""
        position[0] = np.clip(position[0], self.temp_bounds[0], self.temp_bounds[1])
        position[1] = np.clip(
            position[1], self.humidity_bounds[0], self.humidity_bounds[1]
        )
        return position

    def change(self, device_date, indoor_data):
        """模擬室內環境變化"""
        target_humidity = device_date["dehumidifier_hum"].iloc[0]
        target_temp = device_date["ac_temp"].iloc[0]
        base_area = 25  # 參考面積 25 m^2
        cooling_per_kW = 2  # 設每kW每小時降溫 2
        area = 20  # 暫定空間面積為20 m^2
        cooling_capacity = (
            2.8 if device_date["ac_mode"].iloc[0] == 1 else 0
        )  # 暫定冷氣能力 如果為冷氣模式為 2.8 kW 送風模式為 0
        area_factor = base_area / area
        ac_temp_drop = cooling_capacity * cooling_per_kW * area_factor
        temp_final = (
            max(indoor_data[0] - ac_temp_drop, target_temp)
            if device_date["ac_mode"].iloc[0] == 1
            else min(indoor_data[0] + 2, 35)
        )
        ## 濕度
        ac_humidity_drop = min((cooling_capacity / area) * 100, 30)
        dehumidifier_humidity_drop = (
            7.1 * 0.9 if device_date["dehumidifier"].iloc[0] == 1 else 0
        )
        rh_final = (
            max(
                indoor_data[1] - ac_humidity_drop - dehumidifier_humidity_drop,
                target_humidity,
            )
            if device_date["dehumidifier"].iloc[0] == 1
            else max(indoor_data[1] - ac_humidity_drop - dehumidifier_humidity_drop, 45)
        )

        # 更新室內狀態
        new_indoor_data = np.array([temp_final, rh_final]).reshape(1, 2)
        return new_indoor_data

    def optimize(self, indoor_data, trees) -> Tuple[np.ndarray, float, List[float]]:
        """
        執行鯨魚優化演算法

        返回:
            best_position: 最佳溫度和濕度值
            best_fitness: 找到的最佳適應度值
            fitness_history: 每次迭代的最佳適應度值列表
        """
        # 初始化群體
        population = indoor_data.copy()
        # 初始化最佳解
        fitness_values = np.array(
            [self.fitness_function(pos[0], pos[1], trees) for pos in population]
        )
        best_whale_idx = np.argmin(fitness_values)
        best_position = population[best_whale_idx].copy()
        best_fitness = fitness_values[best_whale_idx]

        # 優化歷史記錄
        fitness_history = [best_fitness]

        # 主迴圈
        for iteration in range(self.max_iter):
            # 更新a參數
            a = 2 - iteration * (2 / self.max_iter)

            # 更新每隻鯨魚的位置
            for i in range(self.n_whales):
                # 隨機參數
                r = random.random()
                A = 2 * a * r - a
                C = 2 * r
                l = random.uniform(-1, 1)
                p = random.random()

                if p < 0.5:
                    # 包圍獵物或搜索獵物
                    if abs(A) < 1:
                        # 包圍獵物
                        D = abs(C * best_position - population[i])
                        new_position = best_position - A * D
                    else:
                        # 搜索獵物
                        random_whale = population[random.randint(0, self.n_whales - 1)]
                        D = abs(C * random_whale - population[i])
                        new_position = random_whale - A * D
                else:
                    # 螺旋更新位置
                    D = abs(best_position - population[i])
                    spiral = (
                        D * np.exp(self.b * l) * np.cos(2 * np.pi * l) + best_position
                    )
                    new_position = spiral

                # 應用邊界並更新位置
                population[i] = self.apply_bounds(new_position)

                # 如果需要，更新最佳解
                current_fitness = self.fitness_function(
                    population[i][0], population[i][1], trees
                )
                if current_fitness < best_fitness:
                    best_position = population[i].copy()
                    best_fitness = current_fitness

            fitness_history.append(best_fitness)

        # 根據pmv變化決定使用哪個位置進行預測

        indoor_pmv = pmv_ppd(
            tdb=indoor_data[-1][0],
            tr=indoor_data[-1][0],
            vr=0.25,
            rh=indoor_data[-1][1],
            met=1,
            clo=0.5,
            limit_inputs=False,
        )

        best_pmv = pmv_ppd(
            tdb=best_position[0],
            tr=best_position[0],
            vr=0.25,
            rh=best_position[1],
            met=1,
            clo=0.5,
            limit_inputs=False,
        )
        # 插入Decision Tree
        if abs(self.pmv_up - indoor_pmv["pmv"]) < abs(self.pmv_up - best_pmv["pmv"]):
            device_state = self.predict_control(
                trees,
                indoor_data[-1][0],
                indoor_data[-1][1],
                indoor_data[-1][0],
                indoor_data[-1][1],
            )
        else:
            device_state = self.predict_control(
                trees,
                indoor_data[-1][0],
                indoor_data[-1][1],
                best_position[0],
                best_position[1],
            )
        device_state = pd.DataFrame([device_state])

        if "dehumidifier" in self.device:  # 未持有除濕機時不運算
            pass
        else:
            device_state.loc[device_state.index[0], "dehumidifier"] = 0  # 除濕機開關
            device_state.loc[device_state.index[0], "dehumidifier_hum"] = (
                0  # 除濕機設定濕度 40-70%
            )

        if "fan" in self.device:  # 未持有電扇時不運算
            pass
        else:
            device_state.loc[device_state.index[0], "fan_state"] = 0

        energy_consumption = self.calculate_power(
            device_state["dehumidifier_hum"][0],
            indoor_data[-1][1],
            device_state["dehumidifier"][0],
            device_state["fan_state"][0],
            device_state["ac_temp"][0],
            indoor_data[-1][0],
            device_state["ac_mode"][0],
            device_state["ac_fan"][0],
        )
        new_indoor_data = self.change(device_state, indoor_data[-1])
        pmv = pmv_ppd(
            tdb=new_indoor_data[0][0],
            tr=new_indoor_data[0][0],
            vr=0.25,
            rh=new_indoor_data[0][1],
            met=1,
            clo=0.5,
            limit_inputs=False,
        )

        return (
            best_position,
            best_fitness,
            fitness_history,
            device_state,
            energy_consumption,
            pmv["pmv"],
            new_indoor_data,
        )


# %%
if __name__ == "__main__":
    start = time.time()
    pmv_ul_ll = pd.read_csv("./config/pmv_ul_ll.csv")
    room_id = "123gmail.com"
    # 初始化優化器
    woa = WhaleOptimizationHEMS(
        n_whales=24,
        max_iter=50,
        temp_bounds=(26.0, 31.0),
        humidity_bounds=(40.0, 70.0),
        pmv_up=pmv_ul_ll.loc[pmv_ul_ll["room_id"] == room_id]["pmv_ul"],
        pmv_down=pmv_ul_ll.loc[pmv_ul_ll["room_id"] == room_id]["pmv_ll"],
    )
    feedback_system = UserFeedbackSystem()
    # %%
    Result = pd.DataFrame()
    fit = []
    Fitness_history = []
    history_indoor, Cost, Pmv = [], [], []
    Ec, Esr = [], []
    trees = woa.decision_tree_train()
    # 執行優化
    for i in range(7):
        if i == 0:
            indoor_data = woa.initialize_population()
            indoor_data[-1][0], indoor_data[-1][1] = 30, 80
        else:
            indoor_data = np.delete(indoor_data, 0, 0)
        history_indoor.append(indoor_data)
        (
            best_position,
            best_fitness,
            fitness_history,
            device_state,
            cost,
            pmv,
            new_indoor_data,
        ) = woa.optimize(indoor_data, trees)
        Pmv.append(pmv)
        Cost.append(cost / 1000)
        env = pd.DataFrame(indoor_data[-1]).T
        env.columns = ["env_temp", "env_humd"]
        fit.append(best_fitness)
        Fitness_history.append(fitness_history)
        best_position = pd.DataFrame(best_position).T
        best_position.columns = ["best_temp", "best_humd"]
        result = pd.concat([env, best_position, device_state], axis=1)
        Result = Result.append(result, ignore_index=True)

        # 比較節能效果
        ec = woa.calculate_power(
            40, Result["env_humd"].iloc[-1], 1, 1, 26, Result["env_temp"].iloc[-1], 1, 1
        )
        Ec.append(ec / 1000)
        esr = (1 - (cost / ec)) * 100
        Esr.append(esr)

        # user feedback
        system_name = room_id
        env_state = {
            "current_temp": float(indoor_data[-1][0]),
            "current_humidity": float(indoor_data[-1][1]),
        }
        system_state = device_state.iloc[0].to_dict()
        user_state = {
            "dehumidifier": 1,
            "dehumidifier_hum": 60,
            "ac_temp": 26,
            "ac_fan": 1,
            "ac_mode": 0,
            "fan_state": 1,
        }
        # 記錄回饋（如果有不同）
        feedback_system.record_feedback(
            system_name, env_state, system_state, user_state
        )
        # 預測使用者偏好（之後可使用）
        predicted = feedback_system.predict_user_preference(
            current_temp=env_state["current_temp"],
            current_humidity=env_state["current_humidity"],
        )
        print("預測使用者偏好：", predicted)

        # %%
        # 更新室內外環境狀態
        indoor_data = np.append(indoor_data, new_indoor_data, 0)

    # 處理結果數據
    Pmv = pd.DataFrame(Pmv, columns=["pmv"])
    Result = pd.concat([Result, Pmv], axis=1)
    Result["ac_temp"] = np.where(Result["ac_mode"] == 0, "-", Result["ac_temp"])
    Result["ac_temp"] = np.where(Result["ac_mode"] == 2, "-", Result["ac_temp"])
    Result["dehumidifier_hum"] = Result["dehumidifier_hum"].astype(int)
    Result["dehumidifier_hum"] = round(Result["dehumidifier_hum"] / 5) * 5
    Result["dehumidifier_hum"] = np.where(
        Result["dehumidifier"] == 0, "-", Result["dehumidifier_hum"]
    )
    end = time.time()
    print(end - start)

    # 計算節能效果
    Result["cost"] = Cost
    Result["ec"] = Ec
    Result["energy_saving_rate"] = Esr
    Result.to_csv("./WOA_TREE能耗與舒適度加權平衡參數測試結果.csv")

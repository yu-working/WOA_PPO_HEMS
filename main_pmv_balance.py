import pandas as pd
from WOA_pmv_balance_online import WhaleOptimizationHEMS
from pythermalcomfort.models import pmv_ppd
import math
from ppo_pmv_balance_online import PPOAgent, HEMSEnvironment
from ppo_pmv_balance_retrain import ppo_retrain
import numpy as np
import time
import db_utility as dbu
import datetime as dtime
import logging
import sys
import uuid
import os
from UserFeedbackSystem import UserFeedbackSystem

start = time.time()
#%% 能耗計算
def calculate_power(dehumidifier_humidity, indoor_humidity,
                    dehumidifier_on, fan_on, ac_temperature, indoor_temp,
                    ac_mode, ac_fan_speed):
    # 計算總功耗
    dehumidifier_power = 120  # 除濕機功率
    fan_power = 60            # 電扇功率
    ac_base_power = 1350      # 冷氣基礎功率
    #ac_fan_speed_coeff = 0.1  # 冷氣風速功率係數
    ac_mode_weight = {'fan':0.5, 'cool':1}        # 冷氣運轉模式係數 0:送風 1:冷氣 2:舒眠
    ac_fan_weight = {'low':0.6, 'high':1, 'auto':0.8} # 風速係數 0:低速 1:高速 2:自動
    

    # 計算除濕機功耗
    if dehumidifier_on == 'on':
        dehumidifier_consumption = dehumidifier_power * max(0, (indoor_humidity - dehumidifier_humidity)/100)
    else: dehumidifier_consumption = 0

    # 計算電扇功耗
    if fan_on == 'on':
        fan_consumption = fan_power
    else: fan_consumption = 0

    # 計算冷氣功耗
    if ac_mode_weight[ac_mode] == 1:
        ac_consumption = ac_base_power * ac_mode_weight[ac_mode] * (1 + 0.1 * max(0, indoor_temp-ac_temperature)) * ac_fan_weight[ac_fan_speed] 
    else: 
        ac_consumption = ac_base_power * ac_mode_weight[ac_mode] * ac_fan_weight[ac_fan_speed] 

    return dehumidifier_consumption + fan_consumption + ac_consumption
#%% IoT數據讀取 (DB連線)
def pmv_balance(data):
    try:
        data = pd.read_csv('./data/data-1743586080241.csv') #sample檔
        #%% log檔製做
        logging.basicConfig(
            filename='./log/main_decision_pmv_balance_'+dtime.datetime.now().strftime('%Y-%m-%d %H%M%S')+'.log',
            level=logging.INFO,
            force=True,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logger = logging.getLogger()
        #%% IoT數據處理
        logger.info('Phase1: IoT資料萃取')
        room_id = list(data.groupby('room_id')) #將同ID資料整理出並以list包裝
        if len(room_id) != 1: # 確認只有一個room_id
            Result = pd.DataFrame([600], columns=['StatusCode'])
            return Result
        #room_id = [elt[1] for elt in room_id] #將ID資料取出
        device_lst = list(data.groupby('appliance_name')) #將同電器資料整理出並以list包裝
        device_name = []
        for name in device_lst:
            device_name.append(name[0]) #將電器名稱取出
        device_lst = [elt[1] for elt in device_lst] #將電器資料取出
        
        device = []
        device_parameter = []
        for i in device_lst:
            a = list(i.groupby('capability_name')) #將電器的參數資料整理出並以list包裝
            device.append([elt[1] for elt in a]) #將電器的參數資料取出
            device_parameter.append([])
            for s in a:
                device_parameter[-1].append(s[0]) #將電器的參數名稱取出
         
        Device = {}
        for i in range(len(device_name)):
            Device[device_name[i]] = device[i] #將萃取完的資料用電器名稱整理成dict
        
        param_iter = iter(device_parameter)
        for category, df_list in Device.items():
            names = next(param_iter)  # 取出對應的一組名字
            Device[category] = {name: df for name, df in zip(names, df_list)} #將電器參數名稱也賦予上去
    
        #%% 實際資料
        logger.info('Phase2: 電器資料萃取')
        try:
            ac_parameter = pd.concat([Device['air_conditioner']['cfg_fan_level']['record_value'].reset_index(drop=True),
                                  Device['air_conditioner']['cfg_mode']['record_value'].reset_index(drop=True), 
                                  Device['air_conditioner']['cfg_temperature']['record_value'].reset_index(drop=True),
                                  Device['air_conditioner']['cfg_temperature']['device_signature'].reset_index(drop=True)], axis=1) # 智慧冷氣參數取出
            ac_parameter.columns = ['cfg_fan_level', 'cfg_mode', 'cfg_temperature', 'device_signature'] #欄位再命名
            operation_parameter = pd.concat([Device['air_conditioner']['cfg_temperature']['operation_direction'].reset_index(drop=True),
                                             Device['air_conditioner']['cfg_mode']['operation_direction'].reset_index(drop=True),
                                             Device['air_conditioner']['cfg_fan_level']['operation_direction'].reset_index(drop=True), 
                                             Device['dehumidifier']['cfg_power']['operation_direction'].reset_index(drop=True),
                                             Device['dehumidifier']['cfg_humidity']['operation_direction'].reset_index(drop=True),
                                             Device['ac_outlet']['cfg_power']['operation_direction'].reset_index(drop=True)])
            user_source = pd.concat([Device['air_conditioner']['cfg_temperature']['source'].reset_index(drop=True),
                                             Device['air_conditioner']['cfg_mode']['source'].reset_index(drop=True),
                                             Device['air_conditioner']['cfg_fan_level']['source'].reset_index(drop=True), 
                                             Device['dehumidifier']['cfg_power']['source'].reset_index(drop=True),
                                             Device['dehumidifier']['cfg_humidity']['source'].reset_index(drop=True),
                                             Device['ac_outlet']['cfg_power']['source'].reset_index(drop=True)])
        except: 
            remote_parameter = pd.concat([Device['remote']['cfg_fan_level']['record_value'].reset_index(drop=True),
                              Device['remote']['cfg_mode']['record_value'].reset_index(drop=True), 
                              Device['remote']['cfg_temperature']['record_value'].reset_index(drop=True),
                              Device['remote']['cfg_temperature']['device_signature'].reset_index(drop=True)], axis=1) #紅外線遙控器冷氣參數取出
            remote_parameter.columns = ['cfg_fan_level', 'cfg_mode', 'cfg_temperature', 'device_signature'] #欄位再命名
            operation_parameter = pd.concat([Device['remote']['cfg_temperature']['operation_direction'].reset_index(drop=True),
                                             Device['remote']['cfg_mode']['operation_direction'].reset_index(drop=True),
                                             Device['remote']['cfg_fan_level']['operation_direction'].reset_index(drop=True), 
                                             Device['dehumidifier']['cfg_power']['operation_direction'].reset_index(drop=True),
                                             Device['dehumidifier']['cfg_humidity']['operation_direction'].reset_index(drop=True),
                                             Device['ac_outlet']['cfg_power']['operation_direction'].reset_index(drop=True)])
            user_source = pd.concat([Device['remote']['cfg_temperature']['source'].reset_index(drop=True),
                                             Device['remote']['cfg_mode']['source'].reset_index(drop=True),
                                             Device['remote']['cfg_fan_level']['source'].reset_index(drop=True), 
                                             Device['dehumidifier']['cfg_power']['source'].reset_index(drop=True),
                                             Device['dehumidifier']['cfg_humidity']['source'].reset_index(drop=True),
                                             Device['ac_outlet']['cfg_power']['source'].reset_index(drop=True)])
        dehumidifier_parameter = pd.concat([Device['dehumidifier']['cfg_humidity']['record_value'].reset_index(drop=True),
                                  Device['dehumidifier']['cfg_power']['record_value'].reset_index(drop=True),
                                  Device['dehumidifier']['cfg_power']['device_signature'].reset_index(drop=True)], axis=1) #除濕機參數取出
        dehumidifier_parameter.columns = ['cfg_humidity', 'cfg_power', 'device_signature']
        fan_parameter = pd.DataFrame(Device['ac_outlet']['cfg_power']['record_value'].reset_index(drop=True)) #風扇參數取出
        fan_parameter = pd.concat([Device['ac_outlet']['cfg_power']['record_value'].reset_index(drop=True),
                                  Device['ac_outlet']['cfg_power']['device_signature'].reset_index(drop=True)], axis=1) #風扇參數取出
        fan_parameter.columns = ['cfg_power', 'device_signature']
        # 資訊轉換 -- 現階段只嘗試冷氣與送風 因此非送風模式的全強制改成冷氣
        if ac_parameter['cfg_mode'][0] != 'fan':
            ac_parameter['cfg_mode'][0] = 'cool'
        # 資訊轉換 -- 現階段只嘗試low, high, auto 因此非low與auto的全強制改成high
        if ac_parameter['cfg_fan_level'][0] != 'low' and ac_parameter['cfg_fan_level'][0] != 'auto':
            ac_parameter['cfg_fan_level'][0] = 'high'
        # 移除異常值
        Device['sensor']['op_humidity'] = Device['sensor']['op_humidity'][~((Device['sensor']['op_humidity']['record_value'] == 'unknown') | (Device['sensor']['op_humidity']['record_value'] == 'unavailable'))]
        Device['sensor']['op_temperature'] = Device['sensor']['op_temperature'][~((Device['sensor']['op_temperature']['record_value'] == 'unknown') | (Device['sensor']['op_temperature']['record_value'] == 'unavailable'))]
        #%% 初始用電量計算
        logger.info('Phase3: 初始用電量計算')
        indoor_data = np.array([(Device['sensor']['op_temperature']['record_value'].astype(float)), Device['sensor']['op_humidity']['record_value'].astype(float)]).T #IoT溫溼度擷取
        try:
            cost_init = calculate_power(dehumidifier_humidity=float(dehumidifier_parameter['cfg_humidity'][0]),
                                        indoor_humidity=indoor_data[0][1], dehumidifier_on=dehumidifier_parameter['cfg_power'][0],
                                        fan_on=fan_parameter['cfg_power'][0], ac_temperature=float(ac_parameter['cfg_temperature'][0]), 
                                        indoor_temp=indoor_data[0][0], ac_mode=ac_parameter['cfg_mode'][0], 
                                        ac_fan_speed=ac_parameter['cfg_fan_level'][0])
        except:
            cost_init = calculate_power(dehumidifier_humidity=float(dehumidifier_parameter['cfg_humidity'][0]),
                                        indoor_humidity=indoor_data[0][1], dehumidifier_on=dehumidifier_parameter['cfg_power'][0],
                                        fan_on=fan_parameter['cfg_power'][0], ac_temperature=float(remote_parameter['cfg_temperature'][0]), 
                                        indoor_temp=indoor_data[0][0], ac_mode=remote_parameter['cfg_mode'][0], 
                                        ac_fan_speed=remote_parameter['cfg_fan_level'][0])
        
        #%% PPO再訓練 -- 若過往24小時有手動操作過抑或初次使用抓不到對應room_id的ppo模型時進行
        if ('write' in operation_parameter.values and 'user' in user_source.values) or os.path.isfile('./config/ppo_pmv_balance_'+room_id[0][0]+'.pt') == False:
            ppo_retrain(indoor_data, room_id=room_id[0][0])
        #%% WOA+PPO環境讀取
        logger.info('Phase4: 環境設置')
        pmv_ul_ll = pd.read_csv('./config/pmv_ul_ll.csv')
        woa = WhaleOptimizationHEMS(
        n_whales=len(indoor_data),
        max_iter=50,
        temp_bounds=(26.0, 33.0),
        humidity_bounds=(60.0, 85.0),
        pmv_up = pmv_ul_ll.loc[pmv_ul_ll['room_id'] == room_id[0][0]]['pmv_ul'],
        pmv_down = pmv_ul_ll.loc[pmv_ul_ll['room_id'] == room_id[0][0]]['pmv_ll']
        )
        Env = HEMSEnvironment(
            pmv_up = pmv_ul_ll.loc[pmv_ul_ll['room_id'] == room_id[0][0]]['pmv_ul'],
            pmv_down = pmv_ul_ll.loc[pmv_ul_ll['room_id'] == room_id[0][0]]['pmv_ll'])
        agent = PPOAgent(state_dim=2, action_dim=4)
        agent.load_model(room_id=room_id[0][0])
        # DecisionTree訓練
        trees = woa.decision_tree_train()
        # 使用者偏好回饋系統
        feedback_system = UserFeedbackSystem()
        isa = pd.read_csv('./config/紅外線遙控器冷氣調控指令集.csv')
        #%% 環境初始數據
        #indoor_data = woa.initialize_population() #內建環境溫濕度生成
        #indoor_data[-1][0], indoor_data[-1][1] = 30, 70
        fit = []
        Fitness_history = []
        Cost_woa = 0
        Cost_ppo = 0
        #device_state_init = []
        state = Env.reset(test=True, indoor_temp=indoor_data[0][0], indoor_humidity=indoor_data[0][1])
        #%%
        logger.info('Phase5: 決策計算')
        # 執行優化
        # 當決策預估節省能耗<14% 時則重複執行
        count = 0
        while (Cost_woa/1000) < cost_init or (Cost_ppo/1000) < cost_init or (Cost_ppo == 0 and Cost_woa == 0):
            count += 1
            Result_woa = pd.DataFrame()
            #best_position, best_fitness, fitness_history, device_state, energy_consumption, pmv['pmv'], new_indoor_data
            best_position, best_fitness, fitness_history, device_state, cost_woa, pmv_woa, _ = woa.optimize(indoor_data, trees)
            Cost_woa+=cost_woa
            env = pd.DataFrame(indoor_data[-1]).T
            env.columns = ['env_temp', 'env_humd']
            fit.append(best_fitness)
            Fitness_history.append(fitness_history)
            best_position = pd.DataFrame(best_position).T
            best_position.columns = ['best_temp', 'best_humd']
            result = pd.concat([env, best_position, device_state], axis=1)
            Result_woa = Result_woa.append(result, ignore_index=True)
            # 處理結果數據
            Pmv_woa = pd.DataFrame([pmv_woa], columns=['pmv'])
            Result_woa = pd.concat([Result_woa, Pmv_woa], axis=1)    
            Result_woa['ac_temp'] = np.where(Result_woa['ac_mode'] == 0, '0', Result_woa['ac_temp'])
            Result_woa['ac_temp'] = np.where(Result_woa['ac_mode'] == 2, '0', Result_woa['ac_temp'])
            Result_woa['dehumidifier_hum']  = Result_woa['dehumidifier_hum'].astype(int)
            Result_woa['dehumidifier_hum'] = round(Result_woa['dehumidifier_hum'] / 5) * 5
            Result_woa['dehumidifier_hum'] = np.where(Result_woa['dehumidifier'] == 0, '0', Result_woa['dehumidifier_hum'])
            #Result_woa.to_csv('C:/Users/hankli/Documents/114計劃相關/調控參數/main_woa能耗與舒適度加權平衡參數測試結果.csv')
            # PPO 決策生成
            env_state_test = pd.DataFrame()
            device_state_test = pd.DataFrame()
            
            
            # 選擇動作
            action, _ = agent.select_action(state)
            # 執行動作
            next_state, reward, done, env_state, device_state, pmv_ppo, cost_ppo = Env.step(action, isa)
            Cost_ppo+=cost_ppo
            env_state_test = env_state_test.append(pd.DataFrame(env_state).T, ignore_index=True)
            device_state_test = device_state_test.append(pd.DataFrame(device_state).T, ignore_index=True)
                
            # 儲存決策結果
            Pmv_ppo = pd.DataFrame([[pmv_ppo]], columns=['pmv'])
            Result_ppo = pd.concat([env_state_test, device_state_test, Pmv_ppo], axis = 1)
            Result_ppo.columns = ['env_temp', 'env_humd', 'dehumidifier', 'dehumidifier_hum',
                     'ac_temp', 'ac_fan', 'ac_mode', 'fan_state', 'pmv']
            Result_ppo['dehumidifier_hum']  = Result_ppo['dehumidifier_hum'].astype(int)
            Result_ppo['dehumidifier_hum'] = round(Result_ppo['dehumidifier_hum'] / 5) * 5
            Result_ppo['dehumidifier_hum'] = np.where(Result_ppo['dehumidifier'] == 0, '0', Result_ppo['dehumidifier_hum'])
            #Result_ppo.to_csv('C:/Users/hankli/Documents/114計劃相關/調控參數/main_ppo能源與舒適度加權參數應用結果.csv')
           # 判斷是否達成節能
            if (Cost_woa/1000) < cost_init or (Cost_ppo/1000) < cost_init \
                 and (Result_ppo['pmv'][0] < 1 or Result_woa['pmv'][0] < 1) :
                break
            elif count == 4:
                break
            else:
                Cost_woa = 0
                Cost_ppo = 0
        end = time.time()
        print(end-start)
        # 判斷使用的決策
        if Cost_woa > Cost_ppo:
            Result = Result_woa
            alg = 'woa'
        elif Cost_woa < Cost_ppo:
            Result = Result_ppo
            alg = 'ppo'
        elif Cost_woa == Cost_ppo:
            if abs(Result_ppo['pmv'][0]) < abs(Result_woa['pmv'][0]):
                Result = Result_ppo
                alg = 'ppo'
            else: 
                Result = Result_woa
                alg = 'woa'
        
        #使用者偏好
        current_state = {
            'current_temp' : Result.iloc[0, 0],
            'current_humidity': Result.iloc[0, 1]
        }
        system_state = Result[['dehumidifier', 'dehumidifier_hum', 'ac_temp', 'ac_fan', 'ac_mode', 'fan_state']].iloc[0].to_dict()
        user_state = {
            'dehumidifier': 1,
            'dehumidifier_hum': 60,
            'ac_temp': 26,
            'ac_fan': 1,
            'ac_mode': 0,
            'fan_state': 1
        }
        feedback_system.record_feedback(room_id[0][0], current_state, system_state, user_state)
        # 預測使用者偏好（之後可使用）
        predicted = feedback_system.predict_user_preference(current_state['current_temp'], current_state['current_humidity'])

        #%% 數值轉換
        logger.info('Phase6: 數值轉換')
        Result['dehumidifier'] = np.where((Result['dehumidifier'] == 0) | (Result['dehumidifier'] == '0'), 'off', Result['dehumidifier'])
        Result['dehumidifier'] = np.where((Result['dehumidifier'] == 1) | (Result['dehumidifier'] == '1'), 'on', Result['dehumidifier'])
        Result['fan_state'] = np.where((Result['fan_state'] == 0) | (Result['fan_state'] == '0'), 'off', Result['fan_state'])
        Result['fan_state'] = np.where((Result['fan_state'] == 1) | (Result['fan_state'] == '1'), 'on', Result['fan_state'])
        Result['ac_mode'] = np.where((Result['ac_mode'] == 1) | (Result['ac_mode'] == '1'), 'cool', Result['ac_mode'])
        Result['ac_mode'] = np.where((Result['ac_mode'] == 0) | (Result['ac_mode'] == '0'), 'fan', Result['ac_mode'])
        Result['ac_fan'] = np.where((Result['ac_fan'] == 0) | (Result['ac_fan'] == '0'), 'low', Result['ac_fan'])
        Result['ac_fan'] = np.where((Result['ac_fan'] == 1) | (Result['ac_fan'] == '1'), 'high', Result['ac_fan'])
        Result['ac_fan'] = np.where((Result['ac_fan'] == 2) | (Result['ac_fan'] == '2'), 'auto', Result['ac_fan'])
        
        
        try:
            Result['ac_temp'] = np.where(Result['ac_temp'] == '-', '0', Result['ac_temp'])
        except: pass
        try:
            Result = Result.drop(['env_temp', 'env_humd', 'best_temp', 'best_humd'], axis=1)
        except:
            Result = Result.drop(['env_temp', 'env_humd'], axis=1)

        # 使用者偏好數值轉換  
        try:  
            predicted['dehumidifier'] = str(predicted['dehumidifier']).replace('0', 'off')
            predicted['dehumidifier'] = str(predicted['dehumidifier']).replace('1', 'on')
            predicted['fan_state'] = str(predicted['fan_state']).replace('0', 'off')
            predicted['fan_state'] = str(predicted['fan_state']).replace('1', 'on')
            predicted['ac_mode'] = str(predicted['ac_mode']).replace('1', 'cool')
            predicted['ac_mode'] = str(predicted['ac_mode']).replace('0', 'fan')
            predicted['ac_fan'] = str(predicted['ac_fan']).replace('0', 'low')
            predicted['ac_fan'] = str(predicted['ac_fan']).replace('1', 'high')
            predicted['ac_fan'] = str(predicted['ac_fan']).replace('2', 'auto')
        except: pass
        Result['StatusCode'] = 200
        #%% 決策insert到DB -- 此段現階段設計不使用
        # 決策整理成插入格式
        logger.info('Phase7: 格式整理')
        UUID = uuid.uuid4().hex
        decision = pd.DataFrame(columns = ['room_id','decision_remark', 'appliance_name', 'capability_name', 'record_value'])
        decision.loc[0] = [room_id[0][0], UUID,'dehumidifier', 'cfg_humidity', Result['dehumidifier_hum'][0]]
        decision.loc[1] = [room_id[0][0], UUID,'dehumidifier', 'cfg_power', Result['dehumidifier'][0]]
        try:
            ac_parameter
            decision.loc[2] = [room_id[0][0], UUID,'air_conditioner', 'cfg_temperature', Result['ac_temp'][0]]
            decision.loc[3] = [room_id[0][0], UUID,'air_conditioner', 'cfg_mode', Result['ac_mode'][0]]
            decision.loc[4] = [room_id[0][0], UUID,'air_conditioner', 'cfg_fan_level', Result['ac_fan'][0]]
        except NameError:
            decision.loc[2] = [room_id[0][0], UUID,'remote', 'cfg_temperature', Result['ac_temp'][0]]
            decision.loc[3] = [room_id[0][0], UUID,'remote', 'cfg_mode', Result['ac_mode'][0]]
            decision.loc[4] = [room_id[0][0], UUID,'remote', 'cfg_fan_level', Result['ac_fan'][0]]
        decision.loc[5] = [room_id[0][0], UUID,'ac_outlet', 'cfg_power', Result['fan_state'][0]]
        situation = pd.DataFrame(['pmv_balance'], columns=['inference_situation'])
        situation = pd.concat([situation]*6, ignore_index=True)
        Alg = pd.DataFrame([alg], columns=['inference_algorithm'])
        Alg = pd.concat([Alg]*6, ignore_index=True)
        decision = pd.concat([decision, situation, Alg], axis=1)
        #%% insert
        logger.info('Phase8: 塞入DB')
        '''
        db_conn = dbu.DB_UTILITY('pc0') #資料庫連接
        db_conn.insert_table(decision,'open_energyhub.decision_parameter_records')
        db_conn.DB_disconnect()
    ''' 
        logging.info('run end')
        return Result, predicted
    #%% 例外紀錄
    except Exception as e:
        logging.error("錯誤類型：%s" % Exception)
        logging.error("錯誤事件：%s" % e)
        logging.error('Error Line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
        print("錯誤類型：%s" % Exception)
        print("錯誤事件：%s" % e)
        print('Error Line {}'.format(sys.exc_info()[-1].tb_lineno),type(e).__name__,e)
        #db_conn.DB_disconnect()
        Result = pd.DataFrame([400], columns=['StatusCode'])
        return Result
        

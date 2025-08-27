import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pythermalcomfort.models import pmv_ppd
import time
import pandas as pd
import matplotlib.pyplot as plt

class HEMSEnvironment:
    def __init__(self, pmv_up, pmv_down):
        # 舒適度約束條件
        self.TEMP_MIN = 22.0
        self.TEMP_MAX = 26.0
        self.HUMIDITY_MIN = 45.0
        self.HUMIDITY_MAX = 65.0
        self.pmv_up = pmv_up
        self.pmv_down = pmv_down
        
        self.reset()
        
    def reset(self, test=False, indoor_temp=30, indoor_humidity=70):
        # 初始環境狀態
        if test:
            self.indoor_temp = indoor_temp  # 初始室內溫度
            self.indoor_humidity = indoor_humidity  # 初始室內濕度
            #self.outdoor_temp = 35  # 外部溫度
           # self.outdoor_humidity = 80 # 外部濕度
            #self.co2 = 1500 # 二氧化碳濃度
        else:
            self.indoor_temp = np.random.uniform(26, 35)  # 初始室內溫度
            self.indoor_humidity = np.random.uniform(60, 85)  # 初始室內濕度
            #self.outdoor_temp = np.random.uniform(26, 35)  # 外部溫度
            #self.outdoor_humidity = np.random.uniform(60, 85)  # 外部濕度
           # self.co2 = np.random.uniform(400, 2000) # 二氧化碳濃度
        
        return self._get_state()[0]
    
    def _get_state(self):
        # 狀態表示：歸一化處理
        return np.array([
            (self.indoor_temp - 20) / 20,  # 歸一化室內溫度
            (self.indoor_humidity - 40) / 50,  # 歸一化室內濕度
            #(self.outdoor_temp - 20) / 20,  # 歸一化外部溫度
           # (self.outdoor_humidity - 40) / 50,  # 歸一化外部濕度
           # (self.co2 - 400) / 500  # 歸一化二氧化碳濃度
        ]), np.array([self.indoor_temp, self.indoor_humidity])

    def calculate_power(self, dehumidifier_humidity, 
                        dehumidifier_on, fan_on, ac_temperature, 
                        ac_mode, ac_fan_speed):
        # 計算總功耗
        dehumidifier_power = 120  # 除濕機功率
        fan_power = 60            # 電扇功率
        ac_base_power = 1350      # 冷氣基礎功率
        ac_mode_weight = {0:0.5, 1:1, 2:0.9}        # 冷氣運轉模式係數 0:送風 1:冷氣 2:舒眠
        ac_fan_weight = {0:0.6, 1:1, 2:0.8} # 風速係數 0:低速 1:高速 2:自動

        # 計算除濕機功耗
        dehumidifier_consumption = dehumidifier_power * max(0, (self.indoor_humidity - dehumidifier_humidity)/100) if dehumidifier_on else 0

        # 計算電扇功耗
        fan_consumption = fan_power if fan_on else 0

        # 計算冷氣功耗
        if ac_mode_weight[ac_mode] == 1:
            ac_consumption = ac_base_power * ac_mode_weight[ac_mode] * (1 + 0.1 * max(0, self.indoor_temp-int(ac_temperature))) * ac_fan_weight[ac_fan_speed] 
        else: 
            ac_consumption = ac_base_power * ac_mode_weight[ac_mode] * ac_fan_weight[ac_fan_speed] 

        return dehumidifier_consumption + fan_consumption + ac_consumption
    
    def step(self, action, isa):
        # 解析動作空間
        # 0-1: 除濕機啟閉狀態
        # 2-3: 除濕機設定濕度 (40-70%)
        # 4-5: 除濕機運轉模式 (0-正常, 1-強力)
        # 6-7: 冷氣啟閉狀態
        # 8-9: 冷氣設定溫度 (22-28)
        # 10-11: 冷氣運轉模式 (0-節能, 1-強力)
        # 12-13: 冷氣風扇強弱 (0-低, 1-高)
        # 14-15: 電風扇啟閉狀態
        
        # 解析和處理動作
        dehumidifier_on = action[0] > 0.5  # 除濕機開關
        dehumidifier_humidity = 50 + min(1, action[1]) * 10  # 除濕機設定濕度 40-70%
        
        # 從指令集中選擇冷氣控制指令
        ac_command_index = min(int(action[3] * (len(isa) - 1)), len(isa)-1)
        ac_command = isa.iloc[ac_command_index]
        
        fan_on = action[2] > 0.3  # 電扇開關
        
        # 計算能源消耗
        energy_consumption = self.calculate_power(dehumidifier_humidity, 
                                                  int(dehumidifier_on), int(fan_on), ac_command['ac_temp'], 
                                                  ac_command['ac_mode'], ac_command['ac_fan'])
        
        # 溫濕度變化模型
        temp_change = 0
        humidity_change = 0
        
        # 除濕機影響
        if dehumidifier_on:
            target_humidity = dehumidifier_humidity
            humidity_change = -7.1 * 0.9  # 濕度變化率
        
        # 冷氣影響
        if ac_command['ac_mode'] == 1:  # 冷氣模式
            temp_diff = self.indoor_temp - int(ac_command['ac_temp'])
            cooling_rate = 0.7
        else:
            cooling_rate = 0
            temp_diff = 0
            
        # 外部環境影響
        #outdoor_influence = self.indoor_temp * 0.1
        
        temp_change = -cooling_rate * temp_diff
        
        # 風扇影響
        if fan_on:
            # 風扇會加速溫度趨近外部溫度
            temp_change += 0.005 * self.indoor_temp
        
        # 更新狀態
        self.indoor_temp = np.clip(self.indoor_temp + temp_change, 20, 35)
        self.indoor_humidity = np.clip(self.indoor_humidity + humidity_change, 40, 85)
        
        # 計算PMV舒適度指標
        pmv = pmv_ppd(tdb=self.indoor_temp, tr=self.indoor_temp, vr=0.25, 
                      rh=self.indoor_humidity, met=1, clo=0.5, limit_inputs=False)
        
        cost = energy_consumption
        # 違反舒適度約束時給予懲罰
        pmv_value = (pmv['pmv'])
        if float(self.pmv_down)<pmv_value<float(self.pmv_up):
            pass
        else:
            pmv['pmv'] = 10  # 較大的懲罰
            
        # 確保舒眠模式只在二氧化碳濃度低時使用
       # if self.co2 >=1000 and ac_command['ac_mode'] == 2:
        #    energy_consumption += 10000  # 較大的懲罰
        
        # 獎勵函數：最小化能源消耗與PMV的乘積
        reward = -energy_consumption*abs(pmv['pmv'])
        '''
        # 更新外部條件
        self.outdoor_temp = np.clip(
            self.outdoor_temp + np.random.uniform(-1, 1), 
            20, 40
        )
        
        self.outdoor_humidity = np.clip(
            self.outdoor_humidity + np.random.uniform(-2, 2), 
            40, 90
        )
        '''
        pmv = pmv_ppd(tdb=self.indoor_temp, tr=self.indoor_temp, vr=0.25, 
                      rh=self.indoor_humidity, met=1, clo=0.5, limit_inputs=False) #計算真實pmv蓋過懲罰數值(回傳用)
        return self._get_state()[0], reward, False, self._get_state()[1], \
                  [int(dehumidifier_on), int(dehumidifier_humidity), ac_command['ac_temp'], 
                  ac_command['ac_fan'], ac_command['ac_mode'], int(fan_on)], pmv['pmv'], cost
         

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        # 策略網絡架構
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim * 2)
        )
    
    def forward(self, state):
        x = self.actor(state)
        mean, std = torch.chunk(x, 2, dim=1)
        std = F.softplus(std) + 1e-5
        return mean, std

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        # 價值網絡架構
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, state):
        return self.critic(state)

class PPOAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.00005):
        # 初始化策略網絡和價值網絡
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.value_net = ValueNetwork(state_dim)
        
        # 設定優化器
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)
        
        # PPO超參數
        self.clip_range = 0.2
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01

    def select_action(self, state):
        # 根據當前狀態選擇動作
        state = torch.FloatTensor(state).unsqueeze(0)
        mean, std = self.policy_net(state)
        
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return abs(action.detach()).numpy()[0], log_prob.item()

    def update(self, states, actions, rewards, log_probs, dones):
        # 將數據轉換為張量
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        log_probs = torch.FloatTensor(log_probs)
        dones = torch.FloatTensor(dones)

        # 計算價值和優勢
        values = self.value_net(states).squeeze()
        advantages = rewards - values.detach()
        
        # 計算新的動作概率
        mean, std = self.policy_net(states)
        dist = torch.distributions.Normal(mean, std)
        
        new_log_probs = dist.log_prob(actions).sum(dim=-1)
        prob_ratio = torch.exp(new_log_probs - log_probs)
        
        # 計算PPO目標函數
        surrogate_loss1 = prob_ratio * advantages
        surrogate_loss2 = torch.clamp(prob_ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
        policy_loss = -torch.min(surrogate_loss1, surrogate_loss2).mean()
        
        # 計算價值損失和熵損失
        value_loss = F.mse_loss(values, rewards)
        entropy_loss = -self.entropy_coef * dist.entropy().mean()
        
        # 總損失
        total_loss = policy_loss + self.value_loss_coef * value_loss + entropy_loss
        
        # 更新網絡參數
        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        total_loss.backward()
        self.policy_optimizer.step()
        self.value_optimizer.step()
        
    def save_model(self):
        """保存模型"""
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'value_state_dict': self.value_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict()
        }, "./config/ppo_pmv_balance.pt")
    
    def load_model(self, room_id):
        """加載模型
        Args:
            path: 模型路徑
        """
        path='./config/ppo_pmv_balance_'+room_id+'.pt'
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])


    
#%%
if __name__ == "__main__":
    start = time.time()
    # 初始化環境和代理
    env = HEMSEnvironment()
    agent = PPOAgent(state_dim=2, action_dim=4)
    isa = pd.read_csv('C:/Users/hankli/Documents/114計劃相關/測試數據/紅外線遙控器冷氣調控指令集.csv')
    
    # 儲存訓練結果的列表
    total_energy_consumption = []
    Device_state = pd.DataFrame()
    Env_state = pd.DataFrame()
    
    # 訓練迴圈
    for episode in range(1000):
        state = env.reset()
        episode_reward = 0
        episode_energy_consumption = 0
        Actions = []
        states, actions, rewards, log_probs, dones, env_states,device_states, pmvs = [], [], [], [], [], [], [], []
        
        for step in range(24):
            # 選擇動作
            action, log_prob = agent.select_action(state)
            Actions.append(action)
            
            # 執行動作
            next_state, reward, done, env_state, device_state, pmv, _ = env.step(action, isa)
            
            # 儲存訓練數據
            pmvs.append(pmv)
            env_states.append(env_state)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            dones.append(done)
            state = next_state
            episode_reward += reward
            device_states.append(device_state)
            episode_energy_consumption += abs(reward)
            
            if step == 23:
                Device_state = Device_state.append(pd.DataFrame(device_state).T, ignore_index=True)
                Env_state = Env_state.append(pd.DataFrame(env_state).T, ignore_index=True)
            if done:
                break
        
        # 更新策略
        if len(states) > 0:
            agent.update(states, actions, rewards, log_probs, dones)
    
        total_energy_consumption.append(episode_energy_consumption)
        
        # 定期輸出訓練進度
        if episode % 100 == 0:
            avg_energy = np.mean(total_energy_consumption[-50:])
            
            print(f"Episode {episode}, Total Reward: {episode_reward:.2f}, Avg Energy Consumption: {avg_energy:.2f}")
            print(f"env_state: {env_states[-1]}, pmv: {pmvs[-1]}")
            print(f"device_state: {device_states[-1]}")
            print('-'*80)
            
    # 儲存訓練結果
    Result = pd.concat([Env_state, Device_state], axis=1)
    Result.columns = ['env_temp', 'env_humd', 'dehumidifier', 'dehumidifier_hum',
             'ac_temp', 'ac_fan', 'ac_mode', 'fan_state']
    Result['ac_temp'] = np.where(Result['ac_mode'] == 0, '-', Result['ac_temp'])
    Result['dehumidifier_hum']  = Result['dehumidifier_hum'].astype(int)
    Result['dehumidifier_hum'] = round(Result['dehumidifier_hum'] / 5) * 5
    Result.to_csv('C:/Users/hankli/Documents/114計劃相關/調控參數/PPO能源與舒適度加權參數測試結果.csv')
    end = time.time()
    print(end-start)
#%% save & load
    agent.save_model()
    new_agent = PPOAgent(state_dim=2, action_dim=4)
    new_agent.load_model()
#%%
    # 測試階段
    start = time.time()
    env_state_test = pd.DataFrame()
    device_state_test = pd.DataFrame()
    Cost = []
    Pmv = []
    state = env.reset(test=True)
    
    for i in range(8):
        # 選擇動作
        action, _ = agent.select_action(state)
        if i == 0:
            env_state_test = env_state_test.append(pd.DataFrame(env._get_state()[1]).T, ignore_index=True)      
            
        # 執行動作
        next_state, reward, done, env_state, device_state, pmv, cost = env.step(action, isa)
        Cost.append(cost)
        Pmv.append(pmv)
        
        if i != 0:
            env_state_test = env_state_test.append(pd.DataFrame(env_state).T, ignore_index=True)
        state = next_state
        
        device_state_test = device_state_test.append(pd.DataFrame(device_state).T, ignore_index=True)
        
    # 儲存測試結果
    Pmv = pd.DataFrame(Pmv)
    Result_test = pd.concat([env_state_test, device_state_test, Pmv], axis = 1)
    Result_test.columns = ['env_temp', 'env_humd', 'dehumidifier', 'dehumidifier_hum',
             'ac_temp', 'ac_fan', 'ac_mode', 'fan_state', 'pmv']
    Result_test['dehumidifier_hum']  = Result_test['dehumidifier_hum'].astype(int)
    Result_test['dehumidifier_hum'] = round(Result_test['dehumidifier_hum'] / 5) * 5
    Result_test['dehumidifier_hum'] = np.where(Result_test['dehumidifier'] == 0, '-', Result_test['dehumidifier_hum'])
    Result_test.to_csv('C:/Users/hankli/Documents/114計劃相關/調控參數/PPO能源與舒適度加權參數應用結果.csv')
    end = time.time()
    print(end-start)
    
#%%
    # 計算節能效果
    data = pd.read_csv('C:/Users/hankli/Documents/114計劃相關/測試數據/nilm_data_ritaluetb_hour.csv')
    data = data.iloc[14:21, :]
    ec = sum(data['w_4'])/60000
    
    print(sum(Cost)/1000, ec)
    print((1-((sum(Cost)/1000)/ec))*100)
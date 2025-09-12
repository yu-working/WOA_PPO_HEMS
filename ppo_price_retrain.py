# 導入必要的庫
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pythermalcomfort.models import pmv_ppd
import time
import pandas as pd

class HEMSEnvironment:
    """家庭能源管理系統環境類"""
    def __init__(self, pmv_up, pmv_down):
        # 舒適度約束條件
        self.TEMP_MIN = 22.0      # 最低舒適溫度限制 (°C)
        self.TEMP_MAX = 26.0      # 最高舒適溫度限制 (°C)
        self.HUMIDITY_MIN = 45.0  # 最低舒適濕度限制 (%)
        self.HUMIDITY_MAX = 65.0  # 最高舒適濕度限制 (%)
        self.pmv_up = pmv_up
        self.pmv_down = pmv_down
        
        self.reset()
        
    def reset(self, indoor_temp=30, indoor_humidity=70):
        """重置環境狀態
        Args:
            test: 是否為測試模式
        Returns:
            初始狀態
        """
        # 初始環境狀態
        self.indoor_temp = indoor_temp  # 初始室內溫度
        self.indoor_humidity = indoor_humidity  # 初始室內濕度
         #self.outdoor_temp = np.random.uniform(26, 35)  # 外部溫度
         #self.outdoor_humidity = np.random.uniform(60, 85)  # 外部濕度
        # self.co2 = np.random.uniform(400, 2000) # 二氧化碳濃度
        
        return self._get_state()[0]
    
    def _get_state(self):
        """獲取當前狀態
        Returns:
            歸一化後的狀態向量和原始狀態值
        """
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
        """計算設備總功耗
        Args:
            dehumidifier_humidity: 除濕機設定濕度
            dehumidifier_on: 除濕機開關狀態
            fan_on: 電扇開關狀態
            ac_temperature: 空調設定溫度
            ac_mode: 空調運行模式
            ac_fan_speed: 空調風速
        Returns:
            總功耗
        """
        # 各設備基礎功率
        dehumidifier_power = 120  # 除濕機功率
        fan_power = 60            # 電扇功率
        ac_base_power = 1350      # 冷氣基礎功率
        
        # 運行模式係數
        ac_mode_weight = {0:0.5, 1:1, 2:0.9}        # 冷氣運轉模式係數 0:送風 1:冷氣 2:舒眠
        ac_fan_weight = {0:0.6, 1:1, 2:0.8} # 風速係數 0:低速 1:高速 2:自動
        
        # 計算除濕機功耗
        dehumidifier_consumption = dehumidifier_power * max(0, (self.indoor_humidity - dehumidifier_humidity)/100) if dehumidifier_on else 0

        # 計算電扇功耗
        fan_consumption = fan_power if fan_on else 0

        # 計算冷氣功耗
        if ac_mode_weight[ac_mode] == 1:
            ac_consumption = ac_base_power * ac_mode_weight[ac_mode] * (1 + 0.1 * max(0, float(self.indoor_temp)-float(ac_temperature))) * ac_fan_weight[ac_fan_speed] 
        else: 
            ac_consumption = ac_base_power * ac_mode_weight[ac_mode] * ac_fan_weight[ac_fan_speed] 

        return dehumidifier_consumption + fan_consumption + ac_consumption
    
    def step(self, action, isa, step):
        """執行一步環境交互
        Args:
            step: 當前時間步
            action: 動作向量
        Returns:
            next_state: 下一狀態
            reward: 獎勵值
            done: 是否結束
            env_state: 環境狀態
            device_state: 設備狀態
            pmv: PMV值
            cost: 能源成本
        """
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
        ac_command_index = min(int(action[3] * (len(isa) - 1)), len(isa)-1)
        ac_command = isa.iloc[ac_command_index]        
        fan_on = action[2] > 0.3
        # 計算能源消耗
        energy_consumption = self.calculate_power(dehumidifier_humidity, 
                                                  int(dehumidifier_on), int(fan_on), ac_command['ac_temp'], 
                                                  ac_command['ac_mode'], ac_command['ac_fan'])
        # 溫濕度變化模型
        base_area = 25 #參考面積 25 m^2
        cooling_per_kW = 2 # 設每kW每小時降溫 2
        area = 20 # 暫定空間面積為20 m^2
        cooling_capacity = 2.8 if ac_command['ac_mode'] == 1 else 0 # 暫定冷氣能力 如果為冷氣模式為 2.8 kW 送風模式為 0 
        area_factor = base_area / area
        ac_temp_drop = cooling_capacity * cooling_per_kW * area_factor
        temp_final = max(self.indoor_temp - ac_temp_drop, float(ac_command['ac_temp'])) if float(ac_command['ac_mode']) == 1 \
        else min(self.indoor_temp + 2, 35)
        ## 濕度
        ac_humidity_drop = min((cooling_capacity / area) * 100, 30)
        dehumidifier_humidity_drop = 7.1 * 0.9 if dehumidifier_on == 1 else 0
        rh_final = max(self.indoor_humidity - ac_humidity_drop - dehumidifier_humidity_drop, dehumidifier_humidity) if dehumidifier_humidity > 45 \
        else max(self.indoor_humidity - ac_humidity_drop - dehumidifier_humidity_drop, 45)
        # 更新室內狀態
        self.indoor_temp, self.indoor_humidity = temp_final, rh_final
        
        # 計算PMV舒適度指標
        pmv = pmv_ppd(tdb=self.indoor_temp, tr=self.indoor_temp, vr=0.25, 
                      rh=self.indoor_humidity, met=1, clo=0.5, limit_inputs=False)
        
        cost = energy_consumption
        # 違反舒適度約束時給予懲罰
        pmv_value = (pmv['pmv'])
        if float(self.pmv_down)<pmv_value<float(self.pmv_up):
            pass
        else:
            energy_consumption += 100000  # 較大的懲罰
        
        # 根據時段計算電價
        if step >= 24:
            step = step-24
        if step in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
            price = 1.96  # 離峰時段
        elif step in [9, 10, 11, 12, 13, 14, 15, 22, 23]:
            price = 4.54  # 半尖峰時段
        else:
            price = 6.92  # 尖峰時段
            
        reward = -energy_consumption*price
        
        # 更新外部環境條件
        '''
        self.outdoor_temp = np.clip(
            self.outdoor_temp + np.random.uniform(-1, 1), 
            20, 40
        )
        self.outdoor_humidity = np.clip(
            self.outdoor_humidity + np.random.uniform(-2, 2), 
            40, 90
        )
        '''
        return self._get_state()[0], reward, False, self._get_state()[1], \
                  [int(dehumidifier_on), int(dehumidifier_humidity), ac_command['ac_temp'], 
                  ac_command['ac_fan'], ac_command['ac_mode'], int(fan_on)], pmv['pmv'], cost/1000*price

class PolicyNetwork(nn.Module):
    """策略網路"""
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim * 2)
        )
    
    def forward(self, state):
        """前向傳播
        Args:
            state: 輸入狀態
        Returns:
            mean: 動作均值
            std: 動作標準差
        """
        x = self.actor(state)
        mean, std = torch.chunk(x, 2, dim=1)
        std = F.softplus(std) + 1e-5
        return mean, std

class ValueNetwork(nn.Module):
    """價值網路"""
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, state):
        """前向傳播
        Args:
            state: 輸入狀態
        Returns:
            價值估計
        """
        return self.critic(state)

class PPOAgent:
    """PPO算法智能體"""
    def __init__(self, state_dim, action_dim, learning_rate=0.0005):
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.value_net = ValueNetwork(state_dim)
        
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)
        
        self.clip_range = 0.2
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01

    def select_action(self, state):
        """選擇動作
        Args:
            state: 當前狀態
        Returns:
            action: 選擇的動作
            log_prob: 動作的對數概率
        """
        state = torch.FloatTensor(state).unsqueeze(0)
        mean, std = self.policy_net(state)
        
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return abs(action.detach()).numpy()[0], log_prob.item()

    def update(self, states, actions, rewards, log_probs, dones):
        """更新策略和價值網路
        Args:
            states: 狀態序列
            actions: 動作序列
            rewards: 獎勵序列
            log_probs: 動作對數概率序列
            dones: 終止標誌序列
        """
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        log_probs = torch.FloatTensor(log_probs)
        dones = torch.FloatTensor(dones)

        values = self.value_net(states).squeeze()
        advantages = rewards - values.detach()
        
        mean, std = self.policy_net(states)
        dist = torch.distributions.Normal(mean, std)
        
        new_log_probs = dist.log_prob(actions).sum(dim=-1)
        prob_ratio = torch.exp(new_log_probs - log_probs)
        
        surrogate_loss1 = prob_ratio * advantages
        surrogate_loss2 = torch.clamp(prob_ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
        policy_loss = -torch.min(surrogate_loss1, surrogate_loss2).mean()
        
        value_loss = F.mse_loss(values, rewards)
        entropy_loss = -self.entropy_coef * dist.entropy().mean()
        
        total_loss = policy_loss + self.value_loss_coef * value_loss + entropy_loss
        
        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        total_loss.backward()
        self.policy_optimizer.step()
        self.value_optimizer.step()
        
    def save_model(self, room_id):
        """保存模型"""
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'value_state_dict': self.value_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict()
        }, './config/ppo_price_'+room_id+'.pt')
    
    def load_model(self, room_id):
        """加載模型
        Args:
            path: 模型路徑
        """
        path='./config/ppo_price_'+room_id+'.pt'
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])


    
#%%
def ppo_retrain(indoor_data, room_id):
    # 訓練部分
    start = time.time()
    pmv_ul_ll = pd.read_csv('./config/pmv_ul_ll.csv')
    env = HEMSEnvironment(
        pmv_up = pmv_ul_ll.loc[pmv_ul_ll['room_id'] == room_id]['pmv_ul'],
        pmv_down = pmv_ul_ll.loc[pmv_ul_ll['room_id'] == room_id]['pmv_ll'])
    agent = PPOAgent(state_dim=2, action_dim=4)
    isa = pd.read_csv('./config/紅外線遙控器冷氣調控指令集.csv')
    
    total_energy_consumption = []
    Device_state = pd.DataFrame()
    Env_state = pd.DataFrame()
    #indoor_data = np.array([Device['sensor']['op_temperature']['record_value'], Device['sensor']['op_humidity']['record_value']]).T #IoT溫溼度擷取
    for episode in range(len(indoor_data)):
        state = env.reset(indoor_temp=indoor_data[episode][0], indoor_humidity=indoor_data[episode][1])
        episode_reward = 0
        episode_energy_consumption = 0
        Actions = []
        states, actions, rewards, log_probs, dones, env_states,device_states, pmvs = [], [], [], [], [], [], [], []
        
        for step in range(24):
            action, log_prob = agent.select_action(state)
            Actions.append(action)
            next_state, reward, done, env_state, device_state, pmv, _ = env.step(action, isa, step)
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
        
        # 打印訓練進度
        if episode % 100 == 0:
            avg_energy = np.mean(total_energy_consumption[-50:])
            
            print(f"Episode {episode}, Total Reward: {episode_reward:.2f}, Avg Energy Consumption: {avg_energy:.2f}")
            print(f"env_state: {env_states[-1]}, pmv: {pmvs[-1]}")
            print(f"device_state: {device_states[-1]}")
            print('-'*80)
            
    # 保存訓練結果
    Result = pd.concat([Env_state, Device_state], axis=1)
    Result.columns = ['env_temp', 'env_humd', 'dehumidifier', 'dehumidifier_humidity',
             'ac_temp', 'ac_fan', 'ac_mode', 'fan_state']
   # Result.to_csv('C:/Users/hankli/Documents/114計劃相關/調控參數/PPO用電分配參數測試結果.csv')
    end = time.time()
    print(end-start)
#%% 保存和加載模型測試
    agent.save_model(room_id)

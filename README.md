# WOA_PPO_HEMS
![License](https://img.shields.io/badge/license-MIT-yellow)
![Language](https://img.shields.io/badge/language-python-blue)

 > 結合PPO 、 WOA 與 Decision tree 的能耗與舒適度加權平衡家庭能源管理系統

## 專案簡介

本專案旨在透過智慧化演算法，模擬家庭能源管理系統（Home Energy Management System, HEMS）的溫濕度調控，以達成能源消耗與室內舒適度之間的最佳平衡。

專案分別採用兩種方法進行最佳化：

#### 1.鯨魚優化演算法（Whale Optimization Algorithm, WOA）

 - 結合 決策樹（Decision Tree） 與 PMV（Predicted Mean Vote） 指標。
 - 模擬 HEMS 的決策過程，平衡能源使用與居住舒適度。

#### 2.近端策略優化（Proximal Policy Optimization, PPO）

 - 結合 PMV 指標，透過強化學習動態調整控制策略。
 - 在不同室內外環境與電價條件下，達到高效的能源管理。

系統會依據以下輸入條件：

 - 室內環境（溫度、濕度）

 - 使用者個人化 PMV 舒適區間

輸出動態決策控制項目包括：

 - 冷氣模式、風速模式與設定溫度

 - 除濕機開關與目標濕度

 - 電扇開關

並比較 WOA 與 PPO 在能源消耗與舒適度上的表現，提供最佳節能建議。

## 核心功能

#### 1.PPO 智能體

 - 使用行為網路 (Policy Network) 和價值網路 (Value Network) 進行學習，透過強化學習不斷優化設備運行策略以降低電費並保持舒適。

#### 2.WOA 優化溫濕度設定並透過 Decision tree 預測設備狀態

 - 搜索最佳室內溫度與濕度，使能源成本最低且 PMV 舒適度在指定範圍內，並根據WOA所計算出的室內條件，透過Decision tree預測並輸出各設備的最佳運行設定。

#### 3.PMV 舒適度評估

 - 使用 pythermalcomfort 計算室內舒適度。

#### 4.電費計算

 - 根據台灣分時電價計算不同時段的耗電成本。


## 主要程式架構

```
WOA_PPO_HEMS/
│
├─ pmv_balance_api.py          # 主程式入口，負責執行整體邏輯與資料輸入/輸出
├─ main_pmv_balance.py         # 主程式，負責接收API的資料並進行資料處理、決策產出儲存與回傳
├─ db_utility.py               # 資料庫連接程式，負責將生成的決策存入DB
├─ ppo_pmv_balance_online.py   # PPO決策運算程式
├─ ppo_pmv_balance_retrain.py  # PPO模型訓練程式
├─ WOA_pmv_balance_online.py   # WOA決策運算程式
├─ UserFeedbackSystem.py       # 使用者調控回饋程式
├─ user_feedback_log.csv       # 使用者調控回饋紀錄
├─ config/
│   ├─ 紅外線遙控器冷氣調控指令集.csv      # 冷氣遙控指令集
│   ├─ pmv_ul_ll.csv                     # 使用者個人化 PMV 舒適區間
│   └─ ppo_pmv_balance_{room_id}.pt      # PPO模型存檔
├─ data/
│   ├─ data-1743586080241.csv           # 測試用 sample 檔
│   └─ nilm_data_ritaluetb_hour.csv     # 歷史用電數據
└─ log/
    └─ main_decision_pmv_balance_{datetime}.log    # 執行log檔
```

## 安裝需求

請先安裝所需套件：
```
pandas == 1.2.4
numpy == 1.24.4
pythermalcomfort == 2.10.0
matplotlib == 3.7.4
scikit-fuzzy==0.5.0
scikit-learn==1.2.2
joblib == 1.3.2
Flask == 3.0.3
psycopg2 == 2.8.6
pymssql == 2.2.1
sshtunnel == 0.4.0
torch == 2.2.0
torchvision == 0.17.0
```

## 執行流程

#### 1.執行 `pmv_balance_api.py`

#### 2.開啟命令提示字元

輸入 `curl -X POST http://127.0.0.1:5000/ -H "Content-Type: application/json" -d "{}"`

#### 3.呼叫 `main_pmv_balance.py` function `pmv_balance`

#### 4.讀取數據

目前使用 `data-1743586080241.csv` 作為演示範例

#### 5.開始生成log檔案

#### 6.電器資料萃取與數據處理

#### 7.建立WOA、PPO環境

 - 若未讀取到PPO再訓練模型則呼叫 `ppo_pmv_balance_retrain.py` 訓練模型
 - 讀取 `pmv_ul_ll.csv`，系統將根據使用者使用習慣，設定個人化的 PMV 舒適範圍（上下限），該範圍將作為最佳化演算法的約束條件，用於平衡能源消耗與使用者舒適度。
 - 初始化WOA、PPO環境

#### 8.決策計算

 - 分別使用WOA與PPO進行決策計算，重複執行直到達成節能效果
 - 迴圈結束後依據節能效果與pmv判斷使用WOA或是PPO之決策

#### 9.數值轉換

 - 將除濕機啟閉['dehumidifier'] 0/1紀錄為off/on
 - 將電風扇啟閉['fan_state'] 0/1紀錄為off/on
 - 將冷氣模式['ac_mode'] 0/1紀錄為cool/fan
 - 將冷氣風扇['ac_fan'] 0/1/2紀錄為low/high/auto

#### 10.將決策計算結果儲存至資料庫

呼叫 `db_utility.py` 連接並將決策新增至資料庫

> [!WARNING]
> 請確保遠端資料庫可連線

#### 11.顯示決策計算結果

程式執行後會輸出：

| 項目 | 說明 |
|-----|------|
| StatusCode | 執行狀態代碼 |
| ac_fan | 冷氣風扇強度 |
| ac_mode | 冷氣模式 |
| ac_temp | 冷氣溫度設定 |
| dehumidifier_hum | 除濕機濕度設定 |
| dehumidifier_state| 除濕機啟閉狀態 |
| fan_state | 電風扇啟閉狀態 |

範例輸出：
```
{
  "StatusCode":"200",
  "ac_fan":"low",
  "ac_mode":"cool",
  "ac_temp":"29",
  "dehumidifier_hum":"60.0",
  "dehumidifier_state":"on",
  "fan_state":"on"
}
```

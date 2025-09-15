import time
import psycopg2
from datetime import datetime, timedelta
import os
import json
import requests
from urllib.parse import urljoin
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

try:
    db_host = os.getenv("DB_HOST")
    db_database = os.getenv("DB_DATABASE")
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    TIME_REC_PATH = os.getenv("TIME_REC_PATH", "./data/id_time_rec.json")
    TIME_TO_SLEEP = int(os.getenv("TIME_TO_SLEEP", 2 * 60))
    API_BASE_URL = os.getenv("API_BASE_URL")
    API_SECRET = os.getenv("API_SECRET")
    QUESTIONNAIRE_ID_PATH = os.getenv(
        "QUESTIONNAIRE_ID_PATH", "./data/questionnaire_id.json"
    )
except Exception as e:
    raise ValueError("setting DB environment variables encountered an error.\n", e)


def send_questionnaire(room_id, questionnaire_data):
    """
    透過 REST API 發送熱舒適度問卷通知給指定的 App。

    Args:
        room_id (str): 房間 ID。
        questionnaire_data (dict): 包含問卷相關資料的字典。
    """

    endpoint = "/wetrun/ask_thermal_comfort"
    url = urljoin(API_BASE_URL, endpoint)

    # 建立 GET 請求的參數
    params = {
        "secret": API_SECRET,
        "dummy": "true",  # 依照你的說明，先使用 dummy 參數
        "room": room_id,
    }

    try:
        # 發送 GET 請求
        response = requests.get(url, params=params)
        response.raise_for_status()  # 如果請求不成功 (例如 4xx 或 5xx)，會觸發例外

        # 解析回傳的 JSON 數據
        result = response.json()
        print(
            f"✅ Success! Sent questionnaire to Room ID {room_id}. Send results: {result}"
        )

    except requests.exceptions.HTTPError as http_err:
        print(f"❌ HTTP error occurred: {http_err}")
        print(f"Response content: {response.text}")
    except json.JSONDecodeError:
        print(f"❌ Failed to decode JSON from response. Content: {response.text}")
    except requests.exceptions.RequestException as req_err:
        print(f"❌ An error occurred during the request: {req_err}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")


def get_sim_data(username: str):
    response = requests.get(
        f"http://openenergyhub.energy-active.org.tw:9002/dryrun/decision?notify=false&secret=a1b2c3d4e5f6g7h8&username={username}"
    )

    js = response.json()
    js_data = pd.DataFrame(js["training_data"])
    js_data.recorded_datetime = pd.to_datetime(js_data.recorded_datetime).dt.strftime(
        "%Y-%m-%d %H:%M"
    )
    pivot_df = js_data.pivot_table(
        index="recorded_datetime",
        columns="capability_name",
        values="record_value",
        aggfunc="first",
    ).reset_index()

    return pivot_df


def get_sim_data_roomid(room_id: str):
    response = requests.get(
        f"http://openenergyhub.energy-active.org.tw:9002/dryrun/decision?notify=false&secret=a1b2c3d4e5f6g7h8&room_id={room_id}"
    )

    js = response.json()
    js_data = pd.DataFrame(js["training_data"])
    js_data.recorded_datetime = pd.to_datetime(js_data.recorded_datetime).dt.strftime(
        "%Y-%m-%d %H:%M"
    )
    pivot_df = js_data.pivot_table(
        index="recorded_datetime",
        columns="capability_name",
        values="record_value",
        aggfunc="first",
    ).reset_index()

    return pivot_df


def connect_user_ai_decision_db(users_to_check: list) -> tuple[list, list]:
    """
    連接到 PostgreSQL 資料庫，並檢查指定使用者的最新決策紀錄，是否手動調整過。"""

    conn = None
    cur = None
    to_remove = []
    try:
        # 建立資料庫連線
        conn = psycopg2.connect(
            host=db_host,
            dbname=db_database,
            user=db_user,
            password=db_password,
        )

        cur = conn.cursor()

        # SQL: 每個 decision_remark 選最新的一筆紀錄
        query = """
            SELECT DISTINCT ON (decision_remark) 
                   decision_remark,
                   next_override_source
            FROM open_energyhub.view_user_ai_decision_started_finished_next_merged
            WHERE decision_remark IN %s
            ORDER BY decision_remark, user_ai_decision_started_datetime DESC;
        """
        cur.execute(query, (tuple(users_to_check),))
        latest_records = cur.fetchall()

        # 找出最新紀錄中 next_override_source = 'user' 的 decision_remark
        manual_adjusted = {row[0] for row in latest_records if row[1] == "user"}

        # 過濾掉這些使用者
        for u in users_to_check:
            if u in manual_adjusted:
                to_remove.append(u)

        users_to_check = [u for u in users_to_check if u not in manual_adjusted]

        return users_to_check, to_remove

    except Exception as e:
        print(f"❌ Connect Decision Database error: {e}")
        return users_to_check, to_remove  # 出錯時就回傳原本的 list，不強制清空

    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()


def background_check_loop():
    while True:
        try:
            # 檢查問卷 JSON 檔案是否存在
            if not os.path.exists(QUESTIONNAIRE_ID_PATH):
                print(f"檔案不存在：{QUESTIONNAIRE_ID_PATH}")
                time.sleep(TIME_TO_SLEEP)
                continue

            # 讀取待填寫問卷的使用者
            with open(QUESTIONNAIRE_ID_PATH, "r", encoding="utf-8") as f:
                questionnaire_data = json.load(f)

            users_to_check = list(questionnaire_data.keys())
            users_to_remove = []

            if not users_to_check:
                time.sleep(TIME_TO_SLEEP)
                continue

            users_to_check, users_to_remove = connect_user_ai_decision_db(
                users_to_check
            )

            for remark in users_to_check:
                # 檢查時間差是否超過 20 分鐘
                last_send_time_str = questionnaire_data[remark]["time"]
                last_send_time = datetime.fromisoformat(last_send_time_str)
                time_difference = datetime.now() - last_send_time

                if time_difference.total_seconds() > 20 * 60:
                    send_questionnaire(
                        questionnaire_data[remark]["room_id"],
                        questionnaire_data[remark],
                    )
                    users_to_remove.append(remark)
                else:
                    try:
                        sim_data_ser = get_sim_data(remark).iloc[-1]
                        op_hum = sim_data_ser["op_humidity"]
                        op_temp = sim_data_ser["op_temperature"]
                        if op_hum is not None and op_temp is not None:
                            if (
                                op_hum <= questionnaire_data[remark]["best_humd"]
                                and op_temp <= questionnaire_data[remark]["best_temp"]
                            ):
                                send_questionnaire(
                                    questionnaire_data[remark]["room_id"],
                                    questionnaire_data[remark],
                                )
                                users_to_remove.append(remark)
                    except Exception as e:
                        print(f"[Error] get_sim_data for decision remark {remark}: {e}")

            # 從 dictionary 中刪除已處理的 key
            if users_to_remove:
                for remark in users_to_remove:
                    questionnaire_data.pop(remark, None)

                with open(QUESTIONNAIRE_ID_PATH, "w", encoding="utf-8") as f:
                    json.dump(questionnaire_data, f, indent=4)

        except FileNotFoundError:
            print(f"檔案不存在：{QUESTIONNAIRE_ID_PATH}")
        except json.JSONDecodeError:
            print(f"檔案格式錯誤：{QUESTIONNAIRE_ID_PATH}，將重新初始化。")
            with open(QUESTIONNAIRE_ID_PATH, "w", encoding="utf-8") as f:
                json.dump({}, f)
        except Exception as e:
            print(f"[Error] background_check_loop: {e}")

        # 每x分鐘執行一次
        time.sleep(TIME_TO_SLEEP)


def chek_user_questionnaire_status(
    room_id: str,
    decision_remark: str,
    best_temp: int,
    best_humd: int,
):
    # 建立 data 資料夾，如果它不存在的話
    os.makedirs(os.path.dirname(TIME_REC_PATH), exist_ok=True)

    time_rec_data = {}

    # 讀取時間記錄檔，若檔案不存在則會創建
    if os.path.exists(TIME_REC_PATH):
        with open(TIME_REC_PATH, "r", encoding="utf-8") as f:
            try:
                time_rec_data = json.load(f)
            except json.JSONDecodeError:
                # 如果檔案是空的或格式錯誤，重新初始化為空字典
                time_rec_data = {}

    current_time = datetime.now()
    should_update = False

    # 檢查 room_id 是否存在且時間差超過24小時
    if room_id in time_rec_data:
        last_update_str = time_rec_data[room_id]
        last_update_time = datetime.fromisoformat(last_update_str)
        time_difference = current_time - last_update_time

        if time_difference.total_seconds() >= 24 * 3600:
            should_update = True
    else:
        # 如果 room_id 不在 dictionary 中，則視為需要更新
        should_update = True

    if should_update:
        # 更新時間記錄
        time_rec_data[room_id] = current_time.isoformat()
        with open(TIME_REC_PATH, "w", encoding="utf-8") as f:
            json.dump(time_rec_data, f, indent=4)

        # 準備要儲存的問卷資料
        questionnaire_data = {
            "room_id": room_id,
            "best_temp": best_temp,
            "best_humd": best_humd,
            "time": current_time.isoformat(),
        }

        # 讀取問卷檔案，若檔案不存在則會創建
        remark_data = {}
        if os.path.exists(QUESTIONNAIRE_ID_PATH):
            with open(QUESTIONNAIRE_ID_PATH, "r", encoding="utf-8") as f:
                try:
                    remark_data = json.load(f)
                except json.JSONDecodeError:
                    remark_data = {}

        # 儲存問卷數據
        remark_data[decision_remark] = questionnaire_data
        with open(QUESTIONNAIRE_ID_PATH, "w", encoding="utf-8") as f:
            json.dump(remark_data, f, indent=4)

        print(f"Room ID {room_id} 的資料已成功更新。")
    else:
        print(f"Room ID {room_id} 的資料在過去24小時內已更新，無需再次操作。")

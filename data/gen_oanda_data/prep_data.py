import requests
import pandas as pd
import time


OANDA_TEST_ACCOUNT_ID
OANDA_TEST_TOKEN

time_frame = "M15"
instrument_list = ["EUR_CHF", "AUD_NZD", "CAD_JPY", "EUR_CAD"]

for instrument in instrument_list: 
    start_date = "2006-08-14T20:00:00.000000000Z"
    candle_length = 1000
    forex_data_list = []
    while candle_length > 0:
        url = f"https://api-fxpractice.oanda.com/v3/accounts/{OANDA_TEST_ACCOUNT_ID}/instruments/{instrument}/candles?granularity={time_frame}&count=5000&from={start_date}&includeFirst=False"
        headers = {"Authorization": f"Bearer {OANDA_TEST_TOKEN}"}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            candles = data["candles"]
            candle_length = len(candles)
            
            if candle_length == 0:
                print(f"Number of candles: {candle_length}")
                break
            
            start_date = candles[-1]["time"]
            for candle in candles:
                local_candle = {
                    "time": candle["time"],
                    "open": candle["mid"]["o"],
                    "high": candle["mid"]["h"],
                    "low": candle["mid"]["l"],
                    "close": candle["mid"]["c"],
                    "volume": candle["volume"]
                }
                forex_data_list.append(local_candle)
            print(f"Start Date: {start_date}")
            # break
            time.sleep(5)
        else:
            print(f"Error: {response.status_code}")
            break
        
    forex_data = pd.DataFrame(forex_data_list)
    forex_data.to_csv(f"{instrument}_{time_frame}_raw_data.csv", index=False)
    print(f"{instrument} data saved")
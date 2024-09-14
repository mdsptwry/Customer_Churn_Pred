import requests
import time

def ping_app():
    web_url = "https://predict-churn.streamlit.app/"
    while True:
        try:
            response = requests.get(web_url)
            if response.status_code == 200:
                print(f"Pring successful: {response.status_code}")
            else:
                print(f"Ping failed: {response.status_code}")
        except Exception as e:
            print(f"Error pinging the app:{e}")
        
        # wait for 10 mins (600 seconds) before pinging again
        time.sleep(600)
    
if __name__=="__main__":
    ping_app()

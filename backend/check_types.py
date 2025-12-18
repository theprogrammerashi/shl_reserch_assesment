import requests
from bs4 import BeautifulSoup
import time

URL = "https://www.shl.com/solutions/products/product-catalog/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

def check(params):
    try:
        r = requests.get(URL, params=params, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        headers = [th.get_text(strip=True) for th in soup.find_all("th", class_="custom__table-heading__title")]
        print(f"Params {params} found headers: {headers}")
    except Exception as e:
        print(f"Params {params} failed: {e}")

check({})
check({"start": 0})
check({"start": 0, "type": 1})
check({"start": 0, "type": 2})

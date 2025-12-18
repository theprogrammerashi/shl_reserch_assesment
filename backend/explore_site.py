import requests

url = "https://www.shl.com/solutions/products/product-catalog/"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

try:
    response = requests.get(url, headers=headers)
    print(f"Status Code: {response.status_code}")
    with open("page_dump.html", "w", encoding="utf-8") as f:
        f.write(response.text)
    print("Page dumped to page_dump.html")
except Exception as e:
    print(f"Error: {e}")

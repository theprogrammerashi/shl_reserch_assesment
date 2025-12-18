import requests
from bs4 import BeautifulSoup
import json
import time
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "https://www.shl.com"
CATALOG_URL = "https://www.shl.com/products/product-catalog/"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_FILE = os.path.join(DATA_DIR, "products.json")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Referer": "https://www.shl.com/"
}

def get_soup(url):
    for _ in range(3):
        try:
            response = requests.get(url, headers=HEADERS, timeout=15)
            if response.status_code == 200:
                return BeautifulSoup(response.text, "html.parser"), response.text
            time.sleep(2)
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            time.sleep(2)
    return None, None

def scrape_catalog():
    all_products = {}
    types_to_scrape = [1, 2, 3, 4] 
    page_size = 12
    
    print(f"Starting catalog scrape for types: {types_to_scrape}...")
    
    for t_type in types_to_scrape:
        start = 0
        consecutive_empty_pages = 0
        type_count = 0
        
        while True:
            url = f"{CATALOG_URL}?start={start}&type={t_type}"
            print(f"Scraping Type {t_type}: {url}")
            
            soup, html = get_soup(url)
            if not soup:
                break

            rows = soup.find_all("tr", attrs={"data-entity-id": True})
            
            if not rows:
                consecutive_empty_pages += 1
                if consecutive_empty_pages > 1:  
                    break
            else:
                consecutive_empty_pages = 0
            
            for row in rows:
                try:
                    title_col = row.find("td", class_="custom__table-heading__title")
                    link = title_col.find("a")
                    if not link: continue
                    
                    name = link.get_text(strip=True)
                    href = link["href"]
                    product_url = BASE_URL + href if href.startswith("/") else href
                    
                    # De-duplicate by URL
                    if product_url in all_products:
                        continue
                    
                    cols = row.find_all("td", class_="custom__table-heading__general")
                    remote = "Yes" if len(cols) > 0 and cols[0].find("span", class_="-yes") else "No"
                    adaptive = "Yes" if len(cols) > 1 and cols[1].find("span", class_="-yes") else "No"
                    
                    test_type_col = row.find("td", class_="product-catalogue__keys")
                    test_types = []
                    if test_type_col:
                        keys = test_type_col.find_all("span", class_="product-catalogue__key")
                        test_types = [k.get_text(strip=True) for k in keys]

                    all_products[product_url] = {
                        "name": name,
                        "url": product_url,
                        "remote_support": remote,
                        "adaptive_support": adaptive,
                        "test_type_codes": test_types 
                    }
                    type_count += 1
                except Exception as e:
                    print(f"Error parsing row: {e}")

            start += page_size
            time.sleep(0.3)
        
        print(f"Found {type_count} products for type {t_type}. Total unique so far: {len(all_products)}")

    products = list(all_products.values())
    print(f"Final catalog count: {len(products)}. Starting details scrape...")
    return products

def scrape_details(product):
    soup, _ = get_soup(product["url"])
    if not soup:
        return product
    
    try:
        # Description
        desc_set = False
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc and meta_desc.get("content"):
            product["description"] = meta_desc["content"]
            desc_set = True
            
        
        content = soup.find("div", class_="product-detail__content")
        if content:
            p = content.find("p")
            if p:
                product["description"] = p.get_text(strip=True)

        
        product["duration"] = 0
        for li in soup.find_all("li"):
            text = li.get_text(strip=True)
            if "Duration" in text:
                match = re.search(r'Duration.*?:.*?(\d+)', text, re.IGNORECASE)
                if match:
                    product["duration"] = int(match.group(1))
                    break
        
        
        mapping = {
            "A": "Ability & Aptitude",
            "B": "Biodata & Situational Judgement",
            "C": "Competencies",
            "D": "Development & 360",
            "E": "Assessment Exercises",
            "K": "Knowledge & Skills",
            "P": "Personality & Behavior",
            "S": "Simulations"
        }
        
        full_types = []
        if "test_type_codes" in product:
            for code in product["test_type_codes"]:
                if code in mapping:
                    full_types.append(mapping[code])
                else:
                    full_types.append(code)
            
           
        
        if not full_types:
             
             pass
             
        product["test_type"] = full_types
        
        del product["test_type_codes"]

    except Exception as e:
        print(f"Error details {product['name']}: {e}")

    return product

def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    products = scrape_catalog()
    
    
    if len(products) < 377:
        print(f"WARNING: Only found {len(products)} products. Goal was 377.")
    
    final_products = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_p = {executor.submit(scrape_details, p): p for p in products}
        for i, future in enumerate(as_completed(future_to_p)):
            try:
                final_products.append(future.result())
                if i % 20 == 0:
                    print(f"Details collected: {i}/{len(products)}")
            except Exception as e:
                print(f"Worker failed: {e}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_products, f, indent=4)
    print("Done.")

if __name__ == "__main__":
    main()

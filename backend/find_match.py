with open("page_dump.html", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if "Individual Test Solutions" in line:
            print(f"Found at line {i+1}")
            print(line.strip()[:200]) 

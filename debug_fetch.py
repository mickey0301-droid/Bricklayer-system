"""
診斷腳本：一步一步測試文章抓取流程
在 Briefings-cowork 資料夾裡執行：python debug_fetch.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from datetime import datetime, timedelta
from urllib.parse import quote
import requests
import xml.etree.ElementTree as ET

HEADERS = {"User-Agent": "Mozilla/5.0 (Briefings Debug)"}

end_time   = datetime.now()
start_time = end_time - timedelta(hours=24)

print(f"時間範圍：{start_time.strftime('%Y/%m/%d %H:%M')} → {end_time.strftime('%Y/%m/%d %H:%M')}")
print("=" * 60)

# 測試各種查詢格式
test_domain = "cna.com.tw"
queries = [
    f"site:{test_domain} after:{start_time.strftime('%Y/%m/%d')} before:{end_time.strftime('%Y/%m/%d')}",
    f"site:{test_domain} when:1d",
    f"site:{test_domain}",
    f"台灣 site:{test_domain}",
    f"(台灣 OR 中國) site:{test_domain}",
]

for q in queries:
    url = f"https://news.google.com/rss/search?q={quote(q)}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        try:
            root = ET.fromstring(r.text)
            items = root.findall(".//item")
            title_sample = items[0].findtext("title") if items else "（無）"
        except:
            items = []
            title_sample = "（XML 解析失敗）"
        print(f"[{len(items):3d} 篇] {q[:70]}")
        if items:
            print(f"        範例標題：{title_sample[:60]}")
    except Exception as e:
        print(f"[ERROR] {q[:70]}")
        print(f"        {e}")
    print()

# 也測試直接 RSS feed
print("=" * 60)
print("測試直接 RSS feed：")
direct_feeds = {
    "中央社": "https://www.cna.com.tw/rss/aall.aspx",
    "自由時報": "https://news.ltn.com.tw/rss/all.xml",
    "聯合報": "https://udn.com/rssfeed/news/2/6638?ch=news",
}
for name, feed_url in direct_feeds.items():
    try:
        r = requests.get(feed_url, headers=HEADERS, timeout=10)
        try:
            root = ET.fromstring(r.text)
            items = root.findall(".//item")
        except:
            items = []
        print(f"[{len(items):3d} 篇] {name}: {feed_url}")
    except Exception as e:
        print(f"[ERROR] {name}: {e}")

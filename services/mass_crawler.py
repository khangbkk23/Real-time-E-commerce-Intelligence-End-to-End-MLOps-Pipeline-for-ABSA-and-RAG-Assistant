import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
import time
import json
import random
import pandas as pd
from datetime import datetime
from urllib.parse import quote
import sys
import os

# --- CẤU HÌNH ---
PROFILE_PATH = "./myshopee_profile_data"
MAX_PRODUCTS_PER_CAT = 10
MAX_PAGES_PER_PROD = 50

class ShopeeMassCrawler:
    def __init__(self, headless=False):
        options = uc.ChromeOptions()
        options.add_argument(f"--user-data-dir={PROFILE_PATH}")
        options.set_capability("goog:loggingPrefs", {"performance": "ALL"})
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('--blink-settings=imagesEnabled=false')
        
        # Thêm timeout để tránh lỗi Read timed out
        options.add_argument("--dns-prefetch-disable")
        options.add_argument("--disable-gpu")

        if headless:
            options.add_argument('--headless=new')
        
        print("🚀 KHOỞI ĐỘNG CRAWLER (STABLE VERSION)...")
        self.driver = uc.Chrome(options=options, headless=headless, use_subprocess=True)
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        # [FIX] Set timeout cho việc load trang (30 giây)
        self.driver.set_page_load_timeout(30)
        
        self.full_data = [] 
        self.current_keyword = ""

    # ---------------------------------------------------
    # HÀM CLICK NEXT PAGE
    # ---------------------------------------------------
    def try_click_next_page(self):
        try:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight - 1200);")
            time.sleep(1)
            
            next_active_xpath = "//button[contains(@class, 'shopee-icon-button--right') and not(contains(@class, 'disabled'))]"
            next_disabled_xpath = "//button[contains(@class, 'shopee-icon-button--right') and contains(@class, 'disabled')]"

            if self.driver.find_elements(By.XPATH, next_disabled_xpath):
                print("Nút Next bị khóa (Đã hết trang).")
                return False

            # Tìm nút Next đang hoạt động
            btns = self.driver.find_elements(By.XPATH, next_active_xpath)
            target_btn = None
            
            for btn in btns:
                if btn.is_displayed() and btn.is_enabled():
                    target_btn = btn
                    break
            
            if target_btn:
                self.driver.execute_script("arguments[0].click();", target_btn)
                return True
            
            return False

        except Exception as e:
            # print(f"Debug Next Error: {e}")
            return False

    def save_current_batch(self):
        if not self.full_data: 
            print("There isn't any new to save.")
            return

        try:
            print(f"\nSAVING DATA FOR: {self.current_keyword.upper()}...")
            df = pd.DataFrame(self.full_data)
            if 'source_url' not in df.columns:
                df['source_url'] = "Unknown"

            df = df.drop_duplicates(subset=['username', 'comment', 'timestamp'])
            df = df[df['comment'].str.len() > 10]
            
            output_folder = "./datasets/raw"
            
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            
            safe_keyword = self.current_keyword.replace(' ', '_')
            file_name_only = f"dataset_{safe_keyword}_{int(time.time())}.csv"
            full_path = os.path.join(output_folder, file_name_only)
            
            cols = ['keyword', 'source_url', 'rating', 'date', 'variant', 'comment', 'username', 'timestamp']
            cols = [c for c in cols if c in df.columns]
            df = df[cols]
            
            df.to_csv(full_path, index=False, encoding='utf-8-sig')
            
            print(f"SAVED: {full_path}")
            print(f"Number of rows: {len(df)}")
            
            self.full_data = [] 
            
        except Exception as e:
            print(f"Error in saving file: {e}")

    def human_like_delay(self, min_sec=2, max_sec=5):
        time.sleep(random.uniform(min_sec, max_sec))

    def check_captcha_safe(self):
        try:
            if 'geetest' in self.driver.page_source.lower():
                self.wait_for_human()
                return
            selectors = ["//div[@class='geetest_window']", "//div[contains(text(), 'xác minh')]"]
            for s in selectors:
                if self.driver.find_elements(By.XPATH, s):
                    self.wait_for_human()
                    return
        except: pass

    def wait_for_human(self):
        print("\n" + "!"*50)
        print("🚨 PHÁT HIỆN CAPTCHA! GIẢI XONG NHẤN ENTER.")
        print("!"*50)
        sys.stdout.write('\a')
        sys.stdout.flush()
        try: input("⌨️  Waiting... ")
        except KeyboardInterrupt: raise KeyboardInterrupt 
        self.human_like_delay(3, 5)

    def process_network_log(self, logs):
        extracted = []
        for entry in logs:
            try:
                log_msg = json.loads(entry["message"])["message"]
                if "Network.responseReceived" not in log_msg["method"]: continue
                params = log_msg["params"]
                if "get_ratings" not in params["response"]["url"]: continue
                
                res = self.driver.execute_cdp_cmd("Network.getResponseBody", {"requestId": params["requestId"]})
                body = json.loads(res['body'])
                
                if 'data' in body and 'ratings' in body['data']:
                    items = body['data']['ratings']
                    for item in items:
                        variant = item["product_items"][0].get("model_name", "") if item.get("product_items") else ""
                        extracted.append({
                            "username": item.get('author_username', ''),
                            "rating": item.get('rating_star', 5),
                            "comment": item.get('comment', ''),
                            "variant": variant,
                            "timestamp": item.get('ctime', 0),
                            "date": datetime.fromtimestamp(item.get('ctime', 0)).strftime('%Y-%m-%d'),
                            "keyword": self.current_keyword,
                            "source_url": self.driver.current_url
                        })
            except: continue
        return extracted

    # ---------------------------------------------------
    # HÀM CRAWL 1 SẢN PHẨM (CÓ FIX LỖI TIMEOUT)
    # ---------------------------------------------------
    def crawl_single_product(self, url):
        print(f"   📦 SP: {url[:60]}...")
        
        try:
            self.driver.get(url)
        except Exception:
            print("      ➡️ Bỏ qua (Lỗi load trang).")
            return

        self.human_like_delay(4, 6)
        self.check_captcha_safe()
        self.driver.execute_script("window.scrollBy(0, 500);")
        try:
            self.driver.execute_script("""
                let tabs = document.querySelectorAll("div");
                for (let t of tabs) {
                    if(t.innerText.includes("Đánh Giá") && t.innerText.length < 20) { t.click(); break; }
                }
            """)
            time.sleep(2)
        except: pass
        
        page = 1
        count_total = 0
        empty_page_count = 0
        
        while True:
            if page > MAX_PAGES_PER_PROD:
                print(f"Dừng (Max {MAX_PAGES_PER_PROD} trang).")
                break

            # 2. Scroll trigger
            self.driver.execute_script("window.scrollBy(0, 1000);")
            time.sleep(1)
            self.driver.execute_script("window.scrollBy(0, 600);")
            self.human_like_delay(2, 4) 
            
            # 3. Lấy dữ liệu
            logs = self.driver.get_log("performance")
            new_data = self.process_network_log(logs)
            
            if new_data:
                self.full_data.extend(new_data)
                count_total += len(new_data)
                empty_page_count = 0
                print(".", end="", flush=True)
            else:
                empty_page_count += 1
                if empty_page_count >= 3: 
                    print(f"\n      🛑 Dừng (3 lần không thấy dữ liệu mới).")
                    break
            
            self.check_captcha_safe()
            if not self.try_click_next_page():
                print(f"\n      🛑 Hết trang (Page {page}).")
                break
                
            self.human_like_delay(3, 5)
            page += 1
        
        print(f" Done (+{count_total} reviews)")

    # ---------------------------------------------------
    # TÌM KIẾM SẢN PHẨM
    # ---------------------------------------------------
    def search_product_links(self, keyword):
        print(f"\n🔎 Tìm Top {MAX_PRODUCTS_PER_CAT} Bán Chạy: '{keyword}'...")
        url = f"https://shopee.vn/search?keyword={quote(keyword)}&sortBy=sales"
        
        try:
            self.driver.get(url)
        except Exception as e:
            print(f"❌ Lỗi load trang tìm kiếm: {e}")
            return []

        self.human_like_delay(5, 8)
        self.check_captcha_safe()
        
        for i in range(4):
            self.driver.execute_script(f"window.scrollBy(0, 1200);")
            self.human_like_delay(1, 2)
        
        links = []
        try:
            elements = self.driver.find_elements(By.CSS_SELECTOR, "a[data-sqe='link']")
            for elem in elements:
                href = elem.get_attribute("href")
                if href and "-i." in href: links.append(href)
        except: pass
        
        if not links:
            raw_links = self.driver.find_elements(By.TAG_NAME, "a")
            for l in raw_links:
                href = l.get_attribute("href")
                if href and "-i." in href and len(href) > 40: links.append(href)

        # [FIX] Đảm bảo chỉ lấy đúng số lượng đã config
        unique_links = list(set(links))[:MAX_PRODUCTS_PER_CAT]
        print(f"✅ Tìm thấy {len(unique_links)} sản phẩm.")
        return unique_links


    def run_multi_campaign(self, categories):
        print(f"BẮT ĐẦU: {len(categories)} DANH MỤC")
        
        try:
            for idx, cat in enumerate(categories):
                print(f"\n\n" + "#"*50)
                print(f"DANH MỤC [{idx+1}/{len(categories)}]: {cat.upper()}")
                print("#"*50)
                
                self.current_keyword = cat
                links = self.search_product_links(cat)
                
                if not links: continue

                for p_idx, link in enumerate(links):
                    print(f"\n🔸 [{p_idx+1}/{len(links)}] {cat}...")
                    
                    self.crawl_single_product(link)
                    
                    self.human_like_delay(6, 10)

                self.save_current_batch()
                print("Nghỉ 30s...")
                time.sleep(30)

        except KeyboardInterrupt:
            print("\n\n" + "!"*50)
            print("NGƯỜI DÙNG DỪNG (Ctrl + C)!")
            print("Đang lưu dữ liệu...")
            self.save_current_batch()
            print("!"*50)

    def close(self):
        try:
            self.driver.quit()
        except: pass

SHOPPING_LIST = [
    # Công nghệ
    # "robot hút bụi lau nhà", 
    # "đồng hồ thông minh thể thao",
    # "bàn phím cơ custom", 
    # "tai nghe chống ồn",
    # "camera wifi ngoài trời",
    "màn hình chuyên đồ hoạ",
    # # Mỹ phẩm
    # "serum vitamin c",
    # "kem dưỡng retinol",
    # "kem chống nắng cho da dầu", 
    # "nước tẩy trang cho da nhạy cảm",
    # # Gia dụng
    # "máy lọc không khí", "máy tăm nước", "ghế công thái học", 
    # "bàn chải điện", "nồi chiên không dầu"
]

if __name__ == "__main__":
    crawler = ShopeeMassCrawler(headless=False)
    try:
        crawler.run_multi_campaign(SHOPPING_LIST)
    except Exception as e:
        print(f"Critical System Error: {e}")
    finally:
        crawler.close()
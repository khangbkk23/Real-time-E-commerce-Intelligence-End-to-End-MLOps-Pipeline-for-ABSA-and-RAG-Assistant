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
PROFILE_PATH = "/home/dikhang_hcmut/myshopee_profile"
MAX_PRODUCTS_PER_CAT = 10     # Lấy Top 10 sản phẩm bán chạy nhất mỗi loại
MAX_PAGES_PER_PROD = 50       # Cố gắng lấy tới 50 trang (khoảng 2500 review/sp)

class ShopeeMassCrawler:
    def __init__(self, headless=False):
        options = uc.ChromeOptions()
        options.add_argument(f"--user-data-dir={PROFILE_PATH}")
        options.set_capability("goog:loggingPrefs", {"performance": "ALL"})
        options.add_argument('--disable-blink-features=AutomationControlled')
        # Tắt hình ảnh để load nhanh
        options.add_argument('--blink-settings=imagesEnabled=false')
        
        if headless:
            options.add_argument('--headless=new')
        
        print("🚀 KHOỞI ĐỘNG CRAWLER VỚI TÍNH NĂNG AUTO-SAVE...")
        self.driver = uc.Chrome(options=options, headless=headless, use_subprocess=True)
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        self.full_data = [] 
        self.current_keyword = ""

    # ---------------------------------------------------
    # HÀM CLICK NEXT PAGE (FIX LỖI KHÔNG CHUYỂN TRANG)
    # ---------------------------------------------------
    def try_click_next_page(self):
        try:
            # 1. Cuộn xuống đáy để load pagination
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight - 1200);")
            time.sleep(1)

            # 2. Tìm nút Next (Mũi tên phải)
            next_buttons_xpaths = [
                "//button[contains(@class, 'shopee-icon-button--right')]", 
                "//button[@class='shopee-icon-button shopee-icon-button--right']"
            ]

            target_btn = None
            for xpath in next_buttons_xpaths:
                try:
                    btns = self.driver.find_elements(By.XPATH, xpath)
                    for btn in btns:
                        # Kiểm tra nút có hiển thị và không bị disabled (mờ đi)
                        if btn.is_displayed() and btn.is_enabled():
                            target_btn = btn
                            break
                    if target_btn: break
                except: continue

            if target_btn:
                # 3. Dùng JavaScript Click (Xuyên vật cản)
                self.driver.execute_script("arguments[0].click();", target_btn)
                return True
            else:
                return False

        except Exception:
            return False

    # ---------------------------------------------------
    # HÀM LƯU FILE CSV
    # ---------------------------------------------------
    def save_current_batch(self):
        if not self.full_data: 
            print("⚠️ Chưa có dữ liệu mới để lưu.")
            return

        try:
            print(f"\n💾 ĐANG LƯU DỮ LIỆU CHO: {self.current_keyword.upper()}...")
            df = pd.DataFrame(self.full_data)
            # Lọc trùng
            df = df.drop_duplicates(subset=['username', 'comment', 'timestamp'])
            # Chỉ lấy comment có nội dung > 5 ký tự
            df = df[df['comment'].str.len() > 5]
            
            # Tên file theo từ khóa + timestamp để không bị ghi đè
            safe_keyword = self.current_keyword.replace(' ', '_')
            filename = f"dataset_{safe_keyword}_{int(time.time())}.csv"
            
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            
            print(f"✅ ĐÃ LƯU THÀNH CÔNG: {filename}")
            print(f"📊 Tổng số dòng: {len(df)}")
            
            # Reset buffer sau khi lưu xong
            self.full_data = [] 
            
        except Exception as e:
            print(f"❌ Lỗi lưu file: {e}")

    # ---------------------------------------------------
    # HUMAN CHECK & CAPTCHA
    # ---------------------------------------------------
    def human_like_delay(self, min_sec=2, max_sec=5):
        time.sleep(random.uniform(min_sec, max_sec))

    def check_captcha_safe(self):
        try:
            # Check nhanh
            if 'geetest' in self.driver.page_source.lower():
                self.wait_for_human()
                return

            selectors = ["//div[@class='geetest_window']", "//div[contains(text(), 'xác minh')]"]
            for s in selectors:
                elems = self.driver.find_elements(By.XPATH, s)
                if elems and elems[0].is_displayed():
                    self.wait_for_human()
                    return
        except: pass

    def wait_for_human(self):
        print("\n" + "!"*60)
        print("🚨 PHÁT HIỆN CAPTCHA! TẠM DỪNG.")
        print("👉 Giải xong nhấn [ENTER] để chạy tiếp.")
        print("👉 Nếu muốn DỪNG LUÔN, nhấn [Ctrl + C].")
        print("!"*60)
        sys.stdout.write('\a')
        sys.stdout.flush()
        
        # Chờ user nhấn Enter (hoặc Ctrl+C sẽ văng ra ngoài)
        input("⌨️  Đang chờ bạn... ")
        
        print("✅ Tiếp tục...")
        self.human_like_delay(3, 5)

    # ---------------------------------------------------
    # CORE CRAWL LOGIC
    # ---------------------------------------------------
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
                            "keyword": self.current_keyword
                        })
            except: continue
        return extracted

    def crawl_single_product(self, url):
        print(f"   📦 SP: {url[:60]}...")
        self.driver.get(url)
        self.human_like_delay(4, 6)
        self.check_captcha_safe()

        # Click Tab Đánh Giá
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
        
        product_reviews = []
        page = 1
        
        # VÒNG LẶP VÉT CẠN (WHILE TRUE)
        while True:
            # Giới hạn an toàn
            if page > MAX_PAGES_PER_PROD:
                print(f"      🛑 Đã đạt giới hạn {MAX_PAGES_PER_PROD} trang. Dừng SP này.")
                break

            # Scroll trigger API
            self.driver.execute_script("window.scrollBy(0, 1000);")
            time.sleep(1)
            self.driver.execute_script("window.scrollBy(0, 600);")
            self.human_like_delay(2, 4) 
            
            logs = self.driver.get_log("performance")
            new_data = self.process_network_log(logs)
            
            if new_data:
                product_reviews.extend(new_data)
                # In dấu chấm để biết đang chạy
                print(".", end="", flush=True)
            
            self.check_captcha_safe()

            # Thử click Next Page
            if not self.try_click_next_page():
                print(f"\n      🛑 Hết trang (Page {page}).")
                break
                
            self.human_like_delay(3, 5) # Chờ load trang mới
            page += 1
        
        print(f" Done ({len(product_reviews)} reviews)")
        return product_reviews

    def search_product_links(self, keyword):
        print(f"\n🔎 Tìm Top 10 Bán Chạy: '{keyword}'...")
        url = f"https://shopee.vn/search?keyword={quote(keyword)}&sortBy=sales"
        self.driver.get(url)
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

        unique_links = list(set(links))[:MAX_PRODUCTS_PER_CAT]
        print(f"✅ Tìm thấy {len(unique_links)} sản phẩm.")
        return unique_links

    # ---------------------------------------------------
    # HÀM CHẠY CHIẾN DỊCH (HỖ TRỢ CTRL+C)
    # ---------------------------------------------------
    def run_multi_campaign(self, categories):
        print(f"🚀 BẮT ĐẦU CHIẾN DỊCH: {len(categories)} DANH MỤC")
        print("💡 MẸO: Nhấn 'Ctrl + C' để DỪNG và LƯU FILE ngay lập tức.")
        
        try:
            for idx, cat in enumerate(categories):
                print(f"\n\n" + "#"*50)
                print(f"📌 DANH MỤC [{idx+1}/{len(categories)}]: {cat.upper()}")
                print("#"*50)
                
                self.current_keyword = cat
                links = self.search_product_links(cat)
                
                if not links: continue

                # Loop từng sản phẩm
                for p_idx, link in enumerate(links):
                    print(f"\n🔸 [{p_idx+1}/{len(links)}] {cat}...")
                    
                    reviews = self.crawl_single_product(link)
                    self.full_data.extend(reviews)
                    
                    # Nghỉ ngơi giữa các sản phẩm
                    self.human_like_delay(6, 10)

                # SAU KHI XONG 1 DANH MỤC -> LƯU FILE NGAY
                self.save_current_batch()
                
                print("💤 Nghỉ giải lao 30s trước khi qua danh mục mới...")
                time.sleep(30)

        except KeyboardInterrupt:
            print("\n\n" + "!"*50)
            print("🛑 NGƯỜI DÙNG ĐÃ DỪNG (Ctrl + C)!")
            print("🛑 Đang tiến hành lưu dữ liệu còn trong bộ nhớ...")
            self.save_current_batch()
            print("!"*50)

    def close(self):
        self.driver.quit()

# ---------------------------------------------------
# DANH SÁCH MẶT HÀNG ĐỂ CRAWL
# ---------------------------------------------------
SHOPPING_LIST = [
    # Công nghệ
    "tai nghe bluetooth", "chuột không dây", "bàn phím cơ", "sạc dự phòng", 
    # Thời trang
    "áo thun nam", "váy nữ", "giày sneaker", 
    # Mỹ phẩm
    "son môi", "kem chống nắng", "sữa rửa mặt", 
    # Gia dụng
    "bình giữ nhiệt", "nồi chiên không dầu", "gấu bông"
]

if __name__ == "__main__":
    crawler = ShopeeMassCrawler(headless=False)
    try:
        crawler.run_multi_campaign(SHOPPING_LIST)
    except Exception as e:
        print(f"❌ Critical Error: {e}")
    finally:
        crawler.close()
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
        
        print("KHỞI ĐỘNG CRAWLER (STABLE VERSION)...")
        self.driver = uc.Chrome(options=options, headless=headless, use_subprocess=True, version_main=144)
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        # Set timeout cho việc load trang (30 giây)
        self.driver.set_page_load_timeout(30)
        
        self.full_data = [] 
        self.current_keyword = ""
        self.seen_reviews = set()

    def try_click_next_page(self):
        print("      ➡️ Đang thử bấm Next Page...", end=" ")
        try:
            self.driver.execute_script("window.scrollBy(0, 600);")
            time.sleep(1)
            next_button_selectors = [
                "//button[contains(@class, 'shopee-icon-button--right')]",
                "//button[contains(@class, 'shopee-icon-button') and .//*[name()='svg' and contains(@class, 'icon-arrow-right')]]",
                "//div[@class='shopee-page-controller']//button[last()]"
            ]

            target_btn = None
            
            for xpath in next_button_selectors:
                try:
                    btns = self.driver.find_elements(By.XPATH, xpath)
                    for btn in btns:
                        class_attr = btn.get_attribute("class") or ""
                        disabled_attr = btn.get_attribute("disabled")
                        
                        if "disabled" in class_attr or disabled_attr is not None:
                            continue
                        
                        if btn.is_displayed():
                            target_btn = btn
                            break
                    if target_btn: break
                except: continue

            if target_btn:
                # 3. Click bằng JavaScript (Mạnh hơn click thường)
                self.driver.execute_script("arguments[0].click();", target_btn)
                print("✅ Click thành công!")
                time.sleep(3) # Chờ trang mới load
                return True
            else:
                print("Không tìm thấy nút Next (Hoặc đã hết trang).")
                return False

        except Exception as e:
            print(f"Lỗi Next Page: {e}")
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
    # def crawl_single_product(self, url):
    #     print(f"   📦 SP: {url[:60]}...")
        
    #     try:
    #         self.driver.get(url)
    #     except Exception:
    #         print("      ➡️ Bỏ qua (Lỗi load trang).")
    #         return

    #     self.human_like_delay(4, 6)
    #     self.check_captcha_safe()
    #     self.driver.execute_script("window.scrollBy(0, 500);")
    #     try:
    #         self.driver.execute_script("""
    #             let tabs = document.querySelectorAll("div");
    #             for (let t of tabs) {
    #                 if(t.innerText.includes("Đánh Giá") && t.innerText.length < 20) { t.click(); break; }
    #             }
    #         """)
    #         time.sleep(2)
    #     except: pass
        
    #     page = 1
    #     count_total = 0
    #     empty_page_count = 0
        
    #     while True:
    #         if page > MAX_PAGES_PER_PROD:
    #             print(f"Dừng (Max {MAX_PAGES_PER_PROD} trang).")
    #             break

    #         # 2. Scroll trigger
    #         self.driver.execute_script("window.scrollBy(0, 1000);")
    #         time.sleep(1)
    #         self.driver.execute_script("window.scrollBy(0, 600);")
    #         self.human_like_delay(2, 4) 
            
    #         # 3. Lấy dữ liệu
    #         logs = self.driver.get_log("performance")
    #         new_data = self.process_network_log(logs)
            
    #         if new_data:
    #             self.full_data.extend(new_data)
    #             count_total += len(new_data)
    #             empty_page_count = 0
    #             print(".", end="", flush=True)
    #         else:
    #             empty_page_count += 1
    #             if empty_page_count >= 3: 
    #                 print(f"\n      🛑 Dừng (3 lần không thấy dữ liệu mới).")
    #                 break
            
    #         self.check_captcha_safe()
    #         if not self.try_click_next_page():
    #             print(f"\n      🛑 Hết trang (Page {page}).")
    #             break
                
    #         self.human_like_delay(3, 5)
    #         page += 1
        
    #     print(f" Done (+{count_total} reviews)")
    
    def crawl_single_product(self, url):
        print(f"   📦 SP: {url[:60]}...")
        
        try:
            self.driver.get(url)
        except Exception:
            print("      ➡️ Bỏ qua (Lỗi load trang).")
            return

        self.human_like_delay(4, 6)
        self.check_captcha_safe()

        # 1. Click Tab "Đánh Giá"
        self.driver.execute_script("window.scrollBy(0, 500);")
        try:
            self.driver.execute_script("""
                let tabs = document.querySelectorAll("div");
                for (let t of tabs) {
                    if(t.innerText.includes("Đánh Giá") && t.innerText.length < 30) { t.click(); break; }
                }
            """)
            time.sleep(2)
        except: pass

        # 2. Định nghĩa mục tiêu
        target_filters = ["1 sao", "2 sao", "3 sao"] 
        
        # 3. Duyệt qua từng bộ lọc
        for target_name in target_filters:
            print(f"\n      🎯 Check filter: [{target_name.upper()}]...", end=" ")
            
            target_btn = None
            try:
                # Tìm lại elements mỗi vòng lặp
                current_filters = self.driver.find_elements(By.CSS_SELECTOR, "div[class*='product-rating-overview__filter']")
                
                # Fallback nếu selector trên không thấy
                if not current_filters:
                     current_filters = self.driver.find_elements(By.CSS_SELECTOR, ".product-rating-overview div")

                for btn in current_filters:
                    btn_text = btn.text.lower()
                    
                    if target_name in btn_text:
                        # [BẢO VỆ 1: NÉ NÚT RỖNG]
                        # Nếu nút chứa "(0)" hoặc kết thúc bằng "(0)" -> Bỏ qua
                        if "(0)" in btn_text or btn_text.strip().endswith("(0)"):
                            print(f"-> Trống (0 review). Skip.", end="")
                            target_btn = None
                        else:
                            target_btn = btn
                        break
            except: pass

            if target_btn:
                # Click nút
                self.driver.execute_script("arguments[0].click();", target_btn)
                print("✅ Click!", end=" ")
                time.sleep(3)
                
                self.driver.get_log("performance") 
            else:
                print("❌ Next.")
                continue 

            # ---------------------------------------------------------
            # BẮT ĐẦU CRAWL DATA CỦA FILTER HIỆN TẠI
            # ---------------------------------------------------------
            page = 1
            empty_count = 0
            count_filter = 0
            
            while True:
                if page > 10: break # Giới hạn 10 trang cho 1 sao

                self.driver.execute_script("window.scrollBy(0, 1000);")
                time.sleep(1)
                self.driver.execute_script("window.scrollBy(0, 600);")
                self.human_like_delay(2, 3) 
                
                logs = self.driver.get_log("performance")
                new_data = self.process_network_log(logs)
                
                if new_data:
                    unique_batch = []
                    for item in new_data:
                        # [BẢO VỆ 3: LỌC CỨNG (HARD FILTER)]
                        # Đây là chốt chặn cuối cùng. Nếu rating > 3 -> VỨT NGAY.
                        current_rating = item.get('rating', 5)
                        if current_rating > 3:
                            continue

                        # Logic chống trùng lặp
                        review_id = f"{item['username']}_{item['timestamp']}"
                        
                        if review_id not in self.seen_reviews:
                            self.seen_reviews.add(review_id)
                            item['source_url'] = url 
                            unique_batch.append(item)
                    
                    if unique_batch:
                        self.full_data.extend(unique_batch)
                        count_filter += len(unique_batch)
                        empty_count = 0
                        print(f"+{len(unique_batch)}", end=" ", flush=True)
                    else:
                        empty_count += 1
                else:
                    empty_count += 1

                if empty_count >= 2: break 

                self.check_captcha_safe()
                if not self.try_click_next_page(): break
                    
                page += 1
            
            print(f" -> Xong (+{count_filter} reviews)")
        
        print("Hoàn thành sản phẩm.")

    # ---------------------------------------------------
    # HÀM TÌM KIẾM (PHIÊN BẢN LẤY TỪ ĐẦU - GIỮ NGUYÊN THỨ TỰ)
    # ---------------------------------------------------
    def search_product_links(self, keyword):
        print(f"\n🔎 Tìm Top {MAX_PRODUCTS_PER_CAT} Bán Chạy Nhất: '{keyword}'...")
        
        # Sắp xếp theo Bán Chạy (Sales)
        url = f"https://shopee.vn/search?keyword={quote(keyword)}&sortBy=sales"
        
        try:
            self.driver.get(url)
        except Exception as e:
            print(f"❌ Lỗi load trang tìm kiếm: {e}")
            return []

        self.human_like_delay(5, 8)
        self.check_captcha_safe()

        for i in range(5):
            self.driver.execute_script(f"window.scrollBy(0, 1000);")
            time.sleep(1) # Chờ 1 chút cho hình ảnh/link hiện ra
        
        links = []
        try:
            elements = self.driver.find_elements(By.CSS_SELECTOR, "a[data-sqe='link']")
            
            for elem in elements:
                href = elem.get_attribute("href")
                if href and "-i." in href: # Link sản phẩm shopee luôn có chuỗi "-i." chứa shopid và itemid
                    links.append(href)
        except: pass
        if not links:
            print("⚠️ Selector chính không thấy, dùng Fallback...")
            raw_links = self.driver.find_elements(By.TAG_NAME, "a")
            for l in raw_links:
                href = l.get_attribute("href")
                if href and "-i." in href and len(href) > 40: 
                    links.append(href)
        seen = set()
        ordered_links = []
        for l in links:
            if l not in seen:
                ordered_links.append(l)
                seen.add(l)
        final_links = ordered_links[:MAX_PRODUCTS_PER_CAT]

        print(f"✅ Đã chọn {len(final_links)} sản phẩm (Top Sales).")
        return final_links


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
    "robot hút bụi mini", 
    "đồng hồ thông minh thể thao 99k",
    "bàn phím cơ giá rẻ", 
    "tai nghe chống ồn giá rẻ",
    "camera wifi giá rẻ",
    "màn hình giá rẻ",
]

if __name__ == "__main__":
    crawler = ShopeeMassCrawler(headless=False)
    try:
        crawler.run_multi_campaign(SHOPPING_LIST)
    except Exception as e:
        print(f"Critical System Error: {e}")
    finally:
        crawler.close()
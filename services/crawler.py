import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import json
import random
import pandas as pd
from datetime import datetime

from captcha_service import ShopeeCaptchaSolver

# ĐƯỜNG DẪN PROFILE CỦA BẠN (GIỮ NGUYÊN)
PROFILE_PATH = "/home/dikhang_hcmut/myshopee_profile"

class ShopeeAdvancedCrawler:
    def __init__(self):
        options = uc.ChromeOptions()
        options.add_argument(f"--user-data-dir={PROFILE_PATH}")
        options.set_capability("goog:loggingPrefs", {"performance": "ALL"})
        
        print("🚀 Khởi động Crawler...")
        self.driver = uc.Chrome(options=options, headless=False, use_subprocess=True)
        self.wait = WebDriverWait(self.driver, 10)

    def process_log_entry(self, logs):
        """Hàm lọc và trích xuất dữ liệu từ log network"""
        extracted_data = []
        for entry in logs:
            try:
                log = json.loads(entry["message"])["message"]
                if "Network.responseReceived" in log["method"]:
                    url = log["params"]["response"]["url"]
                    # Chỉ bắt API get_ratings
                    if "get_ratings" in url:
                        req_id = log["params"]["requestId"]
                        try:
                            # Lấy response body
                            res = self.driver.execute_cdp_cmd("Network.getResponseBody", {"requestId": req_id})
                            body = json.loads(res['body'])
                            
                            if 'data' in body and 'ratings' in body['data']:
                                items = body['data']['ratings']
                                for item in items:
                                    # Lấy thông tin biến thể (Màu/Size)
                                    variant = ""
                                    if item.get("product_items"):
                                        variant = item["product_items"][0].get("model_name", "")

                                    extracted_data.append({
                                        "username": item.get('author_username'),
                                        "rating": item.get('rating_star'),
                                        "comment": item.get('comment'),
                                        "variant": variant,
                                        "timestamp": item.get('ctime'),
                                        "date": datetime.fromtimestamp(item.get('ctime')).strftime('%Y-%m-%d %H:%M:%S'),
                                        "is_anonymous": item.get('anonymous'),
                                        "region": item.get('region', 'VN')
                                    })
                        except:
                            pass # Bỏ qua các request lỗi hoặc ko decode được
            except:
                pass
        return extracted_data

    def click_next_page(self):
        """Tìm và click nút Next trang"""
        try:
            # Nút Next thường là icon mũi tên phải trong pagination
            next_btn = self.driver.find_element(By.XPATH, "//button[contains(@class, 'shopee-icon-button--right')]")
            if next_btn.is_enabled():
                self.driver.execute_script("arguments[0].click();", next_btn)
                return True
        except:
            return False
        return False

    def crawl(self, url, max_pages=5):
        print(f"🔗 Truy cập: {url}")
        self.driver.get(url)
        time.sleep(5) # Chờ load init

        self.handle_antibot()
        all_reviews = []
        
        # 1. Click Tab Đánh giá
        try:
            self.driver.execute_script("""
                let tabs = document.querySelectorAll("div");
                for (let tab of tabs) {
                    if (tab.innerText.includes("Đánh Giá") && tab.innerText.length < 20) {
                        tab.click(); break;
                    }
                }
            """)
            time.sleep(3)
        except: pass

        # 2. Vòng lặp phân trang
        for page in range(1, max_pages + 1):
            print(f"📄 Đang xử lý trang {page}...")
            
            # Scroll nhẹ để trigger load (quan trọng)
            self.driver.execute_script("window.scrollBy(0, 600);")
            time.sleep(2)
            self.driver.execute_script("window.scrollBy(0, 400);")
            time.sleep(3) # Chờ API phản hồi
            
            # Lấy Logs & Parse
            logs = self.driver.get_log("performance")
            new_data = self.process_log_entry(logs)
            
            if new_data:
                print(f"   -> Bắt được {len(new_data)} reviews từ network.")
                all_reviews.extend(new_data)
            else:
                print("   -> Không thấy gói tin API nào.")

            # Thử sang trang tiếp theo
            if page < max_pages:
                if self.click_next_page():
                    print("   -> Đã click Next page. Chờ load...")
                    time.sleep(random.uniform(3, 5))
                else:
                    print("🛑 Không tìm thấy nút Next hoặc đã hết trang.")
                    break
        
        return all_reviews

    def save_csv(self, data, filename="shopee_full_reviews.csv"):
        if not data:
            print("⚠️ Không có dữ liệu để lưu.")
            return
            
        # Deduplicate: Loại bỏ các dòng trùng lặp dựa trên username và comment
        df = pd.DataFrame(data)
        initial_len = len(df)
        df = df.drop_duplicates(subset=['username', 'comment', 'timestamp'])
        
        print(f"📊 Tổng thu được: {initial_len} | Sau khi lọc trùng: {len(df)}")
        
        # Sắp xếp cột cho đẹp
        cols = ['date', 'username', 'rating', 'variant', 'comment', 'timestamp']
        df = df[cols]
        
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"💾 Đã lưu file: {filename}")

    def close(self):
        self.driver.quit()
    
    def handle_antibot(self):
        """Hàm check và xử lý bot"""
        solver = ShopeeCaptchaSolver(self.driver)
        
        # Nếu thấy captcha
        if solver.is_captcha_present():
            print("⚠️ Bị chặn bởi Slider Captcha.")
            
            # Thử giải tối đa 3 lần
            for i in range(3):
                print(f"🔄 Thử giải lần {i+1}...")
                if solver.solve():
                    print("🎉 Giải thành công, tiếp tục crawl.")
                    time.sleep(3)
                    return True
                else:
                    # Nếu thất bại, refresh trang để lấy hình mới dễ hơn
                    self.driver.refresh()
                    time.sleep(5)
            
            return False
        return True

# --- MAIN RUN ---
if __name__ == "__main__":
    crawler = ShopeeAdvancedCrawler()
    try:
        # Thay link sản phẩm của bạn vào đây
        product_url = "https://shopee.vn/%C3%81o-Kho%C3%A1c-D%C3%B9-Ch%E1%BB%91ng-N%E1%BA%AFng-Nam-Couple-TX-UV-Pro-Windbreaker-MOK-1058-i.83192592.28400443877"
        
        # Chạy crawl 5 trang
        data = crawler.crawl(product_url, max_pages=5)
        crawler.save_csv(data)
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
    finally:
        crawler.close()
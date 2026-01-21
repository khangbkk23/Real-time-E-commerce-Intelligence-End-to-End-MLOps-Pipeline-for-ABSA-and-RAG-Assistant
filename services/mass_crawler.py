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

PROFILE_PATH = "/home/dikhang_hcmut/myshopee_profile"
MAX_PRODUCTS = 5
MAX_PAGES_PER_PROD = 3
CAPTCHA_WAIT_TIMEOUT = 300

class ShopeeMassCrawler:
    def __init__(self, headless=False):
        options = uc.ChromeOptions()
        options.add_argument(f"--user-data-dir={PROFILE_PATH}")
        options.set_capability("goog:loggingPrefs", {"performance": "ALL"})
        options.add_argument('--disable-blink-features=AutomationControlled')
        
        if headless:
            options.add_argument('--headless=new')
        
        print("🚀 Starting crawler...")
        self.driver = uc.Chrome(options=options, headless=headless, use_subprocess=True)
        
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        self.captcha_detected_count = 0
        self.total_reviews = 0
        self.products_crawled = 0
        self.consecutive_success = 0
        self.check_captcha_every = 1

    def human_like_delay(self, min_sec=2, max_sec=5):
        time.sleep(random.uniform(min_sec, max_sec))

    def should_check_captcha(self):
        if self.products_crawled == 0:
            return True
        
        if self.consecutive_success >= 3:
            self.check_captcha_every = 5
        
        if self.consecutive_success >= 5:
            self.check_captcha_every = 10
        
        return self.products_crawled % self.check_captcha_every == 0
    
    def detect_captcha(self):
        try:
            captcha_selectors = [
                "//canvas[@class='geetest_canvas_slice']",
                "//div[@class='geetest_window']",
                "//div[contains(@class, 'geetest_panel')]",
            ]
            
            for selector in captcha_selectors:
                try:
                    elements = self.driver.find_elements(By.XPATH, selector)
                    if elements and elements[0].is_displayed():
                        width = elements[0].size.get('width', 0)
                        height = elements[0].size.get('height', 0)
                        if width > 50 and height > 50:
                            return True
                except:
                    continue
            
            page_text = self.driver.page_source.lower()
            if 'geetest_radar' in page_text or 'geetest_holder' in page_text:
                return True
            
        except:
            pass
        
        return False

    def wait_for_human_solve_captcha(self):
        self.captcha_detected_count += 1
        self.consecutive_success = 0
        self.check_captcha_every = 1
        
        print("\n" + "="*70)
        print("🚨 CAPTCHA DETECTED!")
        print("="*70)
        print(f"📊 Captcha count: {self.captcha_detected_count}")
        print(f"📈 Reviews collected: {self.total_reviews}")
        print("\n⏸️  CRAWLER PAUSED")
        print("👉 Please solve the captcha manually in the browser")
        print("👉 Press [ENTER] when done to continue...")
        print("="*70 + "\n")
        
        sys.stdout.write('\a')
        sys.stdout.flush()
        
        start_wait = time.time()
        input("⌨️  Waiting... ")
        elapsed = time.time() - start_wait
        
        if elapsed > CAPTCHA_WAIT_TIMEOUT:
            print("⚠️ Timeout - stopping crawler")
            return False
        
        print("✅ Resuming...")
        self.human_like_delay(3, 5)
        return True

    def safe_action(self, action_func, action_name="Action", force_check=False):
        try:
            if force_check or self.should_check_captcha():
                if self.detect_captcha():
                    if not self.wait_for_human_solve_captcha():
                        return False
            
            result = action_func()
            self.human_like_delay(1, 3)
            return result
            
        except Exception as e:
            print(f"❌ {action_name} failed: {e}")
            return False

    def search_product_links(self, keyword):
        print(f"\n🔎 Searching: '{keyword}'...")
        url = f"https://shopee.vn/search?keyword={quote(keyword)}&sortBy=sales"
        
        self.driver.get(url)
        self.human_like_delay(5, 8)
        
        if self.detect_captcha():
            if not self.wait_for_human_solve_captcha():
                return []
        
        for i in range(5):
            self.driver.execute_script(f"window.scrollBy(0, {random.randint(800, 1200)});")
            self.human_like_delay(2, 4)
        
        product_links = []
        try:
            elements = self.driver.find_elements(By.CSS_SELECTOR, "a[data-sqe='link']")
            for elem in elements:
                link = elem.get_attribute("href")
                if link and "-i." in link and len(link) > 50:
                    product_links.append(link)
                    if len(product_links) >= MAX_PRODUCTS:
                        break
        except:
            pass
        
        if not product_links:
            links = self.driver.find_elements(By.TAG_NAME, "a")
            for l in links:
                href = l.get_attribute("href")
                if href and "-i." in href and len(href) > 50:
                    product_links.append(href)
                    if len(product_links) >= MAX_PRODUCTS:
                        break
        
        product_links = list(set(product_links))[:MAX_PRODUCTS]
        print(f"✅ Found {len(product_links)} products")
        return product_links

    def process_network_log(self, logs):
        extracted = []
        for entry in logs:
            try:
                log_msg = json.loads(entry["message"])["message"]
                if "Network.responseReceived" not in log_msg["method"]:
                    continue
                
                params = log_msg["params"]
                if "get_ratings" not in params["response"]["url"]:
                    continue
                
                res = self.driver.execute_cdp_cmd("Network.getResponseBody", {"requestId": params["requestId"]})
                body = json.loads(res['body'])
                
                if 'data' not in body or 'ratings' not in body['data']:
                    continue
                
                items = body['data']['ratings']
                for item in items:
                    variant = ""
                    if item.get("product_items"):
                        variant = item["product_items"][0].get("model_name", "")
                    
                    extracted.append({
                        "username": item.get('author_username', ''),
                        "rating": item.get('rating_star', 5),
                        "comment": item.get('comment', ''),
                        "variant": variant,
                        "timestamp": item.get('ctime', 0),
                        "date": datetime.fromtimestamp(item.get('ctime', 0)).strftime('%Y-%m-%d'),
                        "region": item.get('region', 'VN')
                    })
            except:
                continue
        
        return extracted

    def crawl_single_product(self, url):
        print(f"   📦 Crawling: {url[:60]}...")
        self.driver.get(url)
        self.human_like_delay(5, 8)
        
        if self.products_crawled < 2 or self.should_check_captcha():
            if self.detect_captcha():
                if not self.wait_for_human_solve_captcha():
                    return []
        
        self.driver.execute_script("window.scrollBy(0, 500);")
        self.human_like_delay(1, 2)
        
        try:
            script = """
                let tabs = document.querySelectorAll("div");
                for (let t of tabs) {
                    if(t.innerText.includes("Đánh Giá") && t.innerText.length < 20) {
                        t.click();
                        break;
                    }
                }
            """
            self.driver.execute_script(script)
            self.human_like_delay(2, 4)
        except:
            pass
        
        product_reviews = []
        
        for page in range(1, MAX_PAGES_PER_PROD + 1):
            self.driver.execute_script("window.scrollBy(0, 1000);")
            self.human_like_delay(2, 3)
            self.driver.execute_script("window.scrollBy(0, 500);")
            self.human_like_delay(2, 4)
            
            logs = self.driver.get_log("performance")
            new_data = self.process_network_log(logs)
            
            if new_data:
                product_reviews.extend(new_data)
                self.total_reviews += len(new_data)
            
            if page < MAX_PAGES_PER_PROD:
                try:
                    btn = self.driver.find_element(By.XPATH, "//button[contains(@class, 'shopee-icon-button--right')]")
                    if btn.is_enabled():
                        self.driver.execute_script("arguments[0].click();", btn)
                        self.human_like_delay(5, 8)
                    else:
                        break
                except:
                    break
        
        return product_reviews

    def run_campaign(self, keyword):
        links = self.search_product_links(keyword)
        
        if not links:
            print("❌ No products found")
            return
        
        full_data = []
        print(f"\n🚀 Starting crawl: {len(links)} products")
        
        for idx, link in enumerate(links):
            print(f"\n[{idx+1}/{len(links)}] Product...")
            
            try:
                reviews = self.crawl_single_product(link)
                
                for r in reviews:
                    r['source_url'] = link
                    r['keyword'] = keyword
                
                full_data.extend(reviews)
                self.products_crawled += 1
                
                if len(reviews) > 0:
                    self.consecutive_success += 1
                    print(f"   ✅ Collected: {len(reviews)} reviews (Success streak: {self.consecutive_success})")
                    
                    if self.consecutive_success >= 3:
                        print(f"   🎯 Stable mode: Checking captcha every {self.check_captcha_every} products")
                else:
                    self.consecutive_success = 0
                
                sleep_time = random.randint(10, 20)
                print(f"   💤 Sleeping {sleep_time}s...")
                time.sleep(sleep_time)
                
            except Exception as e:
                print(f"❌ Error: {e}")
                continue
        
        if full_data:
            df = pd.DataFrame(full_data)
            df = df.drop_duplicates(subset=['username', 'comment', 'timestamp'])
            df = df[df['comment'].str.len() > 5]
            
            filename = f"dataset_{keyword.replace(' ', '_')}_{int(time.time())}.csv"
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            
            print(f"\n🎉 COMPLETED!")
            print(f"   📊 Total reviews: {len(df)}")
            print(f"   🚨 Captcha encounters: {self.captcha_detected_count}")
            print(f"   💾 Saved to: {filename}")
        else:
            print("❌ No data collected")

    def close(self):
        self.driver.quit()


if __name__ == "__main__":
    crawler = ShopeeMassCrawler(headless=False)
    
    try:
        crawler.run_campaign("áo khoác nam")
    except KeyboardInterrupt:
        print("\n⚠️ Stopped by user")
    except Exception as e:
        print(f"❌ Critical error: {e}")
    finally:
        crawler.close()
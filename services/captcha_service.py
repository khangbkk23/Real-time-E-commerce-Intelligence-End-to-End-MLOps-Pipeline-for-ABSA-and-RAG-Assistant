import cv2
import numpy as np
import time
import random
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By

class ShopeeCaptchaSolver:
    def __init__(self, driver):
        self.driver = driver

    def is_captcha_present(self):
        """Kiểm tra xem popup slider có hiện ra không"""
        try:
            # Class chứa popup slider của Shopee (có thể thay đổi tùy thời điểm)
            # Thường là modal chứa ảnh
            element = self.driver.find_element(By.CSS_SELECTOR, ".shopee-popup__container") 
            # Hoặc tìm cái thanh slider
            # self.driver.find_element(By.XPATH, "//div[contains(@class, 'slider')]")
            return True
        except:
            return False

    def get_captcha_image(self):
        """Chụp ảnh phần chứa slider để xử lý"""
        try:
            # Tìm element chứa ảnh gốc (background)
            # Lưu ý: Cần Inspect Element thực tế trên Shopee để lấy đúng class
            # Đây là selector ví dụ thường thấy
            img_container = self.driver.find_element(By.CSS_SELECTOR, "div.shopee-popup__container")
            
            # Chụp màn hình element đó
            screenshot = img_container.screenshot_as_png
            
            # Convert sang định dạng OpenCV
            nparr = np.frombuffer(screenshot, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            print(f"❌ Lỗi chụp ảnh captcha: {e}")
            return None

    def find_gap_offset(self, img):
        """Dùng OpenCV để tìm vị trí mảnh ghép còn thiếu"""
        # 1. Chuyển sang ảnh xám
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Làm mờ để giảm nhiễu
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 3. Phát hiện cạnh (Canny Edge Detection)
        canny = cv2.Canny(blurred, 200, 450)
        
        # 4. Tìm contours (đường viền)
        contours, _ = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # 5. Logic tìm mảnh ghép: Mảnh ghép thường là hình vuông/chữ nhật có kích thước nhất định
        # Shopee puzzle thường khoảng 40x40 đến 60x60 pixel
        best_x = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Lọc nhiễu: Chỉ lấy khung có kích thước hợp lý với mảnh ghép
            if 30 < w < 80 and 30 < h < 80:
                # Mảnh ghép thật thường nằm bên phải (x > 50) chứ ko nằm sát lề trái
                if x > 50: 
                    best_x = x
                    # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2) # Debug vẽ hình
                    break
        
        return best_x

    def human_drag(self, slider_element, distance):
        """Kéo chuột giả lập hành vi người"""
        action = ActionChains(self.driver)
        
        # Click và giữ chuột
        action.click_and_hold(slider_element).perform()
        time.sleep(random.uniform(0.1, 0.5))
        
        # Chia khoảng cách thành các đoạn nhỏ để kéo (Ease-out)
        # Kéo nhanh lúc đầu, chậm dần lúc cuối
        current_pos = 0
        remain_dist = distance
        
        # Đoạn 1: Kéo nhanh (80% quãng đường)
        while current_pos < distance * 0.8:
            move = random.randint(10, 20)
            if current_pos + move > distance: break
            
            # Thêm độ lệch Y tí xíu (tay người ko bao giờ kéo thẳng tắp 100%)
            offset_y = random.randint(-2, 2)
            
            action.move_by_offset(move, offset_y).perform()
            current_pos += move
            remain_dist -= move
            time.sleep(random.uniform(0.01, 0.03))
            
        # Đoạn 2: Kéo chậm (tinh chỉnh vào khớp)
        while remain_dist > 0:
            move = random.randint(2, 5)
            if move > remain_dist: move = remain_dist
            
            offset_y = random.randint(-1, 1)
            action.move_by_offset(move, offset_y).perform()
            remain_dist -= move
            time.sleep(random.uniform(0.05, 0.1)) # Chậm lại
            
        # Đoạn 3: Thả tay
        time.sleep(random.uniform(0.1, 0.3))
        action.release().perform()

    def solve(self):
        """Hàm chính để gọi từ bên ngoài"""
        if not self.is_captcha_present():
            return False
            
        print("🧩 Phát hiện Captcha! Đang thử giải tự động...")
        time.sleep(2) # Chờ ảnh load đủ
        
        # 1. Lấy ảnh
        img = self.get_captcha_image()
        if img is None: return False
        
        # 2. Tính khoảng cách cần kéo
        distance = self.find_gap_offset(img)
        print(f"   -> Khoảng cách tính toán: {distance}px")
        
        if distance == 0:
            print("❌ Không tìm thấy mảnh ghép bằng OpenCV.")
            return False

        # 3. Tìm nút kéo slider
        try:
            # Selector của nút kéo (cần inspect để lấy chính xác class hiện tại)
            slider_btn = self.driver.find_element(By.CSS_SELECTOR, ".shopee-popup__slider-btn") 
            
            # 4. Thực hiện kéo
            # Lưu ý: Cần hiệu chỉnh tỉ lệ (scale) nếu ảnh web hiển thị khác size ảnh gốc
            # Shopee thường scale ảnh. Thử nghiệm thực tế thường nhân hệ số, ví dụ 1.0 hoặc biến động
            self.human_drag(slider_btn, distance)
            
            time.sleep(3)
            # Kiểm tra xem còn captcha không
            if not self.is_captcha_present():
                print("✅ Đã vượt qua Captcha!")
                return True
            else:
                print("❌ Giải thất bại (Kéo sai vị trí).")
                return False
                
        except Exception as e:
            print(f"❌ Lỗi thao tác kéo: {e}")
            return False
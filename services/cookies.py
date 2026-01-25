import undetected_chromedriver as uc
import time
import os

CURRENT_DIR = os.getcwd()
PROFILE_PATH = os.path.join(CURRENT_DIR, "myshopee_profile_data")

def create_profile():
    print(f"Đang khởi tạo Profile tại: {PROFILE_PATH}")
    
    options = uc.ChromeOptions()
    options.add_argument(f"--user-data-dir={PROFILE_PATH}")
    
    driver = uc.Chrome(options=options, headless=False, use_subprocess=True, version_main=144)

    print("Đang vào Shopee...")
    driver.get("https://shopee.vn/buyer/login")
    
    print("BẠN CÓ 2 PHÚT ĐỂ ĐĂNG NHẬP THỦ CÔNG & KÉO CAPTCHA...")
    print("Hãy tick vào nút 'Nhớ mật khẩu' hoặc 'Duy trì đăng nhập'")
    time.sleep(120) 
    
    print("Đã xong bước tạo Profile. Đóng trình duyệt.")
    driver.quit()

if __name__ == "__main__":
    create_profile()
import undetected_chromedriver as uc
import time
import os

# Đường dẫn để lưu Profile Chrome (Nên để trong thư mục Linux của bạn cho nhanh)
# Lưu ý: Đừng để trong /mnt/e/ (Windows) vì dễ lỗi file permission trên WSL
PROFILE_PATH = "/home/dikhang_hcmut/myshopee_profile" 

def create_profile():
    print(f"🚀 Đang khởi tạo Profile tại: {PROFILE_PATH}")
    
    options = uc.ChromeOptions()
    # Dòng lệnh quan trọng nhất: Chỉ định thư mục lưu dữ liệu
    options.add_argument(f"--user-data-dir={PROFILE_PATH}")
    
    # Mở Chrome lên
    driver = uc.Chrome(options=options, headless=False, use_subprocess=True)
    
    print("🌍 Đang vào Shopee...")
    driver.get("https://shopee.vn/buyer/login")
    
    print("⚠️ BẠN CÓ 2 PHÚT ĐỂ ĐĂNG NHẬP THỦ CÔNG & KÉO CAPTCHA...")
    print("👉 Hãy tick vào nút 'Nhớ mật khẩu' hoặc 'Duy trì đăng nhập'")
    time.sleep(120) 
    
    print("✅ Đã xong bước tạo Profile. Đóng trình duyệt.")
    driver.quit()

if __name__ == "__main__":
    create_profile()
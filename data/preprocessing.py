import pandas as pd
import re
import emoji
import codecs
from underthesea import word_tokenize
import os

def clean_str(string):
    if not string:
        return ""
    allowed_chars = r"aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬ" \
                    r"bBcCdDđĐ" \
                    r"eEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆ" \
                    r"fFgGhHiIìÌỉỈĩĨíÍịỊ" \
                    r"jJkKlLmMnNoOòÒỏỎõÕóÓọỌ" \
                    r"ôÔồỒổỔỗỖốỐộỘ" \
                    r"ơƠờỜởỞỡỠớỚợỢ" \
                    r"pPqQrRsStTuUùÙủỦũŨúÚụỤ" \
                    r"ưƯừỪửỬữỮứỨựỰ" \
                    r"vVwWxXyYỳỲỷỶỹỸýÝỵỴzZ" \
                    r"0-9\(\),!?'\.\-\/" 
    
    text = re.sub(f"[^{allowed_chars}]", " ", string)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def text_lowercase(string):
    return string.lower() if string else ""

def tokenize(strings):
    if not strings:
        return []
    try:
        return word_tokenize(strings, format="text")
    except:
        return strings

def load_stopwords(filepath='vietnamese-stopwords.txt'):
    stopwords = set()
    try:
        with codecs.open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                stopword = line.strip().lower()
                if stopword:
                    stopwords.add(stopword)
    except FileNotFoundError:
        print(f"Warning: Không tìm thấy file {filepath}. Bỏ qua bước lọc stopwords.")
    return stopwords

STOPWORDS = load_stopwords()

def remove_stopwords(text):
    if not text:
        return ""
    words = text.split()
    filtered_words = [w for w in words if w not in STOPWORDS]
    return " ".join(filtered_words)

def normalize_teencode(text):
    teencode_dict = {
        "k": "không", "ko": "không", "kh": "không", "khg": "không", "hok": "không",
        "sp": "sản phẩm", "shop": "cửa hàng", 
        "dc": "được", "đc": "được", "dk": "được",
        "r": "rồi", "rùi": "rồi", 
        "ok": "tốt", "oke": "tốt", "okay": "tốt",
        "giao hàng": "vận chuyển", "ship": "vận chuyển",
        "feedback": "đánh giá", "fb": "facebook",
        "sz": "size", "kđ": "không đều", "nt": "như thế",
        "m": "mình", "mik": "mình", "b": "bạn",
        "trc": "trước", "ntn": "như thế này"
    }
    words = text.split()
    return " ".join([teencode_dict.get(w, w) for w in words])

def full_preprocessing(text):
    if not isinstance(text, str):
        return ""
    
    text = emoji.replace_emoji(text, replace='')
    text = text_lowercase(text)
    text = normalize_teencode(text)
    text = clean_str(text)
    text = tokenize(text)
    text = remove_stopwords(text)
    
    return text

def map_sentiment(rating):
    if rating <= 2:
        return 0
    elif rating == 3:
        return 1
    else:
        return 2

if __name__ == "__main__":
    input_file = "datasets/dataset_robot_hút_bụi_lau_nhà_1769048720.csv"
    
    output_dir = "datasets/preprocessed"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    try:
        print("Loading data...")
        df = pd.read_csv(input_file)
        print(f"Original size: {len(df)} rows")

        print("Processing text...")
        df['clean_comment'] = df['comment'].apply(full_preprocessing)

        print("Mapping labels...")
        df['label'] = df['rating'].apply(map_sentiment)

        initial_count = len(df)
        df = df[df['clean_comment'].str.split().str.len() > 3]
        print(f"Removed {initial_count - len(df)} empty/short rows.")

        base_name = os.path.basename(input_file)
        output_file = os.path.join(output_dir, f"clean_{base_name}")
        
        final_columns = ['clean_comment', 'label', 'comment', 'rating', 'source_url', 'keyword', 'date']
        
        available_cols = [c for c in final_columns if c in df.columns]
        df[available_cols].to_csv(output_file, index=False, encoding='utf-8-sig')

        print("PREPROCESSING COMPLETED SUCCESSFULY!")
        print(f"Saved to: {output_file}")
        
        print(df[['clean_comment', 'label']].head(5))

    except FileNotFoundError:
        print(f"❌ Error: Cannot find file '{input_file}'")
    except Exception as e:
        print(f"❌ Error: {e}")
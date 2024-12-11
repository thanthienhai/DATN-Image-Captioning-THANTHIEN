import google.generativeai as genai
import os
from dotenv import load_dotenv

class CaptionGenerator:
    def __init__(self, model_name='gemini-1.5-flash-8b'):
        """
        Khởi tạo Caption Generator
        
        Args:
            model_name (str): Tên model Gemini
        """
        # Nạp biến môi trường
        load_dotenv()
        
        # Lấy API key
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("Không tìm thấy Google API Key")
        
        # Cấu hình API
        genai.configure(api_key=api_key)
        
        # Khởi tạo model
        self.model = genai.GenerativeModel(model_name)
    
    def generate_caption(self, image, prompt=None):
        """
        Sinh caption cho ảnh
        
        Args:
            image (PIL.Image): Ảnh đầu vào
            prompt (str, optional): Câu nhắc tùy chỉnh
        
        Returns:
            str: Caption của ảnh
        """
        try:
            # Prompt mặc định nếu không có
            if prompt is None:
                prompt = "Hãy mô tả chi tiết những gì bạn thấy trong hình ảnh này."
            
            # Sinh caption
            response = self.model.generate_content([image, prompt])
            
            return response.text
        except Exception as e:
            print(f"Lỗi khi sinh caption: {e}")
            return None
    
    def multi_language_caption(self, image, languages=None):
        """
        Sinh caption với nhiều ngôn ngữ
        
        Args:
            image (PIL.Image): Ảnh đầu vào
            languages (list): Danh sách ngôn ngữ
        
        Returns:
            dict: Các caption theo ngôn ngữ
        """
        if languages is None:
            languages = ['vi', 'en']
        
        captions = {}
        for lang in languages:
            prompt = f"Hãy mô tả ngắn gọn những gì bạn thấy trong hình ảnh này bằng tiếng {lang}. Tập trung vào các chủ thể lớn trong hình. Mô tả trong 1 câu văn 10-20 từ. Không được đưa ra các tên riêng. Ví dụ: Two young guys with shaggy hair look at their hands while hanging out in the yard; Two young  White males are outside near many bushes; Two friends enjoy time spent together."
            captions[lang] = self.generate_caption(image, prompt)
        
        return captions
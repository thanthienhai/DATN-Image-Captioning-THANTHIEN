import PIL.Image
import io
import numpy as np
import httpx

class ImageProcessor:
    @staticmethod
    def resize_image(image, target_size=(224, 224), maintain_aspect_ratio=True):
        """
        Resize ảnh với các tùy chọn nâng cao
        
        Args:
            image (PIL.Image or numpy.array): Ảnh đầu vào
            target_size (tuple): Kích thước mục tiêu (width, height)
            maintain_aspect_ratio (bool): Giữ nguyên tỷ lệ ảnh
        
        Returns:
            PIL.Image: Ảnh đã được resize
        """
        # Nếu là numpy array, chuyển sang PIL Image
        if isinstance(image, np.ndarray):
            image = PIL.Image.fromarray(image)
        
        # Nếu giữ nguyên tỷ lệ ảnh
        if maintain_aspect_ratio:
            # Tính toán tỷ lệ resize để không làm méo ảnh
            original_width, original_height = image.size
            target_width, target_height = target_size
            
            # Tính tỷ lệ resize để vừa khung
            ratio = min(target_width / original_width, target_height / original_height)
            new_width = int(original_width * ratio)
            new_height = int(original_height * ratio)
            
            # Resize ảnh với chất lượng cao
            resized_image = image.resize((new_width, new_height), PIL.Image.LANCZOS)
            
            # Tạo ảnh nền để đảm bảo kích thước cuối cùng
            background = PIL.Image.new('RGB', target_size, (255, 255, 255))
            offset = ((target_width - new_width) // 2, (target_height - new_height) // 2)
            background.paste(resized_image, offset)
            
            return background
        
        # Resize thẳng nếu không cần giữ tỷ lệ
        return image.resize(target_size, PIL.Image.LANCZOS)
    
    @staticmethod
    def load_image(image_source):
        """
        Tải ảnh từ các nguồn khác nhau
        
        Args:
            image_source (str or file-like): Nguồn ảnh (URL, path, file upload)
        
        Returns:
            PIL.Image: Ảnh đã tải
        """
        try:
            # Nếu là đường dẫn URL
            if isinstance(image_source, str):
                response = httpx.get(image_source, follow_redirects=True)
                image = PIL.Image.open(io.BytesIO(response.content))
            # Nếu là file upload 
            else:
                image = PIL.Image.open(image_source)
            
            return image
        except Exception as e:
            print(f"Lỗi khi tải ảnh: {e}")
            return None
    
    @staticmethod
    def convert_to_rgb(image):
        """
        Chuyển ảnh về chế độ RGB
        
        Args:
            image (PIL.Image): Ảnh đầu vào
        
        Returns:
            PIL.Image: Ảnh ở chế độ RGB
        """
        return image.convert('RGB')
    
    @staticmethod
    def image_to_bytes(image, format='JPEG', quality=85):
        """
        Chuyển ảnh sang dạng bytes với tùy chọn chất lượng
        
        Args:
            image (PIL.Image): Ảnh đầu vào
            format (str): Định dạng ảnh
            quality (int): Chất lượng ảnh (0-95)
        
        Returns:
            bytes: Ảnh dưới dạng bytes
        """
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=format, quality=quality)
        return img_byte_arr.getvalue()

    @staticmethod
    def process_image(image_source, target_size=(224, 224), maintain_aspect_ratio=True):
        """
        Xử lý hoàn chỉnh ảnh: tải, chuyển RGB, resize
        
        Args:
            image_source (str or file-like): Nguồn ảnh
            target_size (tuple): Kích thước mục tiêu
            maintain_aspect_ratio (bool): Giữ nguyên tỷ lệ ảnh
        
        Returns:
            PIL.Image or None: Ảnh đã được xử lý
        """
        try:
            # Tải ảnh
            image = ImageProcessor.load_image(image_source)
            
            if image is None:
                return None
            
            # Chuyển sang chế độ RGB
            image = ImageProcessor.convert_to_rgb(image)
            
            # Resize ảnh
            resized_image = ImageProcessor.resize_image(
                image, 
                target_size=target_size, 
                maintain_aspect_ratio=maintain_aspect_ratio
            )
            
            return resized_image
        
        except Exception as e:
            print(f"Lỗi trong quá trình xử lý ảnh: {e}")
            return None

# Ví dụ sử dụng
def main():
    try:
        # Tải ảnh từ URL
        url = "https://example.com/image.jpg"
        processed_image = ImageProcessor.process_image(url, target_size=(300, 300))
        
        if processed_image:
            # Lưu ảnh đã xử lý
            processed_image.save("processed_image.jpg")
            
            # Chuyển sang bytes nếu cần
            image_bytes = ImageProcessor.image_to_bytes(processed_image)
            
            print("Xử lý ảnh thành công!")
        else:
            print("Không thể xử lý ảnh.")
    
    except Exception as e:
        print(f"Lỗi chung: {e}")

if __name__ == "__main__":
    main()
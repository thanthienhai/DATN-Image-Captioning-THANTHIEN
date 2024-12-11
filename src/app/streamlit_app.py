import streamlit as st
from image_processor import ImageProcessor
from caption_generator import CaptionGenerator

def main():
    # Thiết lập trang
    st.set_page_config(page_title="Image Captioning", page_icon="🖼️")
    
    # Khởi tạo các module
    image_processor = ImageProcessor()
    caption_generator = CaptionGenerator()
    
    # Tiêu đề ứng dụng
    st.title("🖼️ Công cụ mô tả hình ảnh")
    
    # Các mẫu ảnh
    sample_images = {
        "Tòa Nhà Quốc Hội Anh": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/Palace_of_Westminster_from_the_dome_on_Methodist_Central_Hall.jpg/2560px-Palace_of_Westminster_from_the_dome_on_Methodist_Central_Hall.jpg",
        "Tháp Eiffel": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6c/Paris_-_Eiffel_Tower%2C_View_from_Champ_de_Mars%2C_6_October_2022.jpg/1280px-Paris_-_Eiffel_Tower%2C_View_from_Champ_de_Mars%2C_6_October_2022.jpg",
    }
    

    col1, col2 = st.columns(2)
    
    with col1:
        # Chọn nguồn ảnh
        source = st.radio("Chọn nguồn ảnh:", 
                        ["Upload từ máy tính", "Nhập URL"])
    
    # Xử lý từng nguồn ảnh
    if source == "Upload từ máy tính":
        with col2:
            uploaded_file = st.file_uploader("Chọn ảnh", 
                                            type=['jpg', 'jpeg', 'png', 'webp'],
                                            help="Tải ảnh từ máy tính của bạn")
        if uploaded_file:
            # Tải và xử lý ảnh
            image = image_processor.load_image(uploaded_file)
            image_rgb = image_processor.convert_to_rgb(image)
            # image_resized = image_processor.resize_image(image_rgb)
            image_resized = image_rgb
            
            # Hiển thị ảnh
            st.image(image_resized, caption="Ảnh đã tải lên", use_container_width=True)
            
            # Sinh caption
            if st.button("Sinh Caption"):
                with st.spinner('Đang sinh caption...'):
                    # Sinh caption đa ngôn ngữ
                    captions = caption_generator.multi_language_caption(image_resized)
                    
                    # Hiển thị kết quả
                    st.subheader("Captions:")
                    for lang, caption in captions.items():
                        st.markdown(f"**{lang.upper()}:** {caption}")
    
    else:  # Nhập URL
        with col2:
            image_url = st.text_input("Nhập URL ảnh:")
        
        if image_url:
            try:
                # Tải và xử lý ảnh
                image = image_processor.load_image(image_url)
                image_rgb = image_processor.convert_to_rgb(image)
                image_resized = image_processor.resize_image(image_rgb)
                
                # Hiển thị ảnh
                st.image(image_resized, caption="Ảnh từ URL", use_container_width=True)
                
                # Sinh caption
                if st.button("Sinh Caption"):
                    with st.spinner('Đang sinh caption...'):
                        # Sinh caption đa ngôn ngữ
                        captions = caption_generator.multi_language_caption(image_resized)
                        
                        # Hiển thị kết quả
                        st.subheader("Captions:")
                        for lang, caption in captions.items():
                            st.markdown(f"**{lang.upper()}:** {caption}")
            except Exception as e:
                st.error(f"Lỗi khi tải ảnh: {e}")

if __name__ == "__main__":
    main()
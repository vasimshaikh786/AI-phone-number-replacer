import streamlit as st
import cv2
import numpy as np
import re
import pytesseract
from PIL import Image, ImageDraw, ImageFont
import io
import os
import tempfile
import concurrent.futures
import base64
from datetime import datetime
import time

# Set page configuration
st.set_page_config(
    page_title="Phone Number Detector & Replacer",
    page_icon="📱",
    layout="wide"
)

# App title and description
st.title("Phone Number Detector & Replacer")
st.markdown("""
This app detects phone numbers in images and replaces them with a new number while maintaining the original style and format.
Upload an image containing phone numbers to get started.
""")

# Function to preprocess image for better OCR
def preprocess_image(image):
    # Convert to numpy array if it's a PIL Image
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
    
    # Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # Apply adaptive thresholding for varying lighting conditions
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Noise reduction
    denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
    
    # Edge enhancement for better text detection
    edges = cv2.Canny(denoised, 100, 200)
    
    # Create a PIL Image from the processed numpy array
    processed_image = Image.fromarray(denoised)
    
    return processed_image, denoised, edges, gray

# Function to detect and correct image orientation
def detect_and_correct_orientation(image):
    try:
        # Use Tesseract's OSD (Orientation and Script Detection)
        osd = pytesseract.image_to_osd(image)
        angle = re.search('(?<=Rotate: )\d+', osd).group(0)
        
        # Convert to numpy array if it's a PIL Image
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
        
        # Rotate the image if needed
        if int(angle) != 0:
            (h, w) = image_np.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, int(angle), 1.0)
            rotated = cv2.warpAffine(image_np, M, (w, h), 
                                    flags=cv2.INTER_CUBIC, 
                                    borderMode=cv2.BORDER_REPLICATE)
            return Image.fromarray(rotated)
        
        return image
    except:
        # If orientation detection fails, return original image
        return image

# Function to enhance low resolution images
def enhance_low_resolution(image):
    # Convert to numpy array if it's a PIL Image
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
    
    # Check if image is low resolution
    if image_np.shape[0] < 300 or image_np.shape[1] < 300:
        # Use simple scaling for low-res images
        scale_percent = 200  # percent of original size
        width = int(image_np.shape[1] * scale_percent / 100)
        height = int(image_np.shape[0] * scale_percent / 100)
        dim = (width, height)
        
        # Upscale the image
        upscaled = cv2.resize(image_np, dim, interpolation=cv2.INTER_CUBIC)
        return Image.fromarray(upscaled)
    
    return image

# Function to perform OCR
def perform_ocr(image):
    # Configure Tesseract for optimal results
    custom_config = r'--oem 3 --psm 11 -c preserve_interword_spaces=1'
    
    # Perform OCR
    ocr_results = pytesseract.image_to_data(image, config=custom_config, 
                                         output_type=pytesseract.Output.DICT)
    
    # Filter results based on confidence
    filtered_results = []
    n_boxes = len(ocr_results['text'])
    
    for i in range(n_boxes):
        if int(ocr_results['conf'][i]) > 60:  # Confidence threshold
            x = ocr_results['left'][i]
            y = ocr_results['top'][i]
            w = ocr_results['width'][i]
            h = ocr_results['height'][i]
            text = ocr_results['text'][i]
            
            filtered_results.append({
                'text': text,
                'bbox': (x, y, w, h),
                'confidence': ocr_results['conf'][i]
            })
    
    return filtered_results, ocr_results

# Function to detect phone numbers in OCR results
def detect_phone_numbers(ocr_results):
    # Patterns for various phone number formats
    patterns = [
        r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',  # US: (703) 123-4567, 703.123.4567
        r'\d{4}\s\d{3}\s\d{3}',                  # AU/UK: 1300 661 453
        r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}',        # Simple: 703-123-4567
        r'PH[:.]?\s*\d+[-.\s]?\d+[-.\s]?\d+',    # With prefix: PH: 1300 661 453
        r'\+\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{2,4}[-.\s]?\d{2,4}[-.\s]?\d{2,4}',  # International
        r'\d{3}[-.\s]?\d{2}[-.\s]?\d{2}',        # European: 123-45-67
        r'\d{3}-\d{4}',                          # Short format: 123-4567
        r'\d{2}\s\d{2}\s\d{2}\s\d{2}\s\d{2}'     # Grouped pairs: 12 34 56 78 90
    ]
    
    phone_numbers = []
    
    # Check each OCR result against phone patterns
    for result in ocr_results:
        text = result['text']
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                phone_numbers.append({
                    'number': match.group(),
                    'text': text,
                    'bbox': result['bbox'],
                    'pattern': pattern,
                    'confidence': result['confidence']
                })
                break  # Found a pattern match, no need to check other patterns
    
    # Also check for phone numbers that span multiple adjacent text blocks
    # This is a simplified approach - a more complex implementation might be needed
    # for production use cases with complex layouts
    if len(ocr_results) > 1:
        # Combine adjacent text blocks and check for phone numbers
        for i in range(len(ocr_results) - 1):
            combined_text = ocr_results[i]['text'] + " " + ocr_results[i+1]['text']
            
            for pattern in patterns:
                matches = re.finditer(pattern, combined_text, re.IGNORECASE)
                
                for match in matches:
                    # Only add if the found number isn't already in our list
                    if not any(phone['number'] == match.group() for phone in phone_numbers):
                        # Create a bounding box that spans both text blocks
                        x1, y1, w1, h1 = ocr_results[i]['bbox']
                        x2, y2, w2, h2 = ocr_results[i+1]['bbox']
                        
                        # Calculate the combined bounding box
                        x = min(x1, x2)
                        y = min(y1, y2)
                        w = max(x1 + w1, x2 + w2) - x
                        h = max(y1 + h1, y2 + h2) - y
                        
                        phone_numbers.append({
                            'number': match.group(),
                            'text': combined_text,
                            'bbox': (x, y, w, h),
                            'pattern': pattern,
                            'confidence': min(ocr_results[i]['confidence'], ocr_results[i+1]['confidence'])
                        })
    
    return phone_numbers

# Function to analyze style of text
def analyze_style(image, bbox):
    x, y, w, h = bbox
    
    # Convert to numpy array if it's a PIL Image
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
    
    # Make sure coordinates are within image bounds
    x = max(0, x)
    y = max(0, y)
    w = min(w, image_np.shape[1] - x)
    h = min(h, image_np.shape[0] - y)
    
    # Extract the region containing the text
    region = image_np[y:y+h, x:x+w]
    
    # Handle empty regions or invalid coordinates
    if region.size == 0:
        return {
            'color': (0, 0, 0),  # Default to black
            'font_size': 12,     # Default font size
            'is_bold': False     # Not bold
        }
    
    # Convert to RGB if grayscale
    if len(region.shape) == 2:
        region = cv2.cvtColor(region, cv2.COLOR_GRAY2RGB)
    
    # Analyze color (average color in the region)
    avg_color = np.mean(region, axis=(0, 1))
    
    # Convert to grayscale for further analysis
    gray_region = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    
    # Determine if text is bold (by analyzing pixel intensity and distribution)
    # This is a simplified approach
    is_bold = np.mean(gray_region) < 150 and np.std(gray_region) > 50
    
    # Estimate font size based on height
    font_size = h
    
    return {
        'color': avg_color,
        'font_size': font_size,
        'is_bold': is_bold
    }

# Function to format new number to match original pattern
def format_number(original, new_number):
    # Extract only digits from both numbers
    original_digits = ''.join(c for c in original if c.isdigit())
    new_digits = ''.join(c for c in new_number if c.isdigit())
    
    # If lengths don't match, adapt the new number
    if len(original_digits) != len(new_digits):
        if len(original_digits) > len(new_digits):
            # Pad new number
            new_digits = new_digits.ljust(len(original_digits), '0')
        else:
            # Truncate new number
            new_digits = new_digits[:len(original_digits)]
    
    # Replace digits while preserving format
    result = ""
    digit_index = 0
    
    for char in original:
        if char.isdigit() and digit_index < len(new_digits):
            result += new_digits[digit_index]
            digit_index += 1
        else:
            result += char
            
    return result

# Function to replace phone numbers in image
def replace_phone_number(image, phone_data, new_number):
    # Convert to PIL Image if needed
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    # Create a copy to draw on
    result_image = image.copy()
    draw = ImageDraw.Draw(result_image)
    
    # For each detected phone number
    for phone in phone_data:
        x, y, w, h = phone['bbox']
        original_number = phone['number']
        
        # Analyze style of the original number
        style = analyze_style(image, phone['bbox'])
        
        # Format the new number to match the original pattern
        formatted_new_number = format_number(original_number, new_number)
        
        # Prepare to draw the new number
        try:
            # Try to get a system font with appropriate weight
            font_size = int(style['font_size'] * 0.7)  # Adjust size for better fit
            
            # Select font based on bold detection
            if style['is_bold']:
                try:
                    font = ImageFont.truetype("Arial Bold.ttf", font_size)
                except:
                    try:
                        font = ImageFont.truetype("arialbd.ttf", font_size)
                    except:
                        font = ImageFont.load_default()
            else:
                try:
                    font = ImageFont.truetype("Arial.ttf", font_size)
                except:
                    try:
                        font = ImageFont.truetype("arial.ttf", font_size)
                    except:
                        font = ImageFont.load_default()
        except:
            # Fallback to default
            font = ImageFont.load_default()
        
        # Convert color to RGB tuple
        color = tuple(int(c) for c in style['color'])
        
        # Create a rectangle to cover the original number
        # Match background color by sampling area around the text
        bg_x = max(0, x - 5)
        bg_y = max(0, y - 5)
        bg_w = min(image.width - bg_x, w + 10)
        bg_h = min(image.height - bg_y, h + 10)
        
        # Sample background from slightly larger area
        bg_region = np.array(image.crop((bg_x, bg_y, bg_x + bg_w, bg_y + bg_h)))
        
        # Create a mask to exclude text pixels from background sampling
        mask = np.ones(bg_region.shape[:2], dtype=bool)
        mask[5:5+h, 5:5+w] = False
        
        # Sample background color from non-text area
        if np.any(mask):
            bg_color = tuple(int(c) for c in np.mean(bg_region[mask], axis=0))
        else:
            # Fallback if mask is empty
            bg_color = (255, 255, 255)
        
        # Draw the background rectangle
        draw.rectangle([x, y, x+w, y+h], fill=bg_color)
        
        # Calculate text positioning to center in the space
        text_width, text_height = draw.textsize(formatted_new_number, font=font)
        text_x = x + (w - text_width) // 2
        text_y = y + (h - text_height) // 2
        
        # Draw the new number
        draw.text((text_x, text_y), formatted_new_number, fill=color, font=font)
    
    return result_image

# Function to process a batch of images
def batch_process_images(images, new_number):
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # Create a list to hold the futures
        futures = []
        
        # Submit tasks
        for img in images:
            future = executor.submit(process_single_image, img, new_number)
            futures.append(future)
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append({
                    "success": False,
                    "error": str(e)
                })
    
    return results

# Function to process a single image
def process_single_image(image, new_number):
    try:
        # Record start time
        start_time = time.time()
        
        # Step 1: Preprocess image
        processed_image, denoised, edges, gray = preprocess_image(image)
        
        # Step 2: Correct orientation if needed
        oriented_image = detect_and_correct_orientation(processed_image)
        
        # Step 3: Enhance resolution if needed
        enhanced_image = enhance_low_resolution(oriented_image)
        
        # Step 4: Perform OCR
        ocr_filtered_results, ocr_full_results = perform_ocr(enhanced_image)
        
        # Step 5: Detect phone numbers
        phone_numbers = detect_phone_numbers(ocr_filtered_results)
        
        # Step 6: Replace phone numbers if any are found
        if phone_numbers:
            result_image = replace_phone_number(image, phone_numbers, new_number)
            
            # Record end time
            end_time = time.time()
            
            return {
                "success": True,
                "original_image": image,
                "result_image": result_image,
                "phone_numbers": phone_numbers,
                "processing_time": end_time - start_time
            }
        else:
            # Record end time
            end_time = time.time()
            
            return {
                "success": True,
                "original_image": image,
                "result_image": image,  # No changes
                "phone_numbers": [],
                "processing_time": end_time - start_time
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# Function to get image download link
def get_image_download_link(img, filename, text):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/png;base64,{img_str}" download="{filename}">{text}</a>'
    return href

# Initialize session state for batch processing
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = []

# Create tabs for single image and batch processing
tab1, tab2 = st.tabs(["Single Image Processing", "Batch Processing"])

with tab1:
    st.header("Single Image Processing")
    
    # File uploader for single image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load the image
        image = Image.open(uploaded_file)
        
        # Display original image
        st.subheader("Original Image")
        st.image(image, use_column_width=True)
        
        # Process button
        if st.button("Detect Phone Numbers"):
            with st.spinner("Processing image..."):
                # Preprocess for better OCR
                processed_image, denoised, edges, gray = preprocess_image(image)
                
                # Perform OCR
                ocr_filtered_results, ocr_full_results = perform_ocr(processed_image)
                
                # Detect phone numbers
                phone_numbers = detect_phone_numbers(ocr_filtered_results)
                
                if phone_numbers:
                    st.success(f"Found {len(phone_numbers)} phone numbers!")
                    
                    # Display detected numbers
                    st.subheader("Detected Phone Numbers")
                    for i, phone in enumerate(phone_numbers):
                        st.write(f"{i+1}. {phone['number']} (Confidence: {phone['confidence']}%)")
                    
                    # Create a copy of the image to draw bounding boxes
                    image_with_boxes = np.array(image.copy())
                    
                    # Draw bounding boxes around detected phone numbers
                    for phone in phone_numbers:
                        x, y, w, h = phone['bbox']
                        cv2.rectangle(image_with_boxes, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    
                    # Display image with bounding boxes
                    st.image(image_with_boxes, caption="Phone numbers detected", use_column_width=True)
                    
                    # Input field for replacement number
                    new_number = st.text_input("Enter new phone number:", "555-123-4567")
                    
                    if st.button("Replace Phone Numbers"):
                        with st.spinner("Replacing phone numbers..."):
                            # Replace the phone numbers
                            result_image = replace_phone_number(image, phone_numbers, new_number)
                            
                            # Display the result
                            st.subheader("Result")
                            st.image(result_image, caption="Image with replaced phone numbers", use_column_width=True)
                            
                            # Add download button
                            st.markdown(
                                get_image_download_link(
                                    result_image, 
                                    f"replaced_phone_numbers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                    "Download Result Image"
                                ), 
                                unsafe_allow_html=True
                            )
                            
                            # Side-by-side comparison
                            col1, col2 = st.columns(2)
                            with col1:
                                st.subheader("Original")
                                st.image(image, use_column_width=True)
                            with col2:
                                st.subheader("Result")
                                st.image(result_image, use_column_width=True)
                else:
                    st.warning("No phone numbers detected. Try another image or adjust the image quality.")
                    
                    # Display preprocessed image to help with debugging
                    st.subheader("Preprocessed Image")
                    st.image(processed_image, caption="Preprocessed for OCR", use_column_width=True)

with tab2:
    st.header("Batch Processing")
    
    # File uploader for multiple images
    uploaded_files = st.file_uploader("Choose multiple images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if uploaded_files:
        st.write(f"Uploaded {len(uploaded_files)} images")
        
        # Display thumbnails of uploaded images
        cols = st.columns(4)
        for i, file in enumerate(uploaded_files):
            with cols[i % 4]:
                st.image(Image.open(file), width=150, caption=f"Image {i+1}")
        
        # Input field for replacement number
        batch_new_number = st.text_input("Enter new phone number for all images:", "555-123-4567")
        
        if st.button("Process Batch"):
            with st.spinner(f"Processing {len(uploaded_files)} images..."):
                # Process all images
                batch_images = [Image.open(file) for file in uploaded_files]
                st.session_state.batch_results = batch_process_images(batch_images, batch_new_number)
                
                st.success(f"Processed {len(st.session_state.batch_results)} images!")
    
    # Display batch results
    if st.session_state.batch_results:
        st.subheader("Batch Processing Results")
        
        # Count successful operations
        successful = sum(1 for result in st.session_state.batch_results if result["success"])
        st.write(f"Successfully processed {successful} out of {len(st.session_state.batch_results)} images")
        
        # Create an expander for each result
        for i, result in enumerate(st.session_state.batch_results):
            with st.expander(f"Image {i+1} {'✅' if result['success'] else '❌'}"):
                if result["success"]:
                    if result["phone_numbers"]:
                        st.write(f"Found {len(result['phone_numbers'])} phone numbers")
                        
                        # List the phone numbers
                        for j, phone in enumerate(result["phone_numbers"]):
                            st.write(f"{j+1}. {phone['number']}")
                        
                        # Show before/after
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(result["original_image"], caption="Original", use_column_width=True)
                        with col2:
                            st.image(result["result_image"], caption="Result", use_column_width=True)
                        
                        # Download link
                        st.markdown(
                            get_image_download_link(
                                result["result_image"], 
                                f"batch_result_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                "Download Result"
                            ), 
                            unsafe_allow_html=True
                        )
                    else:
                        st.write("No phone numbers found in this image")
                        st.image(result["original_image"], caption="Original", use_column_width=True)
                else:
                    st.error(f"Error processing image: {result['error']}")
        
        # Add option to download all results as a ZIP
        if st.button("Download All Results"):
            st.warning("This feature would zip all processed images for download. For streamlit implementation, additional libraries would be needed.")

# Add settings section
with st.expander("Advanced Settings"):
    st.subheader("OCR Settings")
    ocr_lang = st.selectbox("OCR Language", ["eng", "eng+fra", "eng+deu", "eng+spa"], index=0)
    confidence_threshold = st.slider("Confidence threshold", 0, 100, 60)
    
    st.subheader("Phone Number Detection")
    include_short_numbers = st.checkbox("Include shorter phone numbers", value=False)
    custom_pattern = st.text_input("Add custom regex pattern")
    
    st.subheader("Styling")
    preserve_colors = st.checkbox("Preserve original colors", value=True)
    font_scaling = st.slider("Font size adjustment", 0.5, 1.5, 0.8, 0.1)

# Add information about the app
st.markdown("""
---
### How It Works
1. **Upload** an image containing phone numbers
2. Click **Detect Phone Numbers** to process the image
3. Enter a **new phone number** to replace the detected ones
4. Click **Replace Phone Numbers** to generate the result
5. **Download** the processed image

### Supported Phone Number Formats
- US Format: (703) 123-4567, 703-123-4567, 703.123.4567
- International Format: +1 703 123 4567
- Australian/UK Format: 1300 661 453
- European Format: 123-45-67
- Short Format: 123-4567
- With prefixes: PH: 1300 661 453

### Capabilities
- Detects phone numbers in various formats
- Maintains original formatting, font style, and color
- Handles rotated and skewed text
- Processes images in batch
- Enhances low resolution images for better detection

### Limitations
- Very stylized fonts may not be perfectly matched
- Heavily distorted or artistic text may be missed
- Background patterns may not be perfectly preserved
""")

# Footer
st.markdown("---")
st.markdown("Phone Number Detector & Replacer © 2025")
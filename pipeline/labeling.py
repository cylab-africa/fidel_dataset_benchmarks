import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import csv

def detect_text_lines(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Failed to load image: {image_path}")
        return None, None, None, None
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to get binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Calculate horizontal projection profile
    horizontal_profile = np.sum(binary, axis=1)
    
    # Smooth the profile to reduce noise
    kernel_size = 10
    kernel = np.ones(kernel_size) / kernel_size
    smoothed_profile = np.convolve(horizontal_profile, kernel, mode='same')
    
    # Find valleys in the profile (line separators)
    avg_profile = np.mean(smoothed_profile)
    threshold = avg_profile * 0.1
    
    valleys = []
    in_valley = False
    valley_start = 0
    min_valley_width = 5
    
    for i in range(len(smoothed_profile)):
        if smoothed_profile[i] < threshold and not in_valley:
            in_valley = True
            valley_start = i
        elif (smoothed_profile[i] >= threshold or i == len(smoothed_profile)-1) and in_valley:
            in_valley = False
            valley_end = i
            valley_width = valley_end - valley_start
            
            # Only consider valleys that are wide enough to be line separators
            if valley_width >= min_valley_width:
                valleys.append((valley_start, valley_end))
    
    # Calculate average content height between valleys
    content_heights = []
    
    for i in range(len(valleys) - 1):
        content_start = valleys[i][1]
        content_end = valleys[i+1][0]
        content_height = content_end - content_start
        if content_height > 10:  # Only consider reasonable content heights
            content_heights.append(content_height)
    
    if content_heights:
        avg_content_height = sum(content_heights) / len(content_heights)
        print(f"Average content height: {avg_content_height:.2f} pixels")
    else:
        avg_content_height = 30
    
    # More conservative merging strategy
    merged_valleys = []
    
    if valleys:
        # Start with the first valley
        current_valley = valleys[0]
        
        for i in range(1, len(valleys)):
            next_valley = valleys[i]
            gap_size = next_valley[0] - current_valley[1]
            
            # Calculate region characteristics for the gap
            region_start = current_valley[1]
            region_end = next_valley[0]
            region_height = region_end - region_start
            
            # For a gap to be considered for merging:
            # 1. It must be significantly smaller than average content height
            # 2. The upper part must be mostly white
            should_merge = False
            
            if region_height < avg_content_height * 0.5:
                # Extract the upper region to check content
                upper_region_height = min(15, region_height)
                if upper_region_height > 0:
                    upper_region = binary[region_start:region_start + upper_region_height, :]
                    
                    # Calculate black pixel density in the upper region
                    if upper_region.size > 0:
                        black_pixel_density = np.sum(upper_region > 0) / upper_region.size
                    else:
                        black_pixel_density = 0
                    
                    print(f"Gap {i}: size={region_height}, upper black density={black_pixel_density:.4f}")
                    
                    # Only merge if the upper region is mostly white AND the gap is small
                    if black_pixel_density < 0.04:
                        should_merge = True
                        print(f"Merging valleys {i-1} and {i}")
            
            if should_merge:
                # Merge with next valley
                current_valley = (current_valley[0], next_valley[1])
            else:
                # Add current valley to merged list and move to next
                merged_valleys.append(current_valley)
                current_valley = next_valley
        
        # Add the last valley
        merged_valleys.append(current_valley)
    
    # Draw lines on the original image
    result = image.copy()
    for valley_start, valley_end in merged_valleys:
        y = (valley_start + valley_end) // 2
        cv2.line(result, (0, y), (result.shape[1], y), (0, 0, 255), 2)
    
    return result, merged_valleys, binary

def extract_text_lines(image_path, valleys):
    image = cv2.imread(image_path)
    lines = []
    
    # Add image top as the first boundary
    boundaries = [0]
    
    # Add middle points between valleys
    for valley_start, valley_end in valleys:
        middle = (valley_start + valley_end) // 2
        boundaries.append(middle)
    
    # Add image bottom as the last boundary
    boundaries.append(image.shape[0])
    
    # Extract each line using boundaries
    for i in range(len(boundaries) - 1):
        line_start = boundaries[i]
        line_end = boundaries[i + 1]
        line_region = image[line_start:line_end, :]
        lines.append(line_region)
    
    return lines

def save_text_lines(lines, output_folder, base_filename, start_index, num_lines_to_extract, line_texts=None):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    csv_data = []
    
    # Save specific lines as images
    end_index = min(start_index + num_lines_to_extract, len(lines))
    
    for i in range(start_index, end_index):
        if i < len(lines):  # Safety check
            line_index = i - start_index  # Index relative to the starting point
            image_filename = f'{base_filename}_line_{line_index+1}.png'
            output_path = os.path.join(output_folder, image_filename)
            
            cv2.imwrite(output_path, lines[i])
            
            # Create CSV entry
            line_text = line_texts[line_index] if line_texts and line_index < len(line_texts) else ""
            csv_data.append([image_filename, line_text])
            
            print(f'Saved line {line_index+1} to {output_path}')
    
    return csv_data

def process_document_batch(images_folder, typed_txts_folder, handwritten_txts_folder, output_folder='extracted_lines', csv_file='line_mapping.csv'):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Prepare CSV file
    csv_data = [["image_filename", "line_text"]]  # Header
    
    # Create a list to track problematic files
    problematic_files = []
    
    # Process each image in the folder
    for filename in os.listdir(images_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
            base_name = os.path.splitext(filename)[0]
            image_path = os.path.join(images_folder, filename)
            typed_txt_path = os.path.join(typed_txts_folder, f"{base_name}.txt")
            handwritten_txt_path = os.path.join(handwritten_txts_folder, f"{base_name}.txt")
            
            # Check if corresponding typed text file exists
            if not os.path.exists(typed_txt_path):
                print(f"Warning: No matching typed text file found for {filename}, skipping...")
                continue
                
            # Check if corresponding handwritten text file exists
            if not os.path.exists(handwritten_txt_path):
                print(f"Warning: No matching handwritten text file found for {filename}, skipping...")
                continue
            
            print(f"Processing {filename}...")
            
            # Read the typed text file and count lines
            with open(typed_txt_path, 'r', encoding='utf-8') as f:
                typed_lines = f.readlines()
                num_typed_lines = len(typed_lines)
            
            # Read the handwritten text file
            with open(handwritten_txt_path, 'r', encoding='utf-8') as f:
                handwritten_lines = f.readlines()
                num_handwritten_lines = len(handwritten_lines)
            
            print(f"Document contains {num_typed_lines} typed lines and {num_handwritten_lines} handwritten lines")
            
            # Detect and extract text lines from the image
            _, valleys, _ = detect_text_lines(image_path)
            
            # Skip if valleys couldn't be detected
            if not valleys:
                print(f"Warning: Could not detect line valleys in {filename}, skipping...")
                problematic_files.append(filename)
                continue
                
            lines = extract_text_lines(image_path, valleys)
            
            # Calculate the start index: skip first 3 lines + number of typed lines
            skip_lines = 4 + num_typed_lines
            
            # Check if there are enough lines extracted to match the handwritten lines
            if skip_lines + num_handwritten_lines > len(lines):
                print(f"Warning: Not enough lines detected in {filename}. Expected at least {skip_lines + num_handwritten_lines}, but got {len(lines)}. Skipping...")
                problematic_files.append(filename)
                continue
            
            # Save only the required handwritten lines
            batch_csv_data = save_text_lines(
                lines, 
                output_folder, 
                base_name, 
                skip_lines, 
                num_handwritten_lines,  # Extract only as many lines as in the handwritten text file
                handwritten_lines
            )
            
            # Add to CSV data
            csv_data.extend(batch_csv_data)
    
    # Write to CSV file
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)
    
    print(f"CSV file created: {csv_file}")
    
    # Write problematic files to a text file
    if problematic_files:
        problem_file_path = "problematic_files.txt"
        with open(problem_file_path, 'w', encoding='utf-8') as f:
            for filename in problematic_files:
                f.write(f"{filename}\n")
        print(f"List of problematic files saved to {problem_file_path}")

# Example usage
if __name__ == "__main__":
    images_folder = "images labeler 6"  # Folder containing scanned images
    typed_txts_folder = "../txt_forms"  # Folder containing typed text files
    handwritten_txts_folder = "/home/tunga/Downloads/All in one corrected-20250419T183314Z-001/All in one corrected"  # Folder containing handwritten text files
    output_folder = "extracted_handwritten_lines"  # Folder to save extracted handwritten lines
    csv_file = "handwritten_line_mapping_4.csv"  # CSV file to map images to text
    
    # Process all documents in the folder
    process_document_batch(images_folder, typed_txts_folder, handwritten_txts_folder, output_folder, csv_file)
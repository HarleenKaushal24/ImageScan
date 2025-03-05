# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 11:58:01 2025

@author: Harleen
"""

# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import cv2
import numpy as np
import os
from PIL import Image
from matplotlib import pyplot as plt

# Constants
EXCEL_FILE = "img1.xlsm"
IMAGE_FOLDER = "extracted_images"
MATCH_RESULTS_PATH = "match_results.xlsx"

def check_files():
    """Check if required files and folders exist."""
    errors = []
    if not os.path.exists(EXCEL_FILE):
        errors.append(f"Excel file '{EXCEL_FILE}' not found!")
    if not os.path.exists(IMAGE_FOLDER) or not os.listdir(IMAGE_FOLDER):
        errors.append("Image folder is missing or empty!")
    return errors

def main():
    """Main Streamlit App."""
    st.title("Image Recognition and Matching")
    
    # Check required files
    errors = check_files()
    if errors:
        for error in errors:
            st.error(error)
        return
    
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        try:
            target_img = np.array(Image.open(uploaded_file))
            target_img = cv2.cvtColor(target_img, cv2.COLOR_RGB2BGR)
            st.image(target_img, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Find Best Matches"):
                find_best_matches(target_img)
        except Exception as e:
            st.error(f"Error processing uploaded image: {e}")

def find_best_matches(target_img):
    """Find and display best matching images."""
    df = pd.read_excel(EXCEL_FILE, engine='openpyxl')

    # Image preprocessing
    try:
        target_img = remove_background(target_img)
        target_img_gray = cv2.cvtColor(cv2.resize(target_img, (300, 300)), cv2.COLOR_BGR2GRAY)
        
        # Ensure SIFT is available
        try:
            sift = cv2.SIFT_create()
        except AttributeError:
            st.error("SIFT feature detection is unavailable. OpenCV might be outdated.")
            return

        target_kp, target_des = sift.detectAndCompute(target_img_gray, None)
        if target_des is None:
            st.error("No features detected in the uploaded image.")
            return

        # Process images
        match_results = process_images(target_kp, target_des, sift, df)
        display_results(match_results, target_img_gray, target_kp, df)

    except Exception as e:
        st.error(f"Unexpected error: {e}")

def process_images(target_kp, target_des, sift, df):
    """Compare target image with dataset and return top matches."""
    match_scores = {}

    for img_file in os.listdir(IMAGE_FOLDER):
        img_path = os.path.join(IMAGE_FOLDER, img_file)
        img_color = cv2.imread(img_path)

        if img_color is None:
            st.warning(f"Could not read {img_file}, skipping.")
            continue

        img_gray = cv2.cvtColor(remove_background(img_color), cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(img_gray, None)
        if des is None:
            continue

        # Feature matching
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = sorted(bf.match(target_des, des), key=lambda x: x.distance)

        match_scores[img_file] = (len(matches), img_gray, img_color, kp, matches)

    return sorted(match_scores.items(), key=lambda x: x[1][0], reverse=True)[:2]

def display_results(top_matches, target_img_gray, target_kp, df):
    """Display top matching images and save results."""
    if not top_matches:
        st.write("No matches found.")
        return

    st.write("### Top 2 Best Matches")
    match_results = []

    for rank, (img_name, (match_count, img_gray, img_color, kp, matches)) in enumerate(top_matches, start=1):
        st.write(f"**Rank {rank}: {img_name} ({match_count} good matches)**")

        matched_row = extract_row_number(img_name, len(df))
        if matched_row is not None:
            matched_data = df.iloc[matched_row]
            st.write(matched_data)

            match_results.append({
                "Rank": rank,
                "Image Name": img_name,
                "Good Matches": match_count,
                "Excel Row": matched_row,
                **matched_data.to_dict()
            })

        # Display matched images
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(cv2.drawKeypoints(target_img_gray, target_kp, None, color=(0, 255, 0)), cmap='gray')
        ax[0].set_title("Target Image")
        ax[1].imshow(cv2.drawKeypoints(img_gray, kp, None, color=(255, 0, 0)), cmap='gray')
        ax[1].set_title(f"Matched Image {rank}")
        st.pyplot(fig)

    # Save results
    pd.DataFrame(match_results).to_excel(MATCH_RESULTS_PATH, index=False)
    st.success("Match results saved to match_results.xlsx")

def remove_background(img):
    """Remove white background."""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            img = img[y:y+h, x:x+w]
        return img
    except Exception:
        return img

def extract_row_number(filename, df_length):
    """Extract row number safely."""
    try:
        parts = filename.split("_")
        if len(parts) > 1:
            row_number = int(parts[1])
            if 0 <= row_number < df_length:
                return row_number
    except ValueError:
        pass
    return None

if __name__ == "__main__":
    main()

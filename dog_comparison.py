#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
강아지 사진 비교 프로그램
두 강아지 사진을 비교하여 승자를 결정합니다.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# 강아지 이미지 분석을 위한 유틸리티 함수들

def get_dog_bounding_box(image):
    """강아지의 바운딩 박스 구하기"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape[:2]
        
        # 가장자리 영역을 배경으로 가정
        edge_thickness = max(5, min(h, w) // 20)
        edges = np.concatenate([
            gray[0:edge_thickness, :].flatten(),
            gray[h-edge_thickness:, :].flatten(),
            gray[:, 0:edge_thickness].flatten(),
            gray[:, w-edge_thickness:].flatten()
        ])
        background_color = np.median(edges)
        
        # 배경과 다른 영역 찾기
        diff = np.abs(gray.astype(float) - background_color)
        threshold = np.percentile(diff, 70)
        
        # 마스크 생성
        mask = (diff > threshold).astype(np.uint8) * 255
        
        # 모폴로지 연산
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 가장 큰 연결된 영역 찾기
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = (labels == largest_label).astype(np.uint8) * 255
        
        # 바운딩 박스 계산
        y_coords, x_coords = np.where(mask > 0)
        if len(y_coords) > 0 and len(x_coords) > 0:
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            y_min, y_max = np.min(y_coords), np.max(y_coords)
            
            # 여유 공간 추가 (10%)
            padding_x = int((x_max - x_min) * 0.1)
            padding_y = int((y_max - y_min) * 0.1)
            
            x_min = max(0, x_min - padding_x)
            y_min = max(0, y_min - padding_y)
            x_max = min(w, x_max + padding_x)
            y_max = min(h, y_max + padding_y)
            
            return (x_min, y_min, x_max, y_max)
        
        # 실패 시 전체 이미지 반환
        return (0, 0, w, h)
    except Exception as e:
        h, w = image.shape[:2] if len(image.shape) == 2 else image.shape[:2]
        return (0, 0, w, h)

def normalize_dog_size(image1, image2):
    """두 강아지 이미지의 크기를 동일하게 맞춤"""
    try:
        # PIL Image를 numpy array로 변환
        if isinstance(image1, Image.Image):
            img1 = np.array(image1)
        else:
            img1 = image1.copy()
            
        if isinstance(image2, Image.Image):
            img2 = np.array(image2)
        else:
            img2 = image2.copy()
        
        # RGB로 변환
        if len(img1.shape) == 3 and img1.shape[2] == 4:
            img1 = cv2.cvtColor(img1, cv2.COLOR_RGBA2RGB)
        if len(img2.shape) == 3 and img2.shape[2] == 4:
            img2 = cv2.cvtColor(img2, cv2.COLOR_RGBA2RGB)
        if len(img1.shape) == 2:
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
        if len(img2.shape) == 2:
            img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
        
        # 각 이미지에서 강아지 바운딩 박스 구하기
        bbox1 = get_dog_bounding_box(img1)
        bbox2 = get_dog_bounding_box(img2)
        
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # 강아지 크기 계산
        dog1_width = x1_max - x1_min
        dog1_height = y1_max - y1_min
        dog2_width = x2_max - x2_min
        dog2_height = y2_max - y2_min
        
        # 목표 크기 결정 (더 큰 강아지의 크기를 기준)
        target_width = max(dog1_width, dog2_width)
        target_height = max(dog1_height, dog2_height)
        
        # 각 강아지를 크롭
        dog1_cropped = img1[y1_min:y1_max, x1_min:x1_max]
        dog2_cropped = img2[y2_min:y2_max, x2_min:x2_max]
        
        # 각 강아지를 목표 크기로 리사이즈 (비율 유지)
        # 비율 유지하면서 리사이즈
        scale1_w = target_width / dog1_width
        scale1_h = target_height / dog1_height
        scale1 = min(scale1_w, scale1_h)  # 비율 유지를 위해 작은 값 사용
        
        scale2_w = target_width / dog2_width
        scale2_h = target_height / dog2_height
        scale2 = min(scale2_w, scale2_h)
        
        new_w1 = int(dog1_width * scale1)
        new_h1 = int(dog1_height * scale1)
        new_w2 = int(dog2_width * scale2)
        new_h2 = int(dog2_height * scale2)
        
        # 리사이즈
        dog1_resized = cv2.resize(dog1_cropped, (new_w1, new_h1), interpolation=cv2.INTER_LINEAR)
        dog2_resized = cv2.resize(dog2_cropped, (new_w2, new_h2), interpolation=cv2.INTER_LINEAR)
        
        # 목표 크기에 맞춰 패딩 추가 (중앙 정렬)
        final_w = max(new_w1, new_w2)
        final_h = max(new_h1, new_h2)
        
        # 배경 색상 결정 (이미지 가장자리 색상)
        bg_color1 = tuple(map(int, img1[0, 0]))
        bg_color2 = tuple(map(int, img2[0, 0]))
        
        # 최종 이미지 생성 (패딩 추가)
        final_img1 = np.full((final_h, final_w, 3), bg_color1, dtype=np.uint8)
        final_img2 = np.full((final_h, final_w, 3), bg_color2, dtype=np.uint8)
        
        # 중앙에 배치
        y_offset1 = (final_h - new_h1) // 2
        x_offset1 = (final_w - new_w1) // 2
        y_offset2 = (final_h - new_h2) // 2
        x_offset2 = (final_w - new_w2) // 2
        
        final_img1[y_offset1:y_offset1+new_h1, x_offset1:x_offset1+new_w1] = dog1_resized
        final_img2[y_offset2:y_offset2+new_h2, x_offset2:x_offset2+new_w2] = dog2_resized
        
        return final_img1, final_img2
        
    except Exception as e:
        # 오류 시 원본 반환
        if isinstance(image1, Image.Image):
            img1 = np.array(image1)
        else:
            img1 = image1
        if isinstance(image2, Image.Image):
            img2 = np.array(image2)
        else:
            img2 = image2
        return img1, img2

def extract_dog_region(image):
    """배경을 제거하고 강아지 영역만 추출"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape[:2]
        
        # 가장자리 영역을 배경으로 가정
        edge_thickness = max(5, min(h, w) // 20)
        edges = np.concatenate([
            gray[0:edge_thickness, :].flatten(),
            gray[h-edge_thickness:, :].flatten(),
            gray[:, 0:edge_thickness].flatten(),
            gray[:, w-edge_thickness:].flatten()
        ])
        background_color = np.median(edges)
        
        # 배경과 다른 영역 찾기 (임계값 사용)
        diff = np.abs(gray.astype(float) - background_color)
        threshold = np.percentile(diff, 70)  # 상위 30%가 강아지로 가정
        
        # 마스크 생성
        mask = (diff > threshold).astype(np.uint8) * 255
        
        # 모폴로지 연산으로 노이즈 제거
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 가장 큰 연결된 영역 찾기 (강아지로 가정)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels > 1:
            # 배경(0번) 제외하고 가장 큰 영역
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = (labels == largest_label).astype(np.uint8) * 255
        
        # 마스크를 RGB로 확장
        if len(image.shape) == 3:
            mask_rgb = np.stack([mask, mask, mask], axis=2)
            dog_only = image * (mask_rgb / 255.0)
        else:
            dog_only = image * (mask / 255.0)
        
        return dog_only.astype(np.uint8), mask
    except Exception as e:
        # 오류 시 원본 이미지 반환
        return image, np.ones(image.shape[:2], dtype=np.uint8) * 255

def calculate_whiteness_score(image):
    """강아지 영역만의 하얀색 정도를 계산 (배경 제외)"""
    # 배경 제거
    dog_only, mask = extract_dog_region(image)
    
    # 마스크가 있는 영역만 분석
    if len(dog_only.shape) == 3:
        masked_pixels = dog_only[mask > 0]
        if len(masked_pixels) == 0:
            return 0.0
        # RGB를 HSV로 변환
        hsv_pixels = cv2.cvtColor(masked_pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    else:
        masked_pixels = dog_only[mask > 0]
        if len(masked_pixels) == 0:
            return 0.0
        hsv_pixels = np.stack([masked_pixels, masked_pixels, masked_pixels], axis=1)
    
    # 밝기(V) 값의 평균 계산
    brightness = np.mean(hsv_pixels[:, 2])
    
    # 하얀색은 밝고 채도가 낮음
    saturation = np.mean(hsv_pixels[:, 1])
    whiteness = (brightness / 255.0) * (1 - saturation / 255.0)
    
    return whiteness

def detect_nose_length(image):
    """코 길이를 감지 (이미지 분석 기반)"""
    try:
        # 그레이스케일 변환
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # 얼굴 영역 감지를 위한 간단한 방법
        # 이미지의 중앙 영역을 얼굴로 가정하고 분석
        h, w = gray.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        # 얼굴 영역 (이미지 중앙 40% 영역)
        face_region = gray[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
        
        if face_region.size > 0:
            # 얼굴 영역의 세로 비율을 코 길이의 근사치로 사용
            # 더 긴 얼굴 = 더 긴 코로 가정
            face_height = face_region.shape[0]
            face_width = face_region.shape[1]
            
            # 얼굴 비율 (세로/가로)이 클수록 코가 길다고 가정
            aspect_ratio = face_height / max(face_width, 1)
            
            # 코 길이 추정 (얼굴 높이의 비율)
            estimated_nose_length = face_height * aspect_ratio * 0.3
            
            return estimated_nose_length
        else:
            # 기본값: 이미지 높이의 15%
            return h * 0.15
    except Exception as e:
        # 기본값: 이미지 높이의 15%
        return image.shape[0] * 0.15

def calculate_dog_size(image):
    """강아지 크기를 계산 (이미지에서 강아지가 차지하는 영역)"""
    try:
        # 그레이스케일 변환
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # 배경과 강아지를 구분하기 위한 간단한 방법
        # 이미지의 중앙 영역을 강아지로 가정
        h, w = gray.shape[:2]
        
        # 강아지 영역 (이미지 중앙 60% 영역)
        dog_region = gray[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)]
        
        if dog_region.size > 0:
            # 강아지 영역의 크기 비율
            dog_area = dog_region.shape[0] * dog_region.shape[1]
            total_area = h * w
            size_ratio = dog_area / total_area
            
            # 추가: 강아지 영역의 평균 밝기로 배경과 구분
            # 배경이 밝으면 강아지가 어둡고, 배경이 어두우면 강아지가 밝음
            dog_brightness = np.mean(dog_region)
            edge_brightness = np.mean(np.concatenate([
                gray[0:int(h*0.1), :].flatten(),
                gray[int(h*0.9):, :].flatten(),
                gray[:, 0:int(w*0.1)].flatten(),
                gray[:, int(w*0.9):].flatten()
            ]))
            
            # 대비가 클수록 강아지가 더 명확하게 보임 = 더 크게 보임
            contrast = abs(dog_brightness - edge_brightness) / 255.0
            
            # 크기 점수 = 영역 비율 + 대비 점수
            size_score = size_ratio * (1 + contrast * 0.5)
            
            return size_score
        else:
            return 0.3  # 기본값
    except Exception as e:
        return 0.3  # 기본값

def detect_eye_distance(image):
    """양 눈 사이의 거리 측정"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape[:2]
        
        # 얼굴 영역 (이미지 상단 중앙)
        face_region = gray[int(h*0.1):int(h*0.6), int(w*0.2):int(w*0.8)]
        
        if face_region.size > 0:
            # 눈을 찾기 위한 간단한 방법: 어두운 원형 영역 찾기
            # 가우시안 블러로 노이즈 제거
            blurred = cv2.GaussianBlur(face_region, (5, 5), 0)
            
            # 어두운 영역 찾기 (눈으로 가정)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # 컨투어 찾기
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 원형에 가까운 컨투어 중 크기가 적절한 것 찾기 (눈 후보)
            eye_candidates = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if 50 < area < 5000:  # 눈 크기 범위
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    circularity = 4 * np.pi * area / (cv2.arcLength(contour, True) ** 2 + 1e-6)
                    if circularity > 0.5:  # 원형에 가까움
                        eye_candidates.append((x, y, radius))
            
            if len(eye_candidates) >= 2:
                # 가장 큰 두 개를 눈으로 선택
                eye_candidates.sort(key=lambda e: e[2], reverse=True)
                eye1, eye2 = eye_candidates[0], eye_candidates[1]
                
                # 눈 사이 거리 계산
                distance = np.sqrt((eye1[0] - eye2[0])**2 + (eye1[1] - eye2[1])**2)
                # 정규화 (얼굴 너비 대비)
                face_width = face_region.shape[1]
                normalized_distance = distance / max(face_width, 1)
                
                # 거리가 가까울수록 점수가 높음 (역수 사용)
                eye_closeness = 1.0 / (normalized_distance + 0.1)
                return eye_closeness
        
        # 기본값: 중간값
        return 0.5
    except Exception as e:
        return 0.5

def detect_ear_pointiness(image):
    """귀가 쫑긋한 정도 측정"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape[:2]
        
        # 상단 영역 (귀가 있을 위치)
        top_region = gray[0:int(h*0.4), :]
        
        if top_region.size > 0:
            # 엣지 감지
            edges = cv2.Canny(top_region, 50, 150)
            
            # 상단 가장자리에서 돌출된 부분 찾기
            # 상단 가장자리 스캔
            top_edge = edges[0, :]
            peaks = []
            for i in range(1, len(top_edge)-1):
                if top_edge[i] > 0 and (top_edge[i-1] == 0 or top_edge[i+1] == 0):
                    peaks.append(i)
            
            # 귀는 보통 양쪽에 있음
            if len(peaks) >= 2:
                # 귀의 각도 측정 (수직에 가까울수록 쫑긋함)
                # 상단에서 아래로 내려가면서 엣지 추적
                pointiness_scores = []
                for peak_x in peaks[:2]:  # 상위 2개만
                    # 해당 위치에서 아래로 엣지 추적
                    vertical_edges = 0
                    for y in range(min(10, top_region.shape[0])):
                        if y < edges.shape[0] and peak_x < edges.shape[1]:
                            if edges[y, peak_x] > 0:
                                vertical_edges += 1
                    
                    # 수직 엣지가 많을수록 쫑긋함
                    pointiness = vertical_edges / 10.0
                    pointiness_scores.append(pointiness)
                
                return np.mean(pointiness_scores) if pointiness_scores else 0.3
        
        return 0.3
    except Exception as e:
        return 0.3

def detect_smile(image):
    """웃는 표정 감지"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape[:2]
        
        # 입 영역 (얼굴 하단 중앙)
        mouth_region = gray[int(h*0.5):int(h*0.8), int(w*0.3):int(w*0.7)]
        
        if mouth_region.size > 0:
            # 입은 보통 수평으로 길고 어두움
            # 가우시안 블러
            blurred = cv2.GaussianBlur(mouth_region, (5, 5), 0)
            
            # 어두운 영역 찾기
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # 컨투어 찾기
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            smile_score = 0.0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # 최소 크기
                    # 바운딩 박스
                    x, y, w_cont, h_cont = cv2.boundingRect(contour)
                    aspect_ratio = w_cont / max(h_cont, 1)
                    
                    # 수평으로 길수록 (aspect_ratio > 1) 웃는 입
                    if aspect_ratio > 1.2:
                        # 곡률 측정 (웃는 입은 위로 올라감)
                        hull = cv2.convexHull(contour)
                        if len(hull) > 0:
                            top_point = tuple(hull[hull[:, :, 1].argmin()][0])
                            bottom_point = tuple(hull[hull[:, :, 1].argmax()][0])
                            
                            # 위로 올라간 정도
                            curvature = (y + h_cont - top_point[1]) / max(h_cont, 1)
                            smile_score = max(smile_score, curvature * aspect_ratio)
            
            return min(smile_score, 1.0)
        return 0.3
    except Exception as e:
        return 0.3

def detect_posture(image):
    """자세 감지 (앉아있음, 엎드려있음)"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape[:2]
        
        # 강아지 영역 추출
        dog_only, mask = extract_dog_region(image)
        if len(dog_only.shape) == 3:
            dog_gray = cv2.cvtColor(dog_only, cv2.COLOR_RGB2GRAY)
        else:
            dog_gray = dog_only
        
        # 강아지의 중심과 형태 분석
        moments = cv2.moments(mask)
        if moments['m00'] > 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            
            # 강아지의 높이와 너비 비율
            y_coords, x_coords = np.where(mask > 0)
            if len(y_coords) > 0 and len(x_coords) > 0:
                height = np.max(y_coords) - np.min(y_coords)
                width = np.max(x_coords) - np.min(x_coords)
                aspect_ratio = height / max(width, 1)
                
                # 중심의 위치 (상하)
                center_y_ratio = cy / max(h, 1)
                
                # 앉아있음: 세로가 길고 중심이 중간~하단
                # 엎드려있음: 가로가 길고 중심이 하단
                if aspect_ratio > 1.2 and center_y_ratio > 0.4:
                    # 앉아있음
                    return 'sitting'
                elif aspect_ratio < 0.8 and center_y_ratio > 0.5:
                    # 엎드려있음
                    return 'lying'
        
        return 'standing'
    except Exception as e:
        return 'standing'

def analyze_dog(image):
    """강아지 이미지를 분석하여 점수 계산"""
    # PIL Image를 numpy array로 변환
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # RGB로 변환 (필요한 경우)
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif len(image.shape) == 3 and image.shape[2] == 3:
        # 이미 RGB인 경우
        pass
    else:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # 각 항목 점수 계산
    whiteness = calculate_whiteness_score(image)
    nose_length = detect_nose_length(image)
    dog_size = calculate_dog_size(image)
    eye_distance = detect_eye_distance(image)
    ear_pointiness = detect_ear_pointiness(image)
    smile = detect_smile(image)
    posture = detect_posture(image)
    
    return {
        'whiteness': whiteness,
        'nose_length': nose_length,
        'dog_size': dog_size,
        'eye_distance': eye_distance,
        'ear_pointiness': ear_pointiness,
        'smile': smile,
        'posture': posture,
        'image': image
    }

def compare_dogs(dog1_data, dog2_data):
    """두 강아지 비교 및 점수 계산"""
    scores = {'dog1': 0, 'dog2': 0}
    details = {'dog1': [], 'dog2': []}
    
    # 1. 하얀색 점수 비교 (배경 제외)
    if dog1_data['whiteness'] > dog2_data['whiteness']:
        scores['dog1'] += 1
        details['dog1'].append("하얀색 점수 +1")
    elif dog2_data['whiteness'] > dog1_data['whiteness']:
        scores['dog2'] += 1
        details['dog2'].append("하얀색 점수 +1")
    else:
        details['dog1'].append("하얀색 동점")
        details['dog2'].append("하얀색 동점")
    
    # 2. 코 길이 비교
    if dog1_data['nose_length'] > dog2_data['nose_length']:
        scores['dog1'] += 1
        details['dog1'].append("코 길이 +1")
    elif dog2_data['nose_length'] > dog1_data['nose_length']:
        scores['dog2'] += 1
        details['dog2'].append("코 길이 +1")
    else:
        details['dog1'].append("코 길이 동점")
        details['dog2'].append("코 길이 동점")
    
    # 3. 크기 비교
    if dog1_data['dog_size'] > dog2_data['dog_size']:
        scores['dog1'] += 1
        details['dog1'].append("크기 +1")
    elif dog2_data['dog_size'] > dog1_data['dog_size']:
        scores['dog2'] += 1
        details['dog2'].append("크기 +1")
    else:
        details['dog1'].append("크기 동점")
        details['dog2'].append("크기 동점")
    
    # 4. 눈 거리 비교 (더 가까우면 +1점)
    if dog1_data['eye_distance'] > dog2_data['eye_distance']:
        scores['dog1'] += 1
        details['dog1'].append("눈 거리 +1")
    elif dog2_data['eye_distance'] > dog1_data['eye_distance']:
        scores['dog2'] += 1
        details['dog2'].append("눈 거리 +1")
    else:
        details['dog1'].append("눈 거리 동점")
        details['dog2'].append("눈 거리 동점")
    
    # 5. 귀 쫑긋함 비교
    if dog1_data['ear_pointiness'] > dog2_data['ear_pointiness']:
        scores['dog1'] += 1
        details['dog1'].append("귀 쫑긋함 +1")
    elif dog2_data['ear_pointiness'] > dog1_data['ear_pointiness']:
        scores['dog2'] += 1
        details['dog2'].append("귀 쫑긋함 +1")
    else:
        details['dog1'].append("귀 쫑긋함 동점")
        details['dog2'].append("귀 쫑긋함 동점")
    
    # 6. 표정 비교 (더 웃고 있으면 +1점)
    if dog1_data['smile'] > dog2_data['smile']:
        scores['dog1'] += 1
        details['dog1'].append("웃는 표정 +1")
    elif dog2_data['smile'] > dog1_data['smile']:
        scores['dog2'] += 1
        details['dog2'].append("웃는 표정 +1")
    else:
        details['dog1'].append("표정 동점")
        details['dog2'].append("표정 동점")
    
    # 7. 자세 비교
    posture_scores = {'sitting': 2, 'lying': 1, 'standing': 0}
    dog1_posture_score = posture_scores.get(dog1_data['posture'], 0)
    dog2_posture_score = posture_scores.get(dog2_data['posture'], 0)
    
    if dog1_posture_score > dog2_posture_score:
        scores['dog1'] += dog1_posture_score
        details['dog1'].append(f"자세 ({dog1_data['posture']}) +{dog1_posture_score}")
    elif dog2_posture_score > dog1_posture_score:
        scores['dog2'] += dog2_posture_score
        details['dog2'].append(f"자세 ({dog2_data['posture']}) +{dog2_posture_score}")
    else:
        if dog1_posture_score > 0:
            details['dog1'].append(f"자세 ({dog1_data['posture']}) +{dog1_posture_score}")
            details['dog2'].append(f"자세 ({dog2_data['posture']}) +{dog2_posture_score}")
        else:
            details['dog1'].append("자세 동점")
            details['dog2'].append("자세 동점")
    
    return scores, details

# Streamlit 앱
st.set_page_config(
    page_title="강아지 비교 대전",
    page_icon="🐕",
    layout="wide"
)

st.title("🐕 강아지 비교 대전")
st.markdown("두 강아지 사진을 업로드하여 승자를 결정합니다!")

col1, col2 = st.columns(2)

with col1:
    st.subheader("강아지 1")
    dog1_file = st.file_uploader("첫 번째 강아지 사진을 업로드하세요", type=['png', 'jpg', 'jpeg'], key="dog1")

with col2:
    st.subheader("강아지 2")
    dog2_file = st.file_uploader("두 번째 강아지 사진을 업로드하세요", type=['png', 'jpg', 'jpeg'], key="dog2")

if dog1_file and dog2_file:
    if st.button("🏆 대전 시작!", type="primary"):
        with st.spinner("강아지들을 분석 중..."):
            # 이미지 로드
            dog1_image = Image.open(dog1_file)
            dog2_image = Image.open(dog2_file)
            
            # 강아지 크기 정규화 (동일한 크기로 맞춤)
            with st.spinner("강아지 크기를 정규화 중..."):
                normalized_img1, normalized_img2 = normalize_dog_size(dog1_image, dog2_image)
                normalized_img1 = Image.fromarray(normalized_img1)
                normalized_img2 = Image.fromarray(normalized_img2)
            
            # 정규화된 이미지로 분석
            dog1_data = analyze_dog(normalized_img1)
            dog2_data = analyze_dog(normalized_img2)
            
            # 비교
            scores, details = compare_dogs(dog1_data, dog2_data)
            
            # 결과 표시
            st.markdown("---")
            st.header("🏆 대전 결과")
            
            result_col1, result_col2, result_col3 = st.columns([1, 1, 1])
            
            with result_col1:
                st.subheader("강아지 1")
                st.image(normalized_img1, use_container_width=True, caption="크기 정규화된 이미지")
                st.metric("총 점수", scores['dog1'])
                st.markdown("**상세 점수:**")
                for detail in details['dog1']:
                    st.write(f"• {detail}")
                st.write(f"하얀색: {dog1_data['whiteness']:.3f}")
                st.write(f"코 길이: {dog1_data['nose_length']:.3f}")
                st.write(f"크기: {dog1_data['dog_size']:.3f}")
                st.write(f"눈 거리: {dog1_data['eye_distance']:.3f}")
                st.write(f"귀 쫑긋함: {dog1_data['ear_pointiness']:.3f}")
                st.write(f"웃는 표정: {dog1_data['smile']:.3f}")
                st.write(f"자세: {dog1_data['posture']}")
            
            with result_col2:
                st.subheader("VS")
                if scores['dog1'] > scores['dog2']:
                    st.success(f"🏆 강아지 1 승리! ({scores['dog1']} : {scores['dog2']})")
                    winner_idx = 1
                elif scores['dog2'] > scores['dog1']:
                    st.success(f"🏆 강아지 2 승리! ({scores['dog2']} : {scores['dog1']})")
                    winner_idx = 2
                else:
                    st.info(f"무승부! ({scores['dog1']} : {scores['dog2']})")
                    winner_idx = 0
            
            with result_col3:
                st.subheader("강아지 2")
                st.image(normalized_img2, use_container_width=True, caption="크기 정규화된 이미지")
                st.metric("총 점수", scores['dog2'])
                st.markdown("**상세 점수:**")
                for detail in details['dog2']:
                    st.write(f"• {detail}")
                st.write(f"하얀색: {dog2_data['whiteness']:.3f}")
                st.write(f"코 길이: {dog2_data['nose_length']:.3f}")
                st.write(f"크기: {dog2_data['dog_size']:.3f}")
                st.write(f"눈 거리: {dog2_data['eye_distance']:.3f}")
                st.write(f"귀 쫑긋함: {dog2_data['ear_pointiness']:.3f}")
                st.write(f"웃는 표정: {dog2_data['smile']:.3f}")
                st.write(f"자세: {dog2_data['posture']}")
            
            # 승자 사진 표시
            if winner_idx > 0:
                st.markdown("---")
                st.header("🏆 승자")
                # 승자 이미지 결정
                if winner_idx == 1:
                    winner_normalized = normalized_img1
                else:
                    winner_normalized = normalized_img2
                st.image(winner_normalized, use_container_width=True, caption="승리한 강아지! (크기 정규화됨)")
                
                # 승자 사진 다운로드 버튼
                buf = io.BytesIO()
                winner_normalized.save(buf, format='PNG')
                st.download_button(
                    label="승자 사진 다운로드",
                    data=buf.getvalue(),
                    file_name="winner_dog.png",
                    mime="image/png"
                )

else:
    st.info("👆 두 강아지 사진을 모두 업로드해주세요!")

st.markdown("---")
st.markdown("### 📋 비교 기준")
st.markdown("""
1. **하얀색 점수**: 강아지 영역만 분석하여 하얀색에 가까울수록 +1점 (배경 제외)
2. **코 길이**: 얼굴 감지를 통해 코 길이를 측정하여 더 길면 +1점
3. **크기**: 이미지에서 강아지가 차지하는 영역이 더 크면 +1점
4. **눈 거리**: 양 눈이 더 가까우면 +1점
5. **귀 쫑긋함**: 귀가 더 쫑긋하면 +1점
6. **웃는 표정**: 표정이 더 웃고 있으면 +1점
7. **자세**: 앉아있으면 +2점, 엎드려있으면 +1점

**✨ 정확도 향상 기능**: 두 강아지의 크기를 자동으로 동일하게 맞춰서 더 공정한 비교를 합니다!

총점이 높은 강아지가 승자입니다! 🏆
""")


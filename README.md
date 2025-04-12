# Suitcase_count

![suitcase](https://github.com/user-attachments/assets/95a72faf-8766-48e4-a0ae-6e63bafb583d)


https://github.com/user-attachments/assets/f7c635ec-7fda-424d-9878-cefe27a52b81

## Overview
In many computer vision applications at airports, accurately counting how many suitcases enter or remain within specific zones—such as security areas or store aisles—is crucial. This project utilizes:

- **YOLOv11** for suitcase detection.  
- **Built-in object tracking** to maintain a consistent ID for each person across frames.  
- **Shapely** or a similar geometry library to define polygons (regions) and check if a person’s “foot point” enters or leaves a region.  
- **Real-time drawing** of bounding boxes, IDs, and region overlays on the video feed.

## Features
1. **Detection**: Uses YOLOv11 to detect suitcase in each frame of a video.  
2. **Tracking**: Assigns unique IDs to each suitcase, allowing accurate counting even if a suitcase leaves and re-enters the region.  
3. **Region-based Counting**: Tracks how many unique IDs enter each polygon-defined region.  
4. **Visualization**: Draws bounding boxes, tracking IDs, region polygons, and counts.

## Requirements and Installation
1. **Python 3.8+** recommended.  
2. **CUDA toolkit** (optional) for GPU acceleration if you have a compatible NVIDIA GPU.  



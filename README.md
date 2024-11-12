# Automated License Plate Recognition (ALPR) System

## Overview

This project is a comprehensive Automated License Plate Recognition (ALPR) system designed for real-time detection and recognition of vehicle license plates in video feeds. It combines advanced object detection and Optical Character Recognition (OCR) methods to provide a robust, scalable, and privacy-conscious solution for various applications, including traffic monitoring, law enforcement, and secure access control.

Key features include:
- **Real-time License Plate Detection** using YOLO (You Only Look Once) for high-speed and accurate plate localization.
- **Optical Character Recognition (OCR)** using Pytesseract to extract alphanumeric characters from license plates.
- **Privacy Protection** through Gaussian blur on detected license plates to safeguard sensitive data.
- **Data Logging** with timestamps for historical tracking and analysis.

## Table of Contents
- [Architecture](#architecture)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Results](#results)
- [Future Enhancements](#future-enhancements)

## Architecture

The ALPR system is organized in modules:
1. **Data Acquisition**: Captures video feeds and processes frames individually.
2. **Pre-processing**: Converts frames to grayscale and applies bilateral filtering to enhance clarity.
3. **License Plate Detection**: YOLO model locates plates, refined with Region of Interest (ROI) filtering.
4. **OCR Processing**: Pytesseract extracts characters, validated against standard license plate formats.
5. **Privacy Protection**: Gaussian blur is selectively applied to protect license plate information.
6. **Data Logging**: Logs detected plates with timestamps in a `.csv` file for analysis.

## Features

- **Scalability**: Operates on both standard CPUs and higher-end GPUs.
- **Real-time Processing**: Efficient for high-traffic applications.
- **Privacy Control**: Selective blurring maintains compliance with privacy standards.
- **Cross-platform Support**: Runs on Windows, macOS, and Linux.

## Installation

To run this ALPR system, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/ALPR-System.git
   cd ALPR-System

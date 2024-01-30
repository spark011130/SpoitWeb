# 골 디택션 모듈
from __future __ import annotations
from typing import List, Tuple
from supervision.video.sink import VideoSink
import torch
from supervision.video.dataclasses import VideoInfo
from supervision.video.source import get_video_frames_generator
from shapely.geometry import Point, Polygon

from ultralytics import YOLO # 객체 탐지 모듈
import supervision as sv # 객체 탐지 라벨링 모듈
import cv2 # 컴퓨터 비전 관련 모듈
from flask import Flask, render_template, request, redirect, jsonify # 웹 프레임워크 관련 모듈
from werkzeug.utils import secure_filename # 링크 보안 관련 모듈
import boto3 # 서버 관련 모듈
import zipfile # 압축 관련 모듈
import os # 경로 관련 모듈
import shutil # 디렉토리의 내용 삭제 관련 모듈
import logging # 서버 로그 모듈
from botocore.exceptions import NoCredentialsError, ClientError # 서버 에러 핸들링 관련 모듈
import pandas as pd # 데이터 분석 모듈
import mplsoccer # 축구 데이터 분석용 모듈
import warnings # pandas warning 무시용
from tqdm import tqdm
from markupsafe import Markup
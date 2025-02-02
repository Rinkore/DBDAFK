import logging
from logging.handlers import RotatingFileHandler
import pyautogui
import pytesseract
import time
import threading
import random
from pynput import keyboard
import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import re
import pystray
from pystray import MenuItem as item
import os
import sys
import json

# ====== 默认配置 ======
DEFAULT_CONFIG = {
    "debug_settings": {
        "debug_mode": True,
        "debug_dir": "debug",
        "log_dir": "logs",
        "image_dir": "images"
    },
    "region_config": {
        "default": [460, 50, 660, 700],
        "continue": [1754, 1004, 47, 24],
        "start_ready": [1677, 934, 125, 29],
        "count": [170, 800, 40, 50],
        "ok": [1382, 644, 44, 24],
        "start": [218, 94, 44, 24],
        "killer": [248, 209, 45, 26]
    },
    "operation_config": {
        "check_interval": 3,
        "image_similarity_threshold": 0.45,
        "image_capture_interval": 0.2
    },
    "turn_config": {
        "base_duration": 0.02,
        "max_duration": 0.4,
        "turn_patterns": [
            ["left", "hold"],
            ["right", "hold"]
        ]
    },
    "skill_config": {
        "skill_patterns": [
            ["leftclick", "press"],
            ["rightclick", "hold"],
            ["leftclick", "hold"],
            ["ctrl", "hold"]
        ],
        "hold_duration_range": [2.0, 2.2]
    }
}

# ====== 全局变量占位 ======
config = None
DEBUG_MODE = False
DEBUG_DIR = ""
DEBUG_LOG_DIR = ""
DEBUG_IMAGE_DIR = ""
region_config = {}
check_interval = 3
IMAGE_SIMILARITY_THRESHOLD = 0.45
IMAGE_CAPTURE_INTERVAL = 0.2
TURN_CONFIG = {}
SKILL_CONFIG = {}
exit_flag = False
movement_lock = threading.Lock()
number_recognition_enabled = True
threads = []
CLEAN_REGEX = re.compile(r'[\s\u3000]')
threshold_adjustment = 0.0
consecutive_turns = 0
consecutive_no_turns = 0

# ====== 配置加载函数 ======
def load_config(config_path="config.json"):
    """加载或生成配置文件"""
    if not os.path.exists(config_path):
        try:
            # 处理路径问题
            config_dir = os.path.dirname(config_path) or os.getcwd()
            os.makedirs(config_dir, exist_ok=True)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(DEFAULT_CONFIG, f, indent=4, ensure_ascii=False)
            logging.info(f"已生成默认配置文件: {os.path.abspath(config_path)}")
        except Exception as e:
            logging.error(f"创建配置文件失败: {str(e)}")
            return DEFAULT_CONFIG
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"加载配置文件失败: {str(e)}")
        return DEFAULT_CONFIG

# ====== 更新全局配置 ======
def update_global_config():
    global config, DEBUG_MODE, DEBUG_DIR, DEBUG_LOG_DIR, DEBUG_IMAGE_DIR
    global region_config, check_interval, IMAGE_SIMILARITY_THRESHOLD, IMAGE_CAPTURE_INTERVAL
    global TURN_CONFIG, SKILL_CONFIG
    
    # 更新调试配置
    DEBUG_MODE = config['debug_settings']['debug_mode']
    DEBUG_DIR = config['debug_settings']['debug_dir']
    DEBUG_LOG_DIR = os.path.join(DEBUG_DIR, config['debug_settings']['log_dir'])
    DEBUG_IMAGE_DIR = os.path.join(DEBUG_DIR, config['debug_settings']['image_dir'])
    
    # 更新区域配置
    region_config = {k: tuple(v) for k, v in config['region_config'].items()}
    
    # 更新操作配置
    check_interval = config['operation_config']['check_interval']
    IMAGE_SIMILARITY_THRESHOLD = config['operation_config']['image_similarity_threshold']
    IMAGE_CAPTURE_INTERVAL = config['operation_config']['image_capture_interval']
    
    # 更新转向配置
    TURN_CONFIG = {
        'base_duration': config['turn_config']['base_duration'],
        'max_duration': config['turn_config']['max_duration'],
        'turn_patterns': [tuple(p) for p in config['turn_config']['turn_patterns']]
    }
    
    # 更新技能配置
    SKILL_CONFIG = {
        'skill_patterns': [tuple(p) for p in config['skill_config']['skill_patterns']],
        'hold_duration_range': tuple(config['skill_config']['hold_duration_range'])
    }

# ====== 日志系统 ======
def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(module)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台日志
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)
    console_handler.setFormatter(formatter)

    # 文件日志
    os.makedirs(DEBUG_LOG_DIR, exist_ok=True)
    log_file = os.path.join(DEBUG_LOG_DIR, 'afk_helper.log')
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5*1024*1024,
        backupCount=3,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
# ====== 调试功能 ======
def save_debug_image(image, prefix):
    try:
        timestamp = int(time.time() * 1000)
        filename = f"{prefix}_{timestamp}.png"
        filepath = os.path.join(DEBUG_IMAGE_DIR, filename)
        
        if isinstance(image, np.ndarray):
            cv2.imwrite(filepath, image)
        elif isinstance(image, Image.Image):
            image.save(filepath)
        else:
            logging.warning(f"Unsupported image format: {type(image)}")
            
        logging.debug(f"Saved debug image: {filepath}")
        return filepath
    except Exception as e:
        logging.error(f"Failed to save debug image: {str(e)}")
        return None

# ====== 系统托盘 ======
def get_icon_path():
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, "Rinkore28.ico")

def create_tray_icon():
    try:
        icon_image = Image.open(get_icon_path())
        menu = (item('退出', on_exit_clicked),)
        return pystray.Icon("DBDAFK", icon_image, "DBDAFK", menu)
    except Exception as e:
        logging.error(f"托盘图标创建失败: {str(e)}")
        return None

def on_exit_clicked(icon):
    global exit_flag
    logging.info("收到退出请求")
    exit_flag = True
    if icon:
        icon.stop()
    os._exit(0)

# ====== 核心功能 ======
def init_position():
    try:
        region = region_config['default']
        x = random.randint(region[0], region[0] + region[2])
        y = random.randint(region[1], region[1] + region[3])
        pyautogui.moveTo(x, y, duration=random.uniform(0.1, 0.3))
        logging.debug(f"初始化位置: ({x}, {y})")
    except Exception as e:
        logging.error(f"初始化失败: {str(e)}")

def preprocess_image(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)
        kernel = np.ones((3,3), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3,3))
        return clahe.apply(cleaned)
    except Exception as e:
        logging.error(f"图像预处理失败: {str(e)}")
        return None

def validate_result(text):
    correction_map = {
        '7': '1', '0': '5', 's': '5', ']': '1',
        '5': '5', '1': '1', '2': '2', '3': '3', '4': '4'
    }
    cleaned = text.strip()
    if cleaned in ['1', '2', '3', '4', '5']:
        return int(cleaned)
    result = correction_map.get(cleaned, '99')
    try: return int(result)
    except ValueError: return 99

def ocr_text(screenshot, region):
    try:
        left, top, width, height = region
        img = screenshot.crop((left, top, left+width, top+height))
        if DEBUG_MODE:
            save_debug_image(img, "ocr")
        text = pytesseract.image_to_string(img, lang='chi_sim', config='--psm 6').strip()
        logging.debug(f"OCR识别结果: {text} (区域: {region})")
        return text
    except Exception as e:
        logging.error(f"OCR失败: {str(e)}")
        return ""

def ocr_number(screenshot, region):
    try:
        left, top, width, height = region
        img_pil = screenshot.crop((left, top, left+width, top+height))
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        processed = preprocess_image(img)
        if DEBUG_MODE:
            save_debug_image(processed, "number")
        text = pytesseract.image_to_string(processed, config='--psm 10 --oem 3 -c tessedit_char_whitelist=12345')
        result = validate_result(text)
        logging.debug(f"数字识别: {text} -> {result}")
        return result
    except Exception as e:
        logging.error(f"数字识别失败: {str(e)}")
        return 99

def click_center(region):
    try:
        x = region[0] + region[2] // 2
        y = region[1] + region[3] // 2
        pyautogui.click(x, y)
        logging.info(f"点击坐标: ({x}, {y}) 区域: {region}")
    except Exception as e:
        logging.error(f"点击失败: {str(e)}")

# ====== 自动化模块 ======
def image_monitor():
    prev_left, prev_right = None, None
    logging.info("图像监控启动")
    while not exit_flag:
        try:
            screenshot = np.array(pyautogui.screenshot())
            if DEBUG_MODE:
                save_debug_image(screenshot, "full_screen")
            gray = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)
            h, w = gray.shape
            left, right = gray[:, :w//2], gray[:, w//2:]
            
            if prev_left is not None and prev_right is not None:
                left_sim = ssim(left, prev_left)
                right_sim = ssim(right, prev_right)
                dynamic_threshold = IMAGE_SIMILARITY_THRESHOLD + threshold_adjustment
                logging.debug(f"相似度: 左={left_sim:.4f} 右={right_sim:.4f} 阈值={dynamic_threshold:.4f}")
                if left_sim > dynamic_threshold or right_sim > dynamic_threshold:
                    perform_turn(left_sim, right_sim)
            
            prev_left, prev_right = left, right
            time.sleep(IMAGE_CAPTURE_INTERVAL)
        except Exception as e:
            logging.error(f"图像监控异常: {str(e)}")
            time.sleep(1)



def perform_turn(left_sim, right_sim):
    global consecutive_turns, consecutive_no_turns
    try:
        dynamic_threshold = IMAGE_SIMILARITY_THRESHOLD + threshold_adjustment
        turn_direction = 'right' if left_sim < right_sim else 'left'
        duration = TURN_CONFIG['base_duration'] + (TURN_CONFIG['max_duration'] - TURN_CONFIG['base_duration']) * (1 - max(left_sim, right_sim))
        duration = min(duration, TURN_CONFIG['max_duration'])
        logging.info(f"转向: 方向={turn_direction} 时长={duration:.2f}s")
        with movement_lock:
            pyautogui.keyDown(turn_direction)
            time.sleep(duration)
            pyautogui.keyUp(turn_direction)
        perform_skill()
        consecutive_turns += 1
        consecutive_no_turns = 0
    except Exception as e:
        logging.error(f"转向失败: {str(e)}")

def perform_skill():
    try:
        skill, action = random.choice(SKILL_CONFIG['skill_patterns'])
        logging.info(f"释放技能: {skill} 类型: {action}")
        with movement_lock:
            if action == 'press':
                if skill == 'leftclick':
                    pyautogui.click(button='left')
                elif skill == 'rightclick':
                    pyautogui.click(button='right')
                else:
                    pyautogui.press(skill)
            elif action == 'hold':
                duration = random.uniform(*SKILL_CONFIG['hold_duration_range'])
                logging.debug(f"技能持续: {duration:.2f}s")
                if skill in ['leftclick', 'rightclick']:
                    pyautogui.mouseDown(button=skill[:-5])
                    time.sleep(duration)
                    pyautogui.mouseUp(button=skill[:-5])
                else:
                    pyautogui.keyDown(skill)
                    time.sleep(duration)
                    pyautogui.keyUp(skill)
    except Exception as e:
        logging.error(f"技能失败: {str(e)},{skill},{skill[:-5]}")

def auto_patrol():
    logging.info("巡逻启动")
    try:
        with movement_lock:
            pyautogui.keyDown('w')
        while not exit_flag:
            if random.random() < 0.25:
                with movement_lock:
                    pyautogui.keyUp('w')
                    time.sleep(random.uniform(0.05, 0.15))
                    pyautogui.keyDown('w')
            time.sleep(0.5)
    except Exception as e:
        logging.error(f"巡逻异常: {str(e)}")
    finally:
        with movement_lock:
            pyautogui.keyUp('w')

# ====== 主逻辑 ======
def process_continue(screenshot):
    text = ocr_text(screenshot, region_config['continue'])
    if "继续" in text:
        logging.info("检测到继续按钮")
        click_center(region_config['continue'])
        reset_system_state()
        return True
    return False

def process_buttons(screenshot):
    buttons = [('ok', '好的'), ('killer', '杀手'), ('start', '开始')]
    for btn_type, text in buttons:
        raw = ocr_text(screenshot, region_config[btn_type])
        cleaned = CLEAN_REGEX.sub('', raw)
        if text in cleaned:
            logging.info(f"检测到按钮: {text}")
            click_center(region_config[btn_type])
            time.sleep(0.5 + random.uniform(0, 0.3))
            return True
    return False

def process_start_ready(screenshot):
    text = ocr_text(screenshot, region_config['start_ready'])
    if any(x in text for x in ["开始游戏", "准备就绪"]):
        logging.info("检测到开始状态")
        click_center(region_config['start_ready'])
        init_position()
        return True
    return False

def process_number(screenshot):
    global number_recognition_enabled
    if not number_recognition_enabled:
        return False
    try:
        count = ocr_number(screenshot, region_config['count'])
        if count == 5:
            logging.info("检测到数字5")
            number_recognition_enabled = False
            click_center(region_config['start_ready'])
            time.sleep(1)
            start_workers()
            return True
        else:
            pyautogui.keyDown('right')
            time.sleep(0.1)
            pyautogui.keyUp('right')
    except Exception as e:
        logging.error(f"数字处理异常: {str(e)}")
    return False

def reset_system_state():
    global exit_flag, number_recognition_enabled
    logging.info("系统状态重置")
    exit_flag = True
    stop_workers()
    init_position()
    number_recognition_enabled = True
    exit_flag = False

def stop_workers():
    global threads
    if threads:
        logging.info("停止工作线程")
        for t in threads:
            if t.is_alive():
                t.join(timeout=1)
        threads.clear()

def start_workers():
    global threads
    if not threads:
        logging.info("启动工作线程")
        threads = [
            threading.Thread(target=auto_patrol, daemon=True),
            threading.Thread(target=image_monitor, daemon=True)
        ]
        for t in threads:
            t.start()

def on_press(key):
    global exit_flag
    if key == keyboard.Key.enter:
        logging.warning("检测到ENTER键，退出")
        exit_flag = True
        with movement_lock:
            for k in ['w','a','d','ctrl']:
                pyautogui.keyUp(k)
            pyautogui.mouseUp(button='left')
            pyautogui.mouseUp(button='right')

# ====== 主循环 ======
def main_loop():
    
    # 加载配置文件
    global config
    config = load_config()
    update_global_config()  # 更新全局变量
    
    # 初始化调试目录
    if DEBUG_MODE:
        os.makedirs(DEBUG_LOG_DIR, exist_ok=True)
        os.makedirs(DEBUG_IMAGE_DIR, exist_ok=True)
    
    # 初始化日志系统
    setup_logging()
    logging.info("===== 程序启动 =====")
    
    # 初始化监听器和系统托盘
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    
    tray_icon = create_tray_icon()
    if tray_icon:
        threading.Thread(target=tray_icon.run, daemon=True).start()
    
    try:
        cycle_count = 0
        while not exit_flag:
            cycle_count += 1
            logging.debug(f"主循环周期 #{cycle_count}")
            start_time = time.time()
            screenshot = pyautogui.screenshot()
            if DEBUG_MODE:
                save_debug_image(screenshot, "main_loop")
            processed = False
            for func in [process_continue, process_buttons, process_start_ready, process_number]:
                if func(screenshot):
                    processed = True
                    break
            elapsed = time.time() - start_time
            remain = check_interval - elapsed
            if remain > 0: 
                time.sleep(remain)
            else: 
                logging.warning(f"循环超时: {elapsed:.2f}s")
    except Exception as e:
        logging.critical(f"主循环崩溃: {str(e)}", exc_info=True)
    finally:
        logging.info("===== 程序退出 =====")
        stop_workers()
        listener.stop()

if __name__ == "__main__":
    main_loop()
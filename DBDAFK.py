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

# ====== 调试目录配置 ======
DEBUG_DIR = "debug"
DEBUG_LOG_DIR = os.path.join(DEBUG_DIR, "logs")
DEBUG_IMAGE_DIR = os.path.join(DEBUG_DIR, "images")

# ====== 全局配置 ======
DEBUG_MODE = False
exit_flag = False
movement_lock = threading.Lock()
number_recognition_enabled = True
threads = []
CLEAN_REGEX = re.compile(r'[\s\u3000]')
check_interval = 3

# ====== 区域配置 ======
region_config = {
    'default': (460, 500, 660, 740),
    'continue': (1754, 1004, 47, 24),
    'start_ready': (1677, 934, 125, 29),
    'count': (170, 800, 40, 50),
    'ok': (1382, 644, 44, 24),
    'start': (218, 94, 44, 24),
    'killer': (248, 209, 45, 26)
}

# ====== 操作配置 ======
TURN_CONFIG = {
    'base_duration': 0.02,
    'max_duration': 0.4,
    'turn_patterns': [
        ('left', 'hold'),
        ('right', 'hold')
    ]
}

SKILL_CONFIG = {
    'skill_patterns': [
        ('leftclick', 'press'),
        ('rightclick', 'hold'),
        ('leftclick', 'hold'),
        ('ctrl', 'hold')
    ],
    'hold_duration_range': (2.0, 2.2)
}

IMAGE_SIMILARITY_THRESHOLD = 0.45
IMAGE_CAPTURE_INTERVAL = 0.2
threshold_adjustment = 0.0
consecutive_turns = 0
consecutive_no_turns = 0

# ====== 日志配置 ======
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

# ====== 调试图片保存 ======
def save_debug_image(image, prefix):
    """保存调试图片到debug/images目录"""
    try:
        timestamp = int(time.time() * 1000)
        filename = f"{prefix}_{timestamp}.png"
        filepath = os.path.join(DEBUG_IMAGE_DIR, filename)
        
        if isinstance(image, np.ndarray):
            cv2.imwrite(filepath, image)
        elif isinstance(image, Image.Image):
            image.save(filepath)
        else:
            logging.warning(f"不支持的图像格式: {type(image)}")
            
        logging.debug(f"调试图片已保存: {filepath}")
        return filepath
    except Exception as e:
        logging.error(f"保存调试图片失败: {str(e)}")
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
        logging.error(f"系统托盘创建失败: {str(e)}")
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
        logging.debug(f"初始化鼠标位置: ({x}, {y})")
    except Exception as e:
        logging.error(f"初始化位置失败: {str(e)}")

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
    """
    验证并修正OCR识别的数字结果
    :param text: OCR识别出的文本
    :return: 修正后的数字（1-5），若无法识别则返回99
    """
    # 定义常见误识别映射
    correction_map = {
        '7': '1',  # 7常被误识别为1
        '0': '5',  # 0常被误识别为5
        's': '5',  # s常被误识别为5
        ']': '1',  # ]常被误识别为1
        '5': '5',  # 明确保留5
        '1': '1',  # 明确保留1
        '2': '2',  # 明确保留2
        '3': '3',  # 明确保留3
        '4': '4'   # 明确保留4
    }
    
    # 清理输入文本
    cleaned = text.strip()
    
    # 如果清理后的文本直接是有效数字，直接返回
    if cleaned in ['1', '2', '3', '4', '5']:
        return int(cleaned)
    
    # 尝试从映射表中获取修正结果
    result = correction_map.get(cleaned, '99')
    
    # 返回修正后的结果
    try:
        return int(result)
    except ValueError:
        return 99

# ====== 修改后的OCR函数 ======
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
        logging.error(f"OCR识别失败: {str(e)}")
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
        logging.debug(f"数字识别结果: {text} -> {result}")
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
        logging.error(f"点击操作失败: {str(e)}")

# ====== 自动操作 ======
def image_monitor():
    prev_left, prev_right = None, None
    logging.info("图像监控线程启动")
    
    while not exit_flag:
        try:
            start_time = time.time()
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
                
                logging.debug(f"图像相似度: 左={left_sim:.4f} 右={right_sim:.4f} 阈值={dynamic_threshold:.4f}")
                
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
        
        duration = TURN_CONFIG['base_duration'] + \
                  (TURN_CONFIG['max_duration'] - TURN_CONFIG['base_duration']) * \
                (1 - max(left_sim, right_sim))
        duration = min(duration, TURN_CONFIG['max_duration'])
        
        logging.info(f"执行转向: 方向={turn_direction} 时长={duration:.2f}s")
        
        with movement_lock:
            pyautogui.keyDown(turn_direction)
            time.sleep(duration)
            pyautogui.keyUp(turn_direction)
        
        perform_skill()
        consecutive_turns += 1
        consecutive_no_turns = 0
    except Exception as e:
        logging.error(f"转向操作失败: {str(e)}")

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
                logging.debug(f"技能持续时间: {duration:.2f}s")
                if skill in ['leftclick', 'rightclick']:
                    pyautogui.mouseDown(button=skill[:-5])
                    time.sleep(duration)
                    pyautogui.mouseUp(button=skill[:-5])
                else:
                    pyautogui.keyDown(skill)
                    time.sleep(duration)
                    pyautogui.keyUp(skill)
    except Exception as e:
        logging.error(f"技能释放失败: {str(e)},{skill},{skill[:-5]}")

def auto_patrol():
    logging.info("巡逻线程启动")
    try:
        with movement_lock:
            pyautogui.keyDown('w')
        
        while not exit_flag:
            if random.random() < 0.25:
                with movement_lock:
                    pyautogui.keyUp('w')
                    sleep_time = random.uniform(0.05, 0.15)
                    time.sleep(sleep_time)
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
    buttons = [
        ('ok', '好的'),
        ('start', '开始'),
        ('killer', '杀手')
    ]
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
        logging.info("检测到开始游戏状态")
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
            logging.info("检测到数字5，触发自动化")
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
    logging.info("重置系统状态")
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
        logging.warning("检测到ENTER键，安全退出")
        exit_flag = True
        with movement_lock:
            for k in ['w','a','d','ctrl']:
                pyautogui.keyUp(k)
            pyautogui.mouseUp(button='left')
            pyautogui.mouseUp(button='right')

def main_loop():
    if DEBUG_MODE:
        setup_logging()
        logging.info("===== 程序启动 =====")
        logging.info(f"调试目录: {DEBUG_DIR}")
        logging.info(f"日志文件: {os.path.join(DEBUG_LOG_DIR, 'afk_helper.log')}")
        logging.info(f"调试图片: {DEBUG_IMAGE_DIR}")
    
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
            for func in [process_continue, process_buttons, 
                        process_start_ready, process_number]:
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
    if DEBUG_MODE:
        # 创建调试目录
        os.makedirs(DEBUG_LOG_DIR, exist_ok=True)
        os.makedirs(DEBUG_IMAGE_DIR, exist_ok=True)
    main_loop()
"""
YOLOv5+DeepSORT多目标跟踪系统
主程序入口
"""
import os
# 解决OpenMP库冲突问题 (PyTorch/NumPy等库都包含OpenMP)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
from PyQt5.QtWidgets import QApplication
from gui.main_window import MainWindow


def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 设置应用程序样式
    app.setStyle('Fusion')
    
    # 创建主窗口
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

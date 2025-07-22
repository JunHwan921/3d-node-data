import sys
import subprocess
from PyQt5 import QtWidgets, QtCore, QtGui

class NodeEditorLauncher(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Node Editor - 런처")
        self.setFixedSize(600, 400)
        
        # 스타일 설정
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2d2d2d;
            }
            QPushButton {
                background-color: #404040;
                color: white;
                border: 2px solid #555;
                padding: 20px;
                font-size: 16px;
                font-weight: bold;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #505050;
                border-color: #2196F3;
            }
            QPushButton:pressed {
                background-color: #303030;
            }
            QLabel {
                color: white;
                font-size: 14px;
            }
        """)
        
        # 중앙 위젯
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        
        # 레이아웃
        layout = QtWidgets.QVBoxLayout(central_widget)
        layout.setContentsMargins(50, 50, 50, 50)
        layout.setSpacing(30)
        
        # 타이틀
        title = QtWidgets.QLabel("3D Node Editor")
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet("""
            font-size: 32px;
            font-weight: bold;
            color: #2196F3;
            margin-bottom: 20px;
        """)
        layout.addWidget(title)
        
        # 설명
        desc = QtWidgets.QLabel("작업을 선택하세요")
        desc.setAlignment(QtCore.Qt.AlignCenter)
        desc.setStyleSheet("font-size: 18px; color: #aaa;")
        layout.addWidget(desc)
        
        # 버튼들
        # 1. 기본 편집기
        basic_btn = QtWidgets.QPushButton("🛠️ 기본 편집기")
        basic_btn.setToolTip("노드 생성, 연결, 기본 편집 작업")
        basic_btn.clicked.connect(self.launch_basic_editor)
        layout.addWidget(basic_btn)
        
        # 2. 패널 편집기
        panel_btn = QtWidgets.QPushButton("🏗️ 패널 편집기")
        panel_btn.setToolTip("패널 생성 및 PANER 연결 작업")
        panel_btn.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0;
            }
            QPushButton:hover {
                background-color: #7B1FA2;
            }
        """)
        panel_btn.clicked.connect(self.launch_panel_editor)
        layout.addWidget(panel_btn)
        
        # 3. 통합 편집기
        full_btn = QtWidgets.QPushButton("🌟 통합 편집기")
        full_btn.setToolTip("모든 기능이 포함된 전체 편집기")
        full_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)
        full_btn.clicked.connect(self.launch_full_editor)
        layout.addWidget(full_btn)
        
        layout.addStretch()
        
        # 상태바
        self.statusBar().showMessage("프로그램을 선택하세요")
    
    def launch_basic_editor(self):
        """기본 편집기 실행"""
        try:
            subprocess.Popen([sys.executable, "main.py", "--mode=basic"])
            self.statusBar().showMessage("기본 편집기 실행 중...", 3000)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "오류", f"실행 실패: {str(e)}")
    
    def launch_panel_editor(self):
        """패널 편집기 실행"""
        try:
            subprocess.Popen([sys.executable, "panel_editor.py"])
            self.statusBar().showMessage("패널 편집기 실행 중...", 3000)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "오류", f"실행 실패: {str(e)}")
    
    def launch_full_editor(self):
        """통합 편집기 실행"""
        try:
            subprocess.Popen([sys.executable, "main.py", "--mode=full"])
            self.statusBar().showMessage("통합 편집기 실행 중...", 3000)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "오류", f"실행 실패: {str(e)}")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    launcher = NodeEditorLauncher()
    launcher.show()
    sys.exit(app.exec_())
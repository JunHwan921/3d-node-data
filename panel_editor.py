# panel_editor.py
import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from PyQt5 import QtWidgets, QtCore
from src.scene_manager import NodeEditor3D
import pyqtgraph.opengl as gl

class PanelEditor(QtWidgets.QMainWindow):
    def __init__(self, input_file=None):
        super().__init__()
        self.editor = NodeEditor3D()
        
        self.setWindowTitle("패널 편집기 - 외장 그룹 전용")
        self.resize(1200, 800)
        
        # UI 설정
        self.setup_ui()
        
        # 데이터 로드
        if input_file and Path(input_file).exists():
            self.load_exterior_data(input_file)
    
    def setup_ui(self):
        """간소화된 UI"""
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QtWidgets.QHBoxLayout(central_widget)
        
        # 좌측 패널
        left_panel = self.create_left_panel()
        
        # 3D 뷰
        self.gl_widget = gl.GLViewWidget()
        self.gl_widget.setCameraPosition(distance=100, elevation=30, azimuth=45)
        
        layout.addWidget(left_panel)
        layout.addWidget(self.gl_widget, 1)
        
        # 툴바
        self.create_toolbar()
    
    def create_left_panel(self):
        """패널 작업 전용 도구"""
        panel = QtWidgets.QWidget()
        panel.setMaximumWidth(300)
        panel.setStyleSheet("background-color: #2d2d2d; color: white;")
        
        layout = QtWidgets.QVBoxLayout(panel)
        
        # 패널 생성 섹션
        title = QtWidgets.QLabel("🏗️ 패널 생성")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)
        
        # 분할 설정
        div_layout = QtWidgets.QHBoxLayout()
        self.div_x = QtWidgets.QSpinBox()
        self.div_x.setRange(1, 20)
        self.div_x.setValue(3)
        self.div_y = QtWidgets.QSpinBox()
        self.div_y.setRange(1, 20)
        self.div_y.setValue(3)
        
        div_layout.addWidget(QtWidgets.QLabel("분할:"))
        div_layout.addWidget(self.div_x)
        div_layout.addWidget(QtWidgets.QLabel("x"))
        div_layout.addWidget(self.div_y)
        layout.addLayout(div_layout)
        
        # 패널 생성 버튼
        create_btn = QtWidgets.QPushButton("패널 생성")
        create_btn.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0;
                color: white;
                padding: 10px;
                font-weight: bold;
            }
        """)
        create_btn.clicked.connect(self.create_panels)
        layout.addWidget(create_btn)
        
        # 정보 표시
        self.info_text = QtWidgets.QTextEdit()
        self.info_text.setMaximumHeight(200)
        self.info_text.setReadOnly(True)
        layout.addWidget(self.info_text)
        
        layout.addStretch()
        return panel
    
    def create_toolbar(self):
        """툴바"""
        toolbar = self.addToolBar('Main')
        toolbar.addAction('💾 결과 저장', self.save_results)
        toolbar.addAction('🔄 메인으로 전송', self.send_to_main)
        toolbar.addAction('❌ 닫기', self.close)
    
    def load_exterior_data(self, filepath):
        """외장 그룹 데이터 로드"""
        self.editor.load_csv(filepath)
        self.update_scene()
        self.info_text.append(f"✅ 외장 그룹 로드: {len(self.editor.scene.nodes)}개 노드")
    
    def create_panels(self):
        """패널 생성 로직"""
        # 여기서 패널 생성
        div_x = self.div_x.value()
        div_y = self.div_y.value()
        
        self.info_text.append(f"\n🏗️ {div_x}x{div_y} 패널 생성 중...")
        # 실제 패널 생성 코드...
        
    def save_results(self):
        """결과 저장"""
        filepath = "temp/panel_results.csv"
        self.editor.save_csv(filepath, include_lines=True)
        self.info_text.append(f"💾 결과 저장: {filepath}")
    
    def send_to_main(self):
        """메인 프로그램으로 결과 전송"""
        # 파일로 저장 후 메인에서 다시 로드하는 방식
        self.save_results()
        QtWidgets.QMessageBox.information(
            self, "전송 완료",
            "패널 데이터가 저장되었습니다.\n메인 프로그램에서 'temp/panel_results.csv'를 로드하세요."
        )
    
    def update_scene(self):
        """3D 씬 업데이트"""
        # 간단한 렌더링 코드
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="입력 파일")
    args = parser.parse_args()
    
    app = QtWidgets.QApplication(sys.argv)
    editor = PanelEditor(input_file=args.input)
    editor.show()
    sys.exit(app.exec_())
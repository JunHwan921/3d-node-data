"""
3D 노드 에디터 메인 프로그램
"""
import sys
from pathlib import Path

# src 디렉토리를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent))

from src.scene_manager import NodeEditor3D
from src.data_structures import LineType
from src.midas_parser import MidasMGBParser  # 절대 import로 변경
import pyqtgraph.opengl as gl
from OpenGL.GL import glMatrixMode, glLoadIdentity, glOrtho, GL_PROJECTION, GL_MODELVIEW
# ✅ 여기에 추가!
import numpy as np  # 이미 있을 수도 있음
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import cdist

# ✅ PanelMapping 클래스 정의 (파일 상단, 전역 레벨)
class PanelMapping:
    """패널 맵핑 정보를 저장하는 클래스"""
    def __init__(self):
        self.panels = {}  # panel_id: {'nodes': [n1, n2, n3, n4], 'type': 'rect', 'group': group_id}
        self.next_panel_id = 1
    
    def add_panel(self, corner_nodes, panel_type='rect', group_id=4):
        """패널 추가"""
        panel_id = f"P{self.next_panel_id:04d}"
        self.panels[panel_id] = {
            'nodes': [n.number for n in corner_nodes],
            'type': panel_type,
            'group': group_id,
            'corners': corner_nodes
        }
        self.next_panel_id += 1
        return panel_id
    
    def to_csv(self, filepath):
        """패널 맵핑 정보를 CSV로 저장"""
        import csv
        with open(filepath, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['panel_id', 'node1', 'node2', 'node3', 'node4', 'type', 'group'])
            
            for panel_id, info in self.panels.items():
                nodes = info['nodes']
                writer.writerow([
                    panel_id,
                    nodes[0], nodes[1], nodes[2], nodes[3],
                    info['type'],
                    info['group']
                ])
# ==================== PanelMapping 클래스 끝 ====================

def test_basic_functionality():
    """기본 기능 테스트"""
    print("=== 3D 노드 에디터 테스트 ===\n")
    
    # 에디터 생성
    editor = NodeEditor3D()
    
    # 1. CSV 파일 로드
    print("1. CSV 파일 로드 테스트")
    csv_path = "data/sample.csv"
    if Path(csv_path).exists():
        editor.load_csv(csv_path)
    else:
        print("   샘플 CSV 파일이 없습니다. 수동으로 노드를 추가합니다.")
        # 수동으로 노드 추가
        editor.add_node_at_position(0.0, 0.0, 0.0, 1)
        editor.add_node_at_position(1.0, 0.0, 0.0, 2)
        editor.add_node_at_position(1.0, 1.0, 0.0, 3)
        editor.add_node_at_position(0.0, 1.0, 0.0, 4)
        editor.add_node_at_position(0.5, 0.5, 1.0, 5)
    
    print(f"   총 노드 수: {len(editor.scene.nodes)}\n")
    
    # 2. 노드 선택
    print("2. 노드 선택 테스트")
    editor.scene.select_nodes_in_region((-0.5, -0.5, -0.5), (1.5, 1.5, 0.5))
    selected_info = editor.scene.get_selected_info()
    print(f"   선택된 노드: {selected_info['count']}개")
    print(f"   노드 번호: {selected_info['numbers']}")
    print(f"   평균 위치: {selected_info['average_position']}\n")
    
    # 3. 라인 연결
    print("3. 라인 연결 테스트")
    if editor.scene.connect_selected_nodes(LineType.MATERIAL):
        print(f"   생성된 라인 수: {len(editor.scene.lines)}\n")
    
    # 4. 새 노드 추가
    print("4. 새 노드 추가 테스트")
    new_node = editor.add_node_at_position(2.0, 2.0, 2.0)
    print(f"   새 노드 번호: {new_node.number}\n")
    
    # 5. CSV 저장
    print("5. CSV 저장 테스트")
    output_path = "output/test_output.csv"
    Path("output").mkdir(exist_ok=True)
    
    # 노드만 저장
    if editor.save_csv(output_path):
        print(f"   노드만 저장: {output_path}")
    
    # 노드와 라인 함께 저장
    if editor.save_csv(output_path, include_lines=True):
        print(f"   전체 데이터 저장: {output_path} + .json\n")
    
    # 6. 실행 취소
    print("6. 실행 취소 테스트")
    print(f"   히스토리 크기: {len(editor.scene.history)}")
    if editor.scene.undo():
        print(f"   실행 취소 후 노드 수: {len(editor.scene.nodes)}\n")
    
    # 7. 씬 정보
    print("7. 씬 정보")
    bounds_min, bounds_max = editor.scene.get_bounds()
    center = editor.scene.get_center()
    print(f"   씬 경계: {bounds_min} ~ {bounds_max}")
    print(f"   씬 중심: {center}")
    print(f"   총 노드 수: {len(editor.scene.nodes)}")
    print(f"   총 라인 수: {len(editor.scene.lines)}")


def interactive_mode():
    """대화형 모드"""
    editor = NodeEditor3D()
    
    print("\n=== 3D 노드 에디터 대화형 모드 ===")
    print("명령어:")
    print("  load <파일경로> - CSV 파일 로드")
    print("  save <파일경로> - CSV 파일 저장")
    print("  add <x> <y> <z> - 노드 추가")
    print("  select all - 모든 노드 선택")
    print("  select box <x1> <y1> <z1> <x2> <y2> <z2> - 박스 영역 선택")
    print("  connect material|paner - 선택된 노드 연결")
    print("  delete - 선택된 노드 삭제")
    print("  undo - 실행 취소")
    print("  info - 씬 정보 표시")
    print("  quit - 종료")
    print()
    
    while True:
        try:
            command = input("> ").strip().lower()
            
            if command == "quit":
                break
                
            elif command.startswith("load "):
                filepath = command[5:].strip()
                editor.load_csv(filepath)
                
            elif command.startswith("save "):
                filepath = command[5:].strip()
                editor.save_csv(filepath, include_lines=True)
                
            elif command.startswith("add "):
                parts = command[4:].split()
                if len(parts) == 3:
                    x, y, z = map(float, parts)
                    editor.add_node_at_position(x, y, z)
                else:
                    print("사용법: add <x> <y> <z>")
                    
            elif command == "select all":
                editor.scene.select_all_nodes()
                info = editor.scene.get_selected_info()
                print(f"선택됨: {info['count']}개 노드")
                
            elif command.startswith("select box "):
                parts = command[11:].split()
                if len(parts) == 6:
                    coords = list(map(float, parts))
                    min_coords = tuple(coords[:3])
                    max_coords = tuple(coords[3:])
                    editor.scene.select_nodes_in_region(min_coords, max_coords)
                    info = editor.scene.get_selected_info()
                    print(f"선택됨: {info['count']}개 노드")
                else:
                    print("사용법: select box <x1> <y1> <z1> <x2> <y2> <z2>")
                    
            elif command.startswith("connect "):
                line_type_str = command[8:].strip()
                if line_type_str == "material":
                    editor.scene.connect_selected_nodes(LineType.MATERIAL)
                elif line_type_str == "paner":
                    editor.scene.connect_selected_nodes(LineType.PANER)
                else:
                    print("사용법: connect material|paner")
                    
            elif command == "delete":
                count = len(editor.scene.selected_nodes)
                editor.scene.remove_selected_nodes()
                print(f"{count}개 노드 삭제됨")
                
            elif command == "undo":
                editor.scene.undo()
                
            elif command == "info":
                print(f"노드: {len(editor.scene.nodes)}개")
                print(f"라인: {len(editor.scene.lines)}개")
                print(f"선택: {len(editor.scene.selected_nodes)}개")
                bounds_min, bounds_max = editor.scene.get_bounds()
                print(f"경계: {bounds_min} ~ {bounds_max}")
                
            else:
                print("알 수 없는 명령어입니다.")
                
        except Exception as e:
            print(f"오류: {str(e)}")

class OrthoViewWidget(gl.GLViewWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.opts['fov'] = 1  # 최소 FOV로 직교에 가깝게
    
    def paintGL(self):
        from PyQt5.QtGui import QMatrix4x4
        from PyQt5.QtGui import QVector3D
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        w, h = self.width(), self.height()
        aspect = w / h if h else 1
        size = self.opts.get('distance', 50)
        # ✅ 최소값 보장
        size = max(size, 1.0)  # 최소값 1.0
        glOrtho(-size*aspect, size*aspect, -size, size, -1000, 1000)
        # 뷰 매트릭스 직접 설정
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        view = QMatrix4x4()
        view.lookAt(
            self.cameraPosition(),
            self.opts['center'],
            QVector3D(0, 0, 1)  # up vector
        )
        super().paintGL() 

def gui_mode_pyqtgraph(mode="full"):  # ✅ mode 매개변수 추가
    """GUI 모드 - PyQtGraph 3D 뷰어"""
    import sys
    import numpy as np
    from PyQt5 import QtCore, QtWidgets, QtGui
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
    from PyQt5.QtGui import QVector4D
    #------------------------------------------------------------------------------------------
         
    
    class PyQtGraph3DViewer(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.editor = NodeEditor3D()
            
            # 윈도우 설정
            self.setWindowTitle('3D Node Editor - PyQtGraph')
            self.resize(1400, 900)
            
            # ✅ 메뉴바 생성 (툴바보다 먼저)
            self.create_menubar()
            
            # 중앙 위젯
            self.central_widget = QtWidgets.QWidget()
            self.setCentralWidget(self.central_widget)
            
            # ✨ 메인 레이아웃 (세로) ✨
            main_layout = QtWidgets.QVBoxLayout(self.central_widget)
            main_layout.setContentsMargins(0, 0, 0, 0)
            
            # 툴바 생성 (맨 위에 고정)
            self.create_toolbar()
            
            # ✨ 스플리터 생성 (좌우 분할) ✨
            splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
            
            # ✨ 좌측 패널 ✨
            self.left_panel = QtWidgets.QWidget()
            self.left_panel.setMinimumWidth(250)
            self.left_panel.setMaximumWidth(800)
            self.left_panel.setStyleSheet("""background-color: #2d2d2d; border: 1px solid #555; color: white;""")
            
            # ✅ 스크롤 영역 추가
            scroll_area = QtWidgets.QScrollArea()
            scroll_area.setWidget(self.left_panel)
            scroll_area.setWidgetResizable(True)
            scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
            scroll_area.setStyleSheet("""
                QScrollArea {
                    background-color: #2d2d2d;
                    border: 1px solid #555;
                }
                QScrollBar:vertical {
                    background-color: #2d2d2d;
                    width: 12px;
                }
                QScrollBar::handle:vertical {
                    background-color: #555;
                    border-radius: 6px;
                }
                QScrollBar::handle:vertical:hover {
                    background-color: #666;
                }
            """)
            
            # ✨ 좌측 패널 레이아웃 (완전히 새로 작성) ✨
            panel_layout = QtWidgets.QVBoxLayout(self.left_panel)
            panel_layout.setContentsMargins(10, 10, 10, 10)
            panel_layout.setSpacing(5)

            # 1. 레이어 관리
            layer_label = QtWidgets.QLabel("레이어 관리")
            layer_label.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 10px; color: white;")
            panel_layout.addWidget(layer_label)

            self.beam_checkbox = QtWidgets.QCheckBox("BEAM")
            self.beam_checkbox.setChecked(True)
            self.beam_checkbox.stateChanged.connect(self.toggle_beam_layer)
            panel_layout.addWidget(self.beam_checkbox)

            self.truss_checkbox = QtWidgets.QCheckBox("TRUSS")
            self.truss_checkbox.setChecked(True)
            self.truss_checkbox.stateChanged.connect(self.toggle_truss_layer)
            panel_layout.addWidget(self.truss_checkbox)

            # 2. 구분선
            separator1 = QtWidgets.QFrame()
            separator1.setFrameShape(QtWidgets.QFrame.HLine)
            separator1.setStyleSheet("color: #555;")
            panel_layout.addWidget(separator1)

            # 3. 카메라 뷰
            view_label = QtWidgets.QLabel("카메라 뷰")
            view_label.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 5px; color: white;")
            panel_layout.addWidget(view_label)

            # 4. 뷰 버튼들 (Top, Front, Left, Right)
            for view_name in ["Top", "Front", "Left", "Right"]:
                btn = QtWidgets.QPushButton(view_name)
                btn.setStyleSheet("""
                    QPushButton { 
                        background-color: #404040; 
                        color: white; 
                        border: 1px solid #666; 
                        padding: 5px; 
                        margin: 2px;
                    }
                    QPushButton:hover { 
                        background-color: #505050; 
                    }
                """)
                btn.clicked.connect(lambda checked, view=view_name.lower(): self.set_view(view))
                panel_layout.addWidget(btn)

            # 5. 구분선
            separator2 = QtWidgets.QFrame()
            separator2.setFrameShape(QtWidgets.QFrame.HLine)
            separator2.setStyleSheet("color: #555;")
            panel_layout.addWidget(separator2)

            # 6. 노드 생성 도구
            node_tools_label = QtWidgets.QLabel("노드 생성 도구")
            node_tools_label.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 5px; color: white;")
            panel_layout.addWidget(node_tools_label)

            # 7. 중점 노드 생성 버튼
            self.midpoint_btn = QtWidgets.QPushButton("중점 노드 생성")
            self.midpoint_btn.setStyleSheet("""
                QPushButton { 
                    background-color: #404040; 
                    color: white; 
                    border: 1px solid #666; 
                    padding: 8px; 
                    margin: 2px;
                    font-size: 12px;
                }
                QPushButton:hover { 
                    background-color: #505050; 
                }
                QPushButton:pressed { 
                    background-color: #303030; 
                }
            """)
            self.midpoint_btn.clicked.connect(self.toggle_midpoint_mode)
            panel_layout.addWidget(self.midpoint_btn)

            # 8. 모드 상태 표시
            self.mode_label = QtWidgets.QLabel("모드: 일반")
            self.mode_label.setStyleSheet("color: #aaa; font-size: 11px; margin: 5px;")
            panel_layout.addWidget(self.mode_label)

            # ─── 거리 측정 도구 섹션 시작 ───
            # 구분선
            separator3 = QtWidgets.QFrame()
            separator3.setFrameShape(QtWidgets.QFrame.HLine)
            separator3.setStyleSheet("color: #555;")
            panel_layout.addWidget(separator3)

            # 거리 측정 도구 라벨
            distance_label = QtWidgets.QLabel("거리 측정 도구")
            distance_label.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 5px; color: white;")
            panel_layout.addWidget(distance_label)

            # 거리 측정 버튼
            self.distance_btn = QtWidgets.QPushButton("거리 측정")
            self.distance_btn.setStyleSheet("""
                QPushButton { 
                    background-color: #2196F3; 
                    color: white; 
                    border: 1px solid #1976D2; 
                    padding: 8px; 
                    margin: 2px;
                    font-size: 12px;
                }
                QPushButton:hover { 
                    background-color: #1976D2; 
                }
            """)
            self.distance_btn.clicked.connect(self.toggle_distance_mode)
            panel_layout.addWidget(self.distance_btn)

            # 거리 결과 라벨
            self.distance_result_label = QtWidgets.QLabel("측정 대기 중...")
            self.distance_result_label.setStyleSheet("color: #888; font-size: 11px; margin: 5px;")
            self.distance_result_label.setWordWrap(True)
            panel_layout.addWidget(self.distance_result_label)

            # 거리 입력
            distance_input_layout = QtWidgets.QHBoxLayout()
            distance_input_label = QtWidgets.QLabel("거리:")
            distance_input_label.setStyleSheet("color: white; font-size: 11px;")
            self.distance_input = QtWidgets.QLineEdit()
            self.distance_input.setPlaceholderText("5.2")
            self.distance_input.setStyleSheet("""
                QLineEdit {
                    background-color: #404040;
                    color: white;
                    border: 1px solid #666;
                    padding: 3px;
                }
            """)
            self.distance_input.setMaximumWidth(60)
            distance_input_layout.addWidget(distance_input_label)
            distance_input_layout.addWidget(self.distance_input)
            distance_input_layout.addWidget(QtWidgets.QLabel("m"))
            distance_input_layout.addStretch()
            panel_layout.addLayout(distance_input_layout)
            # ... (거리 측정 도구 코드 - 이미 있음) ...

            # 노드 삽입 버튼 추가
            self.insert_node_btn = QtWidgets.QPushButton("노드 삽입")
            # ... (버튼 스타일 코드) ...
            self.insert_node_btn.clicked.connect(self.insert_node_at_distance)
            self.insert_node_btn.setEnabled(False)
            panel_layout.addWidget(self.insert_node_btn)  # ← 한 번만!
            # ─── 거리 측정 도구 섹션 끝 ───

            # ✅ 여기에 패턴 인식 도구 추가!
            # 구분선
            separator4 = QtWidgets.QFrame()
            separator4.setFrameShape(QtWidgets.QFrame.HLine)
            separator4.setStyleSheet("color: #555;")
            panel_layout.addWidget(separator4)

            # 패턴 인식 도구 라벨
            pattern_label = QtWidgets.QLabel("패턴 인식 도구 🤖")
            pattern_label.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 5px; color: white;")
            panel_layout.addWidget(pattern_label)

            # ✅ 사각형 패널 생성 버튼 (패턴 학습 대신)
            self.create_panel_btn = QtWidgets.QPushButton("사각형 패널 생성")
            self.create_panel_btn.setStyleSheet("""
                QPushButton { 
                    background-color: #9C27B0; 
                    color: white; 
                    border: 1px solid #7B1FA2; 
                    padding: 8px; 
                    margin: 2px;
                    font-size: 12px;
                }
                QPushButton:hover { 
                    background-color: #7B1FA2; 
                }
                QPushButton:pressed { 
                    background-color: #6A1B9A; 
                }
            """)
            self.create_panel_btn.clicked.connect(self.create_rectangular_panel)
            panel_layout.addWidget(self.create_panel_btn)

            # ===== 수정된 부분 시작: 패널 분할 수 입력 =====
            # 패널 분할 수 입력을 위한 가로 레이아웃
            panel_div_layout = QtWidgets.QHBoxLayout()
            panel_div_label = QtWidgets.QLabel("패널 분할:")
            panel_div_label.setStyleSheet("color: white; font-size: 11px;")

            # ⚠️ 기존 코드 문제점: 같은 SpinBox를 두 번 사용하려고 했음
            # self.panel_divisions를 두 개로 분리해야 함

            # ✅ 수정: X 방향 분할 수
            self.panel_divisions_x = QtWidgets.QSpinBox()
            self.panel_divisions_x.setMinimum(2)
            self.panel_divisions_x.setMaximum(20)
            self.panel_divisions_x.setValue(4)  # 기본값 4

            # ✅ 수정: Y 방향 분할 수
            self.panel_divisions_y = QtWidgets.QSpinBox()
            self.panel_divisions_y.setMinimum(2)
            self.panel_divisions_y.setMaximum(20)
            self.panel_divisions_y.setValue(4)  # 기본값 4

            # 두 SpinBox에 동일한 스타일 적용
            spinbox_style = """
                QSpinBox {
                    background-color: #404040;
                    color: white;
                    border: 1px solid #666;
                    padding: 3px;
                }
            """
            self.panel_divisions_x.setStyleSheet(spinbox_style)
            self.panel_divisions_y.setStyleSheet(spinbox_style)

            # 레이아웃에 위젯들 추가
            panel_div_layout.addWidget(panel_div_label)
            panel_div_layout.addWidget(self.panel_divisions_x)  # X 분할
            panel_div_layout.addWidget(QtWidgets.QLabel("x"))  # "x" 라벨
            panel_div_layout.addWidget(self.panel_divisions_y)  # Y 분할

            # ⚠️ 삭제된 줄: panel_div_layout.addWidget(self.panel_divisions)
            # 이미 위에서 X와 Y로 나누어 추가했으므로 중복 제거

            panel_layout.addLayout(panel_div_layout)
            # ===== 수정된 부분 끝 =====

            # 패널 정보 표시
            self.panel_info_label = QtWidgets.QLabel("4개의 노드를 선택하여 사각형을 정의하세요")
            self.panel_info_label.setStyleSheet("color: #888; font-size: 11px; margin: 5px;")
            self.panel_info_label.setWordWrap(True)
            panel_layout.addWidget(self.panel_info_label)  # ✅ 이제 정상적으로 추가됨!

            # ─── PANER 연결 도구 섹션 시작 ───
            separator5 = QtWidgets.QFrame()
            separator5.setFrameShape(QtWidgets.QFrame.HLine)
            separator5.setStyleSheet("color: #555;")
            panel_layout.addWidget(separator5)

            # PANER 연결 도구 라벨
            paner_label = QtWidgets.QLabel("PANER 연결 도구")
            paner_label.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 5px; color: white;")
            panel_layout.addWidget(paner_label)

            # 십자 연결 버튼
            self.cross_connect_btn = QtWidgets.QPushButton("십자 연결 (PANER)")
            self.cross_connect_btn.setStyleSheet("""
                QPushButton { 
                    background-color: #4CAF50; 
                    color: white; 
                    border: 1px solid #45a049; 
                    padding: 8px; 
                    margin: 2px;
                    font-size: 12px;
                }
                QPushButton:hover { 
                    background-color: #45a049; 
                }
            """)
            self.cross_connect_btn.clicked.connect(self.create_cross_connection)
            panel_layout.addWidget(self.cross_connect_btn)

            # 교차점 정보
            self.intersection_info_label = QtWidgets.QLabel("4개 노드를 선택하세요")
            self.intersection_info_label.setStyleSheet("color: #888; font-size: 11px; margin: 5px;")
            self.intersection_info_label.setWordWrap(True)
            panel_layout.addWidget(self.intersection_info_label)
                        
            # 9. 스페이서
            panel_layout.addStretch()
            
            # ✨ 우측 3D 뷰 ✨
            self.gl_widget = OrthoViewWidget()
            # ✅ 초기 카메라 거리를 충분히 크게 설정
            self.gl_widget.setCameraPosition(distance=1000, elevation=30, azimuth=45)
            self.gl_widget.setBackgroundColor('k')
            
            # ✨ 스플리터에 좌측 패널과 3D 뷰 추가 ✨
            splitter.addWidget(scroll_area) 
            splitter.addWidget(self.gl_widget)
            
            # ✨ 초기 크기 비율 설정 (좌측:우측 = 1:4) ✨
            splitter.setSizes([300, 1100])
            
            # ✨ 메인 레이아웃에 스플리터 추가 (툴바 아래) ✨
            main_layout.addWidget(splitter)
            
            # ✨ 마우스 이벤트 오버라이드를 여기로 이동! ✨
            #self.gl_widget.mousePressEvent = self.mouse_press_event
            #self.gl_widget.mouseReleaseEvent = self.mouse_release_event
            #self.gl_widget.mouseMoveEvent = self.mouse_move_event
    
            # ✨ 새로운 방식: 이벤트 필터
            self.gl_widget.installEventFilter(self)
            
            # 렌더링 품질 조정 (성능 향상)
            import pyqtgraph as pg
            pg.setConfigOption('antialias', False)
            pg.setConfigOption('useOpenGL', True)
                      
            # 축 추가
            self.add_axes()
            
            # 그리드 추가
            self.add_grid()
            
            # 시각화 요소들
            self.scatter_plot = None
            self.line_plots = []
            self.text_items = []
            
            # 선택 관련
            self.selection_mode = False
            self.is_dragging = False
            self.drag_start = None
            self.drag_rect = None
            # ✅ 라인 선택 관련 추가
            self.selected_lines = set()  # 선택된 라인들
            
            # ✅ 줌 모드 관련 초기화 추가
            self.zoom_mode = False
            self.zoom_dragging = False
            self.zoom_start = None
            self.zoom_rect_item = None
            
            # 상태바
            self.status_bar = self.statusBar()
            self.update_status()
            
            # 노드 번호 표시 여부
            self.show_node_numbers = False
            
            # 거리 측정 모드 관련 변수 초기화
            self.distance_mode = False
            self.first_node = None
            self.second_node = None
            self.temp_line = None
            # 패턴 인식 관련 변수
            self.learned_pattern = None
            
             # 카메라 이동 모드 관련
            self.pan_mode = False
            self.pan_start = None
            
            self.midpoint_mode = False
            
            # ✅ 키보드 포커스 설정 (맨 끝에 추가)
            self.setFocusPolicy(QtCore.Qt.StrongFocus)
            self.setFocus()
        
        def create_menubar(self):
            """메뉴바 생성"""
            menubar = self.menuBar()
            
            # 파일 메뉴
            file_menu = menubar.addMenu('파일')
            file_menu.addAction('CSV 불러오기', self.load_csv)
            file_menu.addAction('Elements 불러오기', self.load_elements_csv)
            file_menu.addSeparator()
            file_menu.addAction('CSV 저장', self.save_csv)
            file_menu.addSeparator()
            file_menu.addAction('종료', self.close)
            # ✅ 런처 열기 추가
            file_menu.addAction('🚀 런처 열기', self.open_launcher)
            file_menu.addSeparator()
            
            file_menu.addAction('종료', self.close)
            
            # 편집 메뉴
            edit_menu = menubar.addMenu('편집')
            edit_menu.addAction('모두 선택', self.select_all)
            edit_menu.addAction('선택 해제', self.clear_selection)
            edit_menu.addSeparator()
            edit_menu.addAction('선택 삭제', self.delete_selected)
            edit_menu.addAction('실행 취소', self.undo)
            
            # 보기 메뉴
            view_menu = menubar.addMenu('보기')
            
            # 카메라 뷰 서브메뉴
            camera_menu = view_menu.addMenu('카메라 뷰')
            camera_menu.addAction('Top', lambda: self.set_view('top'))
            camera_menu.addAction('Front', lambda: self.set_view('front'))
            camera_menu.addAction('Left', lambda: self.set_view('left'))
            camera_menu.addAction('Right', lambda: self.set_view('right'))
            
            view_menu.addSeparator()
            view_menu.addAction('뷰 초기화', self.reset_view)
            view_menu.addAction('전체 맞춤', self.fit_to_view) if hasattr(self, 'fit_to_view') else None
            
            # 패널 메뉴 추가
            panel_menu = menubar.addMenu('패널')
            panel_menu.addAction('📐 패널 맵핑 시작', self.start_panel_mapping)
            panel_menu.addAction('🔲 사각형 패널 정의', self.define_rect_panel)
            panel_menu.addAction('📊 맵핑 상태 보기', self.show_mapping_status)
            panel_menu.addSeparator()
            panel_menu.addAction('💾 맵핑 데이터 저장', self.save_mapping_data)
            panel_menu.addAction('🚀 패널 편집기로 전송', self.send_to_panel_editor)
            
            view_menu.addSeparator()
            
            # 표시 옵션
            self.show_numbers_action = view_menu.addAction('노드 번호 표시')
            self.show_numbers_action.setCheckable(True)
            self.show_numbers_action.triggered.connect(self.toggle_node_numbers)
            
            # 레이어 메뉴
            layer_menu = menubar.addMenu('레이어')
            
            # 레이어 체크박스들을 액션으로
            self.beam_action = layer_menu.addAction('BEAM (빨간색)')
            self.beam_action.setCheckable(True)
            self.beam_action.setChecked(True)
            self.beam_action.triggered.connect(lambda checked: self.toggle_beam_layer(
                QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked))
            
            self.truss_action = layer_menu.addAction('TRUSS (녹색)')
            self.truss_action.setCheckable(True)
            self.truss_action.setChecked(True)
            self.truss_action.triggered.connect(lambda checked: self.toggle_truss_layer(
                QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked))
            
            # 그룹 메뉴
            group_menu = menubar.addMenu('그룹')
            
            self.group_actions = []
            for i in range(5):  # Group 1-5
                action = group_menu.addAction(f'Group {i+1}')
                action.setCheckable(True)
                action.setChecked(True)
                action.triggered.connect(lambda checked, group=i: self.toggle_group(group, checked))
                self.group_actions.append(action)
            
            group_menu.addSeparator()
            group_menu.addAction('모든 그룹 표시', self.all_groups_on)
            group_menu.addAction('모든 그룹 숨김', self.all_groups_off)
            
            # 도구 메뉴
            tools_menu = menubar.addMenu('도구')
            tools_menu.addAction('거리 측정', self.toggle_distance_mode) if hasattr(self, 'toggle_distance_mode') else None
            tools_menu.addAction('중점 노드 생성', self.toggle_midpoint_mode) if hasattr(self, 'toggle_midpoint_mode') else None
            tools_menu.addAction('🏗️ 패널 편집기 열기', self.open_panel_editor)
            tools_menu.addSeparator()
            tools_menu.addAction('패턴 학습', self.learn_pattern) if hasattr(self, 'learn_pattern') else None
            
        def open_launcher(self):
            """런처를 별도 창으로 열기 (현재 프로그램 유지)"""
            import subprocess
            
            try:
                # 런처를 새 프로세스로 실행 (현재 프로그램은 유지)
                subprocess.Popen([sys.executable, "launcher.py"])
                self.status_bar.showMessage("런처가 새 창으로 열렸습니다", 3000)
            except FileNotFoundError:
                QtWidgets.QMessageBox.warning(
                    self, '오류',
                    'launcher.py 파일을 찾을 수 없습니다.'
            )
            
        def create_toolbar(self):
            """간소화된 툴바 - 자주 쓰는 기능만"""
            toolbar = self.addToolBar('Main')
            
            # 선택 모드 토글
            self.selection_mode_action = toolbar.addAction('🖱️ Selection Mode')
            self.selection_mode_action.setCheckable(True)
            self.selection_mode_action.triggered.connect(self.toggle_selection_mode)
            
            toolbar.addSeparator()
            
            # 빠른 파일 액세스
            toolbar.addAction('📁 Load', self.load_csv)
            toolbar.addAction('💾 Save', self.save_csv)
            
            toolbar.addSeparator()
            
            # 라인 연결 도구
            toolbar.addWidget(QtWidgets.QLabel("  라인: "))
            self.line_type_combo = QtWidgets.QComboBox()
            self.line_type_combo.addItem("BEAM", LineType.MATERIAL)
            self.line_type_combo.addItem("TRUSS", LineType.TRUSS)
            self.line_type_combo.addItem("PANER", LineType.PANER)
            self.line_type_combo.setMaximumWidth(80)
            toolbar.addWidget(self.line_type_combo)
            
            connect_action = toolbar.addAction('🔗')
            connect_action.setToolTip('선택한 노드 연결')
            connect_action.triggered.connect(
                lambda: self.connect_nodes(self.line_type_combo.currentData())
            )
            
            toolbar.addSeparator()
            
            # ✅ 외장 그룹 설정 버튼 추가
            exterior_group_action = toolbar.addAction('🏢 외장 그룹')
            exterior_group_action.setToolTip('선택한 노드/라인을 외장 그룹(Group 5)으로 설정')
            exterior_group_action.triggered.connect(self.set_selected_as_exterior_group)
            
            toolbar.addSeparator()
            
            # ✅ 좌표 선택 도구 추가
            toolbar.addWidget(QtWidgets.QLabel("  좌표 고정: "))
            
            # X 체크박스
            self.x_coord_checkbox = QtWidgets.QCheckBox("X")
            self.x_coord_checkbox.setStyleSheet("""
                QCheckBox {
                    color: white;
                    padding: 0 5px;
                }
                QCheckBox::indicator {
                    width: 15px;
                    height: 15px;
                }
            """)
            toolbar.addWidget(self.x_coord_checkbox)
            
            # Y 체크박스
            self.y_coord_checkbox = QtWidgets.QCheckBox("Y")
            self.y_coord_checkbox.setStyleSheet("""
                QCheckBox {
                    color: white;
                    padding: 0 5px;
                }
                QCheckBox::indicator {
                    width: 15px;
                    height: 15px;
                }
            """)
            toolbar.addWidget(self.y_coord_checkbox)
            
            # Z 체크박스
            self.z_coord_checkbox = QtWidgets.QCheckBox("Z")
            self.z_coord_checkbox.setStyleSheet("""
                QCheckBox {
                    color: white;
                    padding: 0 5px;
                }
                QCheckBox::indicator {
                    width: 15px;
                    height: 15px;
                }
            """)
            toolbar.addWidget(self.z_coord_checkbox)
            
            # 좌표 선택 버튼
            coord_select_action = toolbar.addAction('📍 좌표 선택')
            coord_select_action.setToolTip('체크된 좌표가 같은 노드/라인 선택')
            coord_select_action.triggered.connect(self.select_by_coordinates)
            
            toolbar.addSeparator()
            
            # 실행 취소
            toolbar.addAction('↩️', self.undo).setToolTip('실행 취소')
            
            # ✅ 줌 모드 토글 추가
            self.zoom_mode_action = toolbar.addAction('🔍 Zoom Mode')
            self.zoom_mode_action.setCheckable(True)
            self.zoom_mode_action.triggered.connect(self.toggle_zoom_mode)
            self.zoom_rect_item = None  # 줌 영역 표시용 사각형
            
            toolbar.addSeparator()
 
            # ✨ 여기에 그룹 관련 메서드들 추가 ✨
            
        def toggle_zoom_mode(self):
            """줌 모드 토글"""
            self.zoom_mode = self.zoom_mode_action.isChecked()
            
            # 다른 모드 해제
            if self.zoom_mode:
                self.selection_mode = False
                self.selection_mode_action.setChecked(False)
                self.setCursor(QtCore.Qt.CrossCursor)
                self.status_bar.showMessage("🔍 줌 모드 - 드래그하여 영역 확대", 2000)
            else:
                self.setCursor(QtCore.Qt.ArrowCursor)
                self.status_bar.showMessage("줌 모드 해제", 2000)    
            
        def toggle_group(self, group_id, visible):
            """특정 그룹 표시/숨김"""
            print(f"🔄 Group {group_id + 1}: {'ON' if visible else 'OFF'}")
            
            # 디버깅: 실제로 변경되는 노드 수 확인
            changed_nodes = 0
            changed_lines = 0
            
            # 노드 표시/숨김
            if hasattr(self.editor.scene, 'nodes'):
                # Group 5일 때 디버깅 추가
                if group_id == 4:  # Group 5
                    print(f"🔍 Group 5 토글 - 전체 노드 검사 중...")
                    
                for node in self.editor.scene.nodes:
                    if hasattr(node, 'group_id') and node.group_id == group_id:
                        # 디버깅: 처음 몇 개 노드만 출력
                        if group_id == 4 and changed_nodes < 5:
                            print(f"   노드 {node.number}: group_id={node.group_id}, is_visible {getattr(node, 'is_visible', True)} → {visible}")
                        
                        node.is_visible = visible
                        changed_nodes += 1
            
            # 라인 표시/숨김
            if hasattr(self.editor.scene, 'lines'):
                for line in self.editor.scene.lines:
                    if hasattr(line, 'group_ids') and group_id in line.group_ids:
                        line.is_visible = visible
                        changed_lines += 1
            
            print(f"   → 변경된 노드: {changed_nodes}개, 라인: {changed_lines}개")
            
            self.update_scene()
            self.update_status()

        def all_groups_on(self):
            """모든 그룹 표시"""
            print("🔛 모든 그룹 ON")
            
            # ✅ group_buttons 대신 group_actions 사용
            if hasattr(self, 'group_actions'):
                for i, action in enumerate(self.group_actions):
                    action.setChecked(True)
                    self.toggle_group(i, True)
            else:
                # group_actions가 없으면 직접 처리
                for i in range(5):  # Group 1-5
                    self.toggle_group(i, True)

        def all_groups_off(self):
            """모든 그룹 숨김"""
            print("⬜ 모든 그룹 OFF")
            
            # ✅ group_buttons 대신 group_actions 사용
            if hasattr(self, 'group_actions'):
                for i, action in enumerate(self.group_actions):
                    action.setChecked(False)
                    self.toggle_group(i, False)
            else:
                # group_actions가 없으면 직접 처리
                for i in range(5):  # Group 1-5
                    self.toggle_group(i, False)
                    
        def toggle_beam_layer(self, state):
            """BEAM 레이어 토글"""
            visible = state == QtCore.Qt.Checked
            print(f"🔴 BEAM 레이어: {'ON' if visible else 'OFF'}")
            
            # BEAM 타입 라인들 표시/숨김
            if hasattr(self.editor.scene, 'lines'):
                for line in self.editor.scene.lines:
                    if hasattr(line, 'line_type') and line.line_type == LineType.MATERIAL:
                        line.is_visible = visible
            
            self.update_scene()

        def toggle_truss_layer(self, state):
            """TRUSS 레이어 토글"""
            visible = state == QtCore.Qt.Checked
            print(f"🟢 TRUSS 레이어: {'ON' if visible else 'OFF'}")
            
            # TRUSS 타입 라인들 표시/숨김
            if hasattr(self.editor.scene, 'lines'):
                for line in self.editor.scene.lines:
                    if hasattr(line, 'line_type') and line.line_type == LineType.TRUSS:
                        line.is_visible = visible
            
            self.update_scene()
 
            
        def add_axes(self):
            """좌표축 추가"""
            # X축 (빨간색)
            x_axis = gl.GLLinePlotItem(
                pos=np.array([[0, 0, 0], [5, 0, 0]]),
                color=(1, 0, 0, 1),
                width=3
            )
            self.gl_widget.addItem(x_axis)
            
            # Y축 (초록색)
            y_axis = gl.GLLinePlotItem(
                pos=np.array([[0, 0, 0], [0, 5, 0]]),
                color=(0, 1, 0, 1),
                width=3
            )
            self.gl_widget.addItem(y_axis)
            
            # Z축 (파란색)
            z_axis = gl.GLLinePlotItem(
                pos=np.array([[0, 0, 0], [0, 0, 5]]),
                color=(0, 0, 1, 1),
                width=3
            )
            self.gl_widget.addItem(z_axis)
            
        def add_grid(self):
            """그리드 추가"""
            # XY 평면 그리드
            grid = gl.GLGridItem()
            grid.scale(2, 2, 1)
            grid.setDepthValue(10)  # 다른 객체 뒤에 그리기
            self.gl_widget.addItem(grid)
            
        def toggle_selection_mode(self):
            """선택 모드 토글"""
            self.selection_mode = self.selection_mode_action.isChecked()
            
            if self.selection_mode:
                self.setCursor(QtCore.Qt.CrossCursor)
                self.status_bar.showMessage("선택 모드 - 드래그하여 노드 선택", 2000)
            else:
                self.setCursor(QtCore.Qt.ArrowCursor)
                self.status_bar.showMessage("카메라 모드 - 드래그하여 회전", 2000)
                
        def toggle_node_numbers(self, checked):
            """노드 번호 표시 토글"""
            self.show_node_numbers = checked
            self.update_scene()
            
            if checked:
                self.status_bar.showMessage("노드 번호 표시 ON", 2000)
            else:
                self.status_bar.showMessage("노드 번호 표시 OFF", 2000)
            
        def load_csv(self):
            """CSV 파일 로드"""
            filepath, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Load CSV", "data/", "CSV Files (*.csv)"
            )
            
            if filepath:
                if self.editor.load_csv(filepath):
                    self.update_scene()
                    self.update_status()
                    # ✅ 로드 후 자동으로 전체 뷰
                    self.fit_to_view()

        def fit_to_view(self):
            """모든 노드가 보이도록 카메라 조정"""
            if not self.editor.scene.nodes:
                return
                
            bounds_min, bounds_max = self.editor.scene.get_bounds()
            center = self.editor.scene.get_center()
            
            # 경계 상자의 대각선 길이
            import numpy as np
            diagonal = np.linalg.norm(bounds_max - bounds_min)
            
            # 적절한 거리 설정 (대각선의 2배 정도)
            distance = max(diagonal * 5, 100)
            
            from PyQt5.QtGui import QVector3D
            self.gl_widget.opts['center'] = QVector3D(center[0], center[1], center[2])
            self.gl_widget.opts['distance'] = distance
            self.gl_widget.setCameraPosition(distance=distance)
            self.gl_widget.update()
                    
         # ✨ 여기에 load_mgb 메서드 추가 ✨
        def load_mgb(self):
            """MGB 파일 로드"""
            print("🔍 load_mgb 메서드 호출됨")  # 디버그 메시지 추가
            
            filepath, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Load MIDAS MGB", "data/", 
                "All Files (*);;MIDAS Files (*.mgb *.mgt);;MGB Files (*.mgb);;MGT Files (*.mgt)"
            )
            
            print(f"📁 선택된 파일: {filepath}")  # 디버그 메시지 추가
            
            if filepath:
                print("📂 파일 로드 시도 중...")  # 디버그 메시지 추가
                
                if hasattr(self.editor, 'load_mgb'):
                    print("✅ editor.load_mgb 메서드 존재")
                    result = self.editor.load_mgb(filepath)
                    print(f"📊 로드 결과: {result}")
                else:
                    print("❌ editor.load_mgb 메서드가 없습니다!")
                    self.status_bar.showMessage("load_mgb 메서드가 구현되지 않았습니다", 3000)
                    return
                    
                if result:
                    self.update_scene()
                    self.update_status()
                    self.status_bar.showMessage(f"MIDAS 파일 로드 완료: {filepath}", 3000)
                    print("✅ 파일 로드 성공")
                else:
                    self.status_bar.showMessage("MIDAS 파일 로드 실패", 3000)
                    print("❌ 파일 로드 실패")
            else:
                print("🚫 파일이 선택되지 않음")
                    
        def save_csv(self):
            """CSV 파일 저장"""
            filepath, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save CSV", "output/", "CSV Files (*.csv)"
            )
            
            if filepath:
                if self.editor.save_csv(filepath, include_lines=True):
                    self.update_status()
                    
        def load_elements_csv(self):
            """Elements CSV 파일 로드"""
            print("🔍 load_elements_csv 메서드 호출됨")
            
            filepath, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Load Elements CSV", "data/", 
                "CSV Files (*.csv);;All Files (*)"
            )
            
            print(f"📁 선택된 Elements 파일: {filepath}")
            
            if filepath:
                print("📂 Elements 파일 로드 시도 중...")
                
                if hasattr(self.editor, 'load_elements_csv'):
                    print("✅ editor.load_elements_csv 메서드 존재")
                    result = self.editor.load_elements_csv(filepath)
                    print(f"📊 Elements 로드 결과: {result}")
                else:
                    print("❌ editor.load_elements_csv 메서드가 없습니다!")
                    self.status_bar.showMessage("load_elements_csv 메서드가 구현되지 않았습니다", 3000)
                    return
                    
                if result:
                    self.update_scene()
                    self.update_status()
                    self.status_bar.showMessage(f"Elements 파일 로드 완료: {filepath}", 3000)
                    print("✅ Elements 파일 로드 성공")
                else:
                    self.status_bar.showMessage("Elements 파일 로드 실패", 3000)
                    print("❌ Elements 파일 로드 실패")
            else:
                print("🚫 Elements 파일이 선택되지 않음")
                    
        def select_all(self):
            """모든 노드 선택"""
            self.editor.scene.select_all_nodes()
            self.update_scene()
            self.update_status()
            
        def clear_selection(self):
            """선택 해제"""
            self.editor.scene.clear_selection()
            self.update_scene()
            self.update_status()
            
        def connect_nodes(self, line_type):
            """선택된 노드 연결"""
            if self.editor.scene.connect_selected_nodes(line_type):
                self.update_scene()
                self.update_status()
                
        def delete_selected(self):
            """선택된 노드 삭제"""
            self.editor.scene.remove_selected_nodes()
            self.update_scene()
            self.update_status()
            
        def undo(self):
            """실행 취소"""
            if self.editor.scene.undo():
                self.update_scene()
                self.update_status()
                
        def reset_view(self):
            """뷰 리셋"""
            self.gl_widget.setCameraPosition(distance=100, elevation=30, azimuth=45)  # ✅ 50 → 100
            self.gl_widget.opts['distance'] = 1000  # ✅ 명시적으로 설정
            self.gl_widget.update()
            
        def update_scene(self):
            """씬 업데이트"""
            # 기존 아이템 제거 (안전하게)
            if self.scatter_plot is not None:
                try:
                    self.gl_widget.removeItem(self.scatter_plot)
                except ValueError:
                    # 이미 제거된 경우 무시
                    pass
                finally:
                    self.scatter_plot = None  # 항상 None으로 리셋
                    
            # 라인 제거도 안전하게
            for line in self.line_plots:
                try:
                    self.gl_widget.removeItem(line)
                except ValueError:
                    pass
            self.line_plots.clear()
            
            # 텍스트 제거도 안전하게
            for text in self.text_items:
                try:
                    self.gl_widget.removeItem(text)
                except ValueError:
                    pass
            self.text_items.clear()
            
            if self.editor.scene.nodes:
                # 노드 포지션과 색상
                positions = []
                colors = []
                
                for node in self.editor.scene.nodes:
                    # ✅ 보이지 않는 노드는 스킵
                    if not getattr(node, 'is_visible', True):
                        continue
                        
                    positions.append(node.position)
                    if node.is_selected:
                        colors.append([1, 1, 0, 1])  # 노란색
                    else:
                        colors.append([1, 1, 1, 1])  # 흰색
                
                # ✅ positions가 비어있으면 스킵
                if positions:
                    positions = np.array(positions)
                    colors = np.array(colors)
                    
                    # 스캐터 플롯 생성 (고정 크기)
                    self.scatter_plot = gl.GLScatterPlotItem(
                        pos=positions,
                        color=colors,
                        size=5,  # 고정 크기
                        pxMode=True  # 픽셀 모드 (화면 크기 고정)
                    )
                    self.gl_widget.addItem(self.scatter_plot)
                
                # 노드 번호 표시
                if self.show_node_numbers and len(self.editor.scene.nodes) < 1000:
                    for node in self.editor.scene.nodes:
                        # PyQtGraph는 3D 텍스트를 직접 지원하지 않으므로
                        # 2D 오버레이로 구현하거나 생략
                        pass
            
            # ✨ 보이는 라인만 렌더링 ✨
            if hasattr(self.editor.scene, 'lines'):
                print(f"🔍 총 라인 수: {len(self.editor.scene.lines)}")  # 디버그
                
                visible_lines = [line for line in self.editor.scene.lines 
                                if getattr(line, 'is_visible', True)]
                
                print(f"👁️  보이는 라인 수: {len(visible_lines)}")  # 디버그
                
                # 라인 그리기 부분
            for line in visible_lines:
                # 색상 결정
                if hasattr(line, 'is_selected') and line.is_selected:
                    # ✅ 선택된 라인은 더 밝게 또는 두껍게
                    if line.line_type == LineType.MATERIAL:
                        color = (1, 0.5, 0.5, 1)  # 밝은 빨강
                    elif line.line_type == LineType.TRUSS:
                        color = (0.5, 1, 0.5, 1)  # 밝은 녹색
                    elif line.line_type == LineType.PANER:
                        color = (1, 0.5, 1, 1)    # 밝은 핑크
                    width = 4  # 두껍게
                else:
                    # 일반 라인
                    if line.line_type == LineType.MATERIAL:
                        color = (1, 0, 0, 1)
                    elif line.line_type == LineType.TRUSS:
                        color = (0, 1, 0, 1)
                    elif line.line_type == LineType.PANER:
                        color = (1, 0, 1, 1)
                    width = 2
                
                line_item = gl.GLLinePlotItem(
                    pos=np.array([line.start_pos, line.end_pos]),
                    color=color,
                    width=width
                )
                self.gl_widget.addItem(line_item)
                self.line_plots.append(line_item)  
                
        def update_status(self):
            """상태바 업데이트"""
            info = self.editor.scene.get_selected_info()
            status = f"Nodes: {len(self.editor.scene.nodes)} | "
            status += f"Selected: {info['count']} | "
            status += f"Lines: {len(self.editor.scene.lines)}"
            self.status_bar.showMessage(status)
            
        def mouse_press_event(self, event):
            """마우스 클릭 이벤트"""
            
            # ✨ 중점 노드 생성 모드 체크 ✨
            print(f"🖱️ 마우스 클릭 감지: {event.button()}, 위치: {event.pos()}")
    
            # ✨ 중점 노드 생성 모드 체크 ✨
            if hasattr(self, 'midpoint_mode'):
                print(f"🔍 midpoint_mode 존재: {self.midpoint_mode}")
                
                if self.midpoint_mode:
                    print("🎯 중점 모드 활성화됨")
                    if event.button() == QtCore.Qt.LeftButton:
                        print("👆 좌클릭 감지")
                        self.handle_line_click(event)
                        event.accept()
                        return
                    else:
                        print(f"❌ 좌클릭이 아님: {event.button()}")
                else:
                    print("⚪ 중점 모드 비활성화됨")
            else:
                print("❌ midpoint_mode 속성 없음")
            
            if self.selection_mode and event.button() == QtCore.Qt.LeftButton:
                self.is_dragging = True
                self.drag_start = event.pos()
                event.accept()
            else:
                # 기본 카메라 컨트롤
                gl.GLViewWidget.mousePressEvent(self.gl_widget, event)
                
        def mouse_move_event(self, event):
            """마우스 이동 이벤트"""
            if self.selection_mode and self.is_dragging:
                # 선택 박스 그리기 (간단한 시각적 피드백)
                # PyQtGraph에서는 2D 오버레이가 복잡하므로 생략
                event.accept()
            else:
                gl.GLViewWidget.mouseMoveEvent(self.gl_widget, event)
                
        def mouse_release_event(self, event):
            """마우스 릴리즈 이벤트"""
            if self.selection_mode and self.is_dragging and event.button() == QtCore.Qt.LeftButton:
                self.is_dragging = False
                end_pos = event.pos()
                
                # 박스 선택 수행
                self.select_nodes_in_box(self.drag_start, end_pos, event.modifiers())
                event.accept()
            else:
                gl.GLViewWidget.mouseReleaseEvent(self.gl_widget, event)
                
        def select_nodes_in_box(self, start_pos, end_pos, modifiers):
            """박스 선택 (투영 + 깊이 검사)"""
            # 1) 픽셀 박스 경계
            min_x = min(start_pos.x(), end_pos.x())
            max_x = max(start_pos.x(), end_pos.x())
            min_y = min(start_pos.y(), end_pos.y())
            max_y = max(start_pos.y(), end_pos.y())

            # ✅ Ctrl 키가 눌려있지 않으면 기존 선택 해제
            if not (modifiers & QtCore.Qt.ControlModifier):
                self.editor.scene.clear_selection()
                # ✅ 라인 선택도 해제
                if hasattr(self, 'selected_lines'):
                    self.selected_lines.clear()
                print("🔄 기존 선택 해제")
            else:
                print("➕ Ctrl 키: 추가 선택 모드")

            # 2) MVP 계산
            mvp = self.gl_widget.projectionMatrix() * self.gl_widget.viewMatrix()
            width, height = self.gl_widget.width(), self.gl_widget.height()

            # 3) 노드별 검사
            selected_count = 0
            for node in self.editor.scene.nodes:
                # 보이지 않는 노드는 스킵
                if not getattr(node, 'is_visible', True):
                    continue
                    
                # 3-1) 클립 공간으로 투영
                clip = mvp.map(QVector4D(
                    node.position[0],
                    node.position[1],
                    node.position[2],
                    1.0
                ))
                if clip.w() == 0:
                    continue

                # 3-2) NDC 변환
                ndc_x = clip.x() / clip.w()
                ndc_y = clip.y() / clip.w()
                ndc_z = clip.z() / clip.w()

                # 4) 깊이 검사: 카메라 앞쪽만
                if ndc_z < -1 or ndc_z > 1:
                    continue

                # 5) 화면 좌표로 변환
                screen_x = (ndc_x + 1) * width / 2
                screen_y = (1 - ndc_y) * height / 2

                # 6) 박스 안에 들어오면 선택
                if min_x <= screen_x <= max_x and min_y <= screen_y <= max_y:
                    # ✅ Ctrl 모드에서 이미 선택된 노드는 선택 해제 (토글)
                    if (modifiers & QtCore.Qt.ControlModifier) and node.is_selected:
                        node.set_selected(False)
                        self.editor.scene.selected_nodes.discard(node)
                        print(f"➖ 노드 {node.number} 선택 해제")
                    else:
                        node.set_selected(True)
                        self.editor.scene.selected_nodes.add(node)
                        selected_count += 1

            # ✅ 라인 선택 추가
            selected_line_count = 0
            if hasattr(self.editor.scene, 'lines'):
                # selected_lines 초기화
                if not hasattr(self, 'selected_lines'):
                    self.selected_lines = set()
                    
                for line in self.editor.scene.lines:
                    # 보이지 않는 라인은 스킵
                    if not getattr(line, 'is_visible', True):
                        continue
                    
                    # 라인의 시작점과 끝점을 화면 좌표로 변환
                    start_screen = self.world_to_screen(line.start_pos, mvp, width, height)
                    end_screen = self.world_to_screen(line.end_pos, mvp, width, height)
                    
                    if start_screen is None or end_screen is None:
                        continue
                    
                    # 라인이 선택 박스 안에 있는지 확인 (양 끝점이 모두 박스 안에 있을 때)
                    start_in_box = (min_x <= start_screen[0] <= max_x and min_y <= start_screen[1] <= max_y)
                    end_in_box = (min_x <= end_screen[0] <= max_x and min_y <= end_screen[1] <= max_y)
                    
                    if start_in_box and end_in_box:
                        if not hasattr(line, 'is_selected'):
                            line.is_selected = False
                        
                        # Ctrl 모드에서 토글
                        if (modifiers & QtCore.Qt.ControlModifier) and line.is_selected:
                            line.is_selected = False
                            self.selected_lines.discard(line)
                            print(f"➖ 라인 선택 해제")
                        else:
                            line.is_selected = True
                            self.selected_lines.add(line)
                            selected_line_count += 1

            # 7) 씬 갱신
            print(f"✅ {selected_count}개 노드, {selected_line_count}개 라인 선택")
            self.update_scene()
            self.update_status()
                
        def keyPressEvent(self, event):
            """키보드 이벤트"""
            # ✅ 스페이스바 처리 (반복 입력 방지)
            if event.key() == QtCore.Qt.Key_Space and not event.isAutoRepeat():
                # 스페이스바로 이동 모드 활성화
                self.pan_mode = True
                self.setCursor(QtCore.Qt.OpenHandCursor)
                self.status_bar.showMessage("카메라 이동 모드 (스페이스 + 드래그)", 2000)
                return
            
            # 기존 코드들
            elif event.key() == QtCore.Qt.Key_A:
                # 노드 추가
                center = self.editor.scene.get_center()
                self.editor.add_node_at_position(
                    center[0] + np.random.rand() * 2,
                    center[1] + np.random.rand() * 2,
                    center[2] + np.random.rand() * 2
                )
                self.update_scene()
                self.update_status()
            elif event.key() == QtCore.Qt.Key_Delete:
                # 선택 노드 삭제
                self.delete_selected()
            elif event.key() == QtCore.Qt.Key_Escape:
                # ✅ 이 부분을 다음과 같이 수정:
                # 거리 측정 모드 취소 체크
                if hasattr(self, 'distance_mode') and self.distance_mode:
                    self.toggle_distance_mode()  # 모드 해제
                # 중점 모드 취소 체크
                elif hasattr(self, 'midpoint_mode') and self.midpoint_mode:
                    self.toggle_midpoint_mode()  # 모드 해제
                else:
                    # 선택 해제
                    self.clear_selection()
                    
            # P키: 패널 정의
            elif event.key() == QtCore.Qt.Key_P and not event.isAutoRepeat():
                if len(self.editor.scene.selected_nodes) == 4:
                    self.define_rect_panel()
                else:
                    self.status_bar.showMessage("4개의 노드를 선택하세요", 2000)
                    
            # ✅ Cmd+Z (Mac) / Ctrl+Z (Windows/Linux) 추가
            elif event.key() == QtCore.Qt.Key_Z:
                if sys.platform == "darwin":  # macOS
                    if event.modifiers() == QtCore.Qt.MetaModifier:  # Cmd 키
                        self.undo()
                        
            # ✅ Cmd+Shift+Z (Mac) / Ctrl+Y (Windows/Linux) - Redo
            elif event.key() == QtCore.Qt.Key_Z and event.modifiers() == (QtCore.Qt.MetaModifier | QtCore.Qt.ShiftModifier):
                if hasattr(self, 'redo'):
                    self.redo()
            elif event.key() == QtCore.Qt.Key_Y and event.modifiers() == QtCore.Qt.ControlModifier:
                if hasattr(self, 'redo'):
                    self.redo()
                    
        # ✅ 여기에 새로운 메서드 추가!
        def keyReleaseEvent(self, event):
            """키보드 릴리즈 이벤트"""
            if event.key() == QtCore.Qt.Key_Space and not event.isAutoRepeat():
                # 스페이스바 떼면 이동 모드 해제
                self.pan_mode = False
                self.pan_start = None
                if self.selection_mode:
                    self.setCursor(QtCore.Qt.CrossCursor)
                else:
                    self.setCursor(QtCore.Qt.ArrowCursor)
                self.status_bar.showMessage("")
                
        def set_view(self, which):
            
            opts = self.gl_widget.opts
            
            
            # 2) 회전 중심(center) 설정
            center_np = self.editor.scene.get_center()         # numpy array (x,y,z)
            from PyQt5.QtGui import QVector3D                 # 필요 시 파일 상단에 한 번만 import해도 OK
            center = QVector3D(center_np[0], center_np[1], center_np[2])
            opts['center'] = center
            # ← 수정: setCameraPosition(center=…) 대신 opts로 center 지정

            # 3) ✅ 구조물 전체가 보이도록 거리 자동 계산
            if self.editor.scene.nodes:
                bounds_min, bounds_max = self.editor.scene.get_bounds()
                
                # 경계 상자의 대각선 길이 계산
                import numpy as np
                diagonal = np.linalg.norm(bounds_max - bounds_min)
                
                # 뷰에 따른 거리 조정 (배율을 훨씬 더 크게)
                view_multipliers = {
                    'top': 60.0,     # 3.0 → 10.0으로 대폭 증가
                    'front': 50.5,   # 3.5 → 12.0으로 대폭 증가
                    'left': 50.5,    # 3.5 → 12.0으로 대폭 증가
                    'right': 50.5,   # 3.5 → 12.0으로 대폭 증가
                }
                
                multiplier = view_multipliers.get(which, 10.0)
                dist = max(diagonal * multiplier, 1000)  # 최소 거리도 1000으로
                
                # 디버그 출력
                print(f"📐 View: {which}")
                print(f"📏 Diagonal: {diagonal:.2f}")
                print(f"📍 Distance: {dist:.2f}")
            else:
                # 노드가 없으면 기본값 사용
                dist = 2000

            self.gl_widget.opts['distance'] = dist

            # 4) CAD 뷰 각도 정의 (elevation, azimuth)
            angles = {
                'top':   (89.99,   0),
                'front': (0,   -90),
                'left':  (0,   180),
                'right': (0,     0),
            }
            elev, azim = angles[which]
            # ← 수정: 각 뷰별로 elev/azim 값 선택

            self.gl_widget.opts['elevation'] = elev
            self.gl_widget.opts['azimuth']   = azim
            # ← 수정: setCameraPosition(elevation=…, azimuth=…) 대신 opts로 지정

            # 5) 뷰어 갱신
            self.gl_widget.update()
            # ← 수정: setCameraPosition 호출 제거 후 update() 로 렌더링 갱신     
            
        def toggle_midpoint_mode(self):
            """중점 노드 생성 모드 토글"""
            # 모드 상태 관리를 위한 변수 (처음 한 번만 초기화)
            if not hasattr(self, 'midpoint_mode'):
                self.midpoint_mode = False
            
            # 모드 전환
            self.midpoint_mode = not self.midpoint_mode
            
            if self.midpoint_mode:
                print("🎯 중점 노드 생성 모드 활성화")
                self.mode_label.setText("모드: 중점 노드 생성 (라인을 클릭하세요)")
                self.mode_label.setStyleSheet("color: #4CAF50; font-size: 11px; margin: 5px;")
                self.midpoint_btn.setText("모드 해제")
            else:
                print("⚪ 일반 모드로 복귀")
                self.mode_label.setText("모드: 일반")
                self.mode_label.setStyleSheet("color: #aaa; font-size: 11px; margin: 5px;")
                self.midpoint_btn.setText("중점 노드 생성")
                
        def eventFilter(self, obj, event):
            """이벤트 필터 - 마우스 클릭 감지"""
            if obj == self.gl_widget:
                # ✅ 속성이 없으면 초기화
                if not hasattr(self, 'zoom_mode'):
                    self.zoom_mode = False
                    self.zoom_dragging = False
                    self.zoom_start = None
                # 마우스 버튼 누르기
                if event.type() == QtCore.QEvent.MouseButtonPress:
                    print(f"🖱️ 이벤트 필터로 마우스 클릭 감지!")
                    
                    # ✅ 줌 모드 체크 (가장 먼저)
                    if self.zoom_mode and event.button() == QtCore.Qt.LeftButton:
                        self.zoom_dragging = True
                        self.zoom_start = event.pos()
                        self.start_zoom_rect(event.pos())
                        return True
                    
                    # ✅ 스페이스 + 마우스 = 카메라 이동
                    if self.pan_mode and event.button() == QtCore.Qt.LeftButton:
                        self.pan_start = event.pos()
                        self.setCursor(QtCore.Qt.ClosedHandCursor)
                        return True
                    
                    # 거리 측정 모드 체크
                    if hasattr(self, 'distance_mode') and self.distance_mode:
                        if event.button() == QtCore.Qt.LeftButton:
                            self.handle_distance_mode_click(event)
                            return True
                    
                    # 중점 모드 체크
                    if hasattr(self, 'midpoint_mode') and self.midpoint_mode:
                        if event.button() == QtCore.Qt.LeftButton:
                            self.handle_line_click(event)
                            return True
                    
                    # 선택 모드 체크
                    if self.selection_mode and event.button() == QtCore.Qt.LeftButton:
                        self.is_dragging = True
                        self.drag_start = event.pos()
                        return True
                
                # 마우스 이동
                elif event.type() == QtCore.QEvent.MouseMove:
                    # ✅ 줌 드래그 처리
                    if self.zoom_mode and self.zoom_dragging:
                        self.update_zoom_rect(event.pos())
                        return True
                    
                    # ✅ 카메라 이동 처리
                    if self.pan_mode and self.pan_start is not None:
                        delta = event.pos() - self.pan_start
                        self.pan_start = event.pos()
                        
                        # 현재 카메라 정보
                        center = self.gl_widget.opts['center']
                        distance = self.gl_widget.opts.get('distance', 1000)
                        
                        # 카메라 이동
                        pan_speed = distance * 0.00005
                        dx = -delta.x() * pan_speed
                        dy = -delta.y() * pan_speed
                        
                        from PyQt5.QtGui import QVector3D
                        new_center = QVector3D(
                            center.x() + dx,
                            center.y() + dy,
                            center.z()
                        )
                        
                        self.gl_widget.opts['center'] = new_center
                        self.gl_widget.update()
                        
                        return True
                    
                    # 선택 모드 드래그
                    if self.selection_mode and self.is_dragging:
                        return True
                
                # 마우스 버튼 떼기
                elif event.type() == QtCore.QEvent.MouseButtonRelease:
                    # ✅ 줌 드래그 종료
                    if self.zoom_mode and self.zoom_dragging and event.button() == QtCore.Qt.LeftButton:
                        self.zoom_dragging = False
                        self.finish_zoom(event.pos())
                        return True
                    
                    # ✅ 카메라 이동 모드 처리
                    if self.pan_mode and event.button() == QtCore.Qt.LeftButton:
                        self.pan_start = None
                        self.setCursor(QtCore.Qt.OpenHandCursor)
                        return True
                    
                    # 선택 모드 처리
                    if self.selection_mode and self.is_dragging and event.button() == QtCore.Qt.LeftButton:
                        self.is_dragging = False
                        end_pos = event.pos()
                        self.select_nodes_in_box(self.drag_start, end_pos, event.modifiers())
                        return True
                
                # 키보드 이벤트도 여기서 처리
                if event.type() == QtCore.QEvent.KeyPress:
                    if event.key() == QtCore.Qt.Key_Space and not event.isAutoRepeat():
                        self.pan_mode = True
                        self.setCursor(QtCore.Qt.OpenHandCursor)
                        self.status_bar.showMessage("카메라 이동 모드 (스페이스 + 드래그)", 2000)
                        return True
                    elif event.key() == QtCore.Qt.Key_Control:
                        self.status_bar.showMessage("Ctrl: 추가 선택 모드", 1000)
                        
                elif event.type() == QtCore.QEvent.KeyRelease:
                    if event.key() == QtCore.Qt.Key_Space and not event.isAutoRepeat():
                        self.pan_mode = False
                        self.pan_start = None
                        if self.selection_mode:
                            self.setCursor(QtCore.Qt.CrossCursor)
                        else:
                            self.setCursor(QtCore.Qt.ArrowCursor)
                        self.status_bar.showMessage("")
                        return True
                    elif event.key() == QtCore.Qt.Key_Control:
                        self.status_bar.showMessage("")
            
            return super().eventFilter(obj, event)
                
                
        def handle_line_click(self, event):
            """실제 클릭한 라인 감지 및 중점 노드 생성"""
            try:
                print(f"🎯 라인 클릭 처리 시작 - 위치: {event.pos()}")
                
                if not hasattr(self.editor.scene, 'lines') or len(self.editor.scene.lines) == 0:
                    print("❌ 라인이 없습니다")
                    return
                
                # 마우스 클릭 위치를 3D 공간으로 변환
                mouse_pos = event.pos()
                clicked_line = self.find_closest_line_to_click(mouse_pos)
                
                if clicked_line is None:
                    print("❌ 클릭 위치 근처에 라인을 찾을 수 없습니다")
                    return
                
                print(f"🎯 클릭된 라인 발견!")
                
                # 중점 계산
                start_pos = clicked_line.start_pos
                end_pos = clicked_line.end_pos
                
                mid_x = (start_pos[0] + end_pos[0]) / 2
                mid_y = (start_pos[1] + end_pos[1]) / 2
                mid_z = (start_pos[2] + end_pos[2]) / 2
                
                print(f"📍 중점 좌표: ({mid_x:.2f}, {mid_y:.2f}, {mid_z:.2f})")
                
                # 새 노드 번호 생성
                new_number = self.editor.node_manager.get_next_number()
                print(f"🔢 새 노드 번호: {new_number}")
                
                # 새 노드 생성
                from src.data_structures import DataPoint, Node3D
                new_datapoint = DataPoint(new_number, mid_x, mid_y, mid_z)
                new_node = Node3D(new_datapoint)
                
                # 씬에 추가
                self.editor.scene.nodes.append(new_node)
                
                print(f"✅ 중점 노드 생성 완료: 노드 {new_number}")
                
                # 3D 뷰 업데이트
                self.update_scene()
                self.update_status()
                
                # 모드 해제
                self.toggle_midpoint_mode()
                
            except Exception as e:
                print(f"❌ 라인 클릭 처리 오류: {e}")

        def find_closest_line_to_click(self, mouse_pos):
            """마우스 클릭 위치에서 가장 가까운 라인 찾기"""
            try:
                # 3D 변환 매트릭스 가져오기
                mvp = self.gl_widget.projectionMatrix() * self.gl_widget.viewMatrix()
                width, height = self.gl_widget.width(), self.gl_widget.height()
                
                mouse_x = mouse_pos.x()
                mouse_y = mouse_pos.y()
                
                closest_line = None
                min_distance = float('inf')
                detection_radius = 15  # 픽셀 단위 감지 반경
                
                print(f"🔍 {len(self.editor.scene.lines)}개 라인 중에서 검색...")
                
                for line in self.editor.scene.lines:
                    # 라인이 보이지 않으면 스킵
                    if not getattr(line, 'is_visible', True):
                        continue
                        
                    # 라인의 시작점과 끝점을 화면 좌표로 변환
                    start_screen = self.world_to_screen(line.start_pos, mvp, width, height)
                    end_screen = self.world_to_screen(line.end_pos, mvp, width, height)
                    
                    if start_screen is None or end_screen is None:
                        continue
                    
                    # 마우스 위치와 라인 사이의 거리 계산
                    distance = self.point_to_line_distance_2d(
                        (mouse_x, mouse_y), start_screen, end_screen
                    )
                    
                    # 감지 반경 내에서 가장 가까운 라인 찾기
                    if distance < detection_radius and distance < min_distance:
                        min_distance = distance
                        closest_line = line
                
                if closest_line:
                    print(f"✅ 가장 가까운 라인 발견 (거리: {min_distance:.1f}px)")
                else:
                    print(f"❌ {detection_radius}px 반경 내에 라인 없음")
                    
                return closest_line
                
            except Exception as e:
                print(f"❌ 라인 검색 오류: {e}")
                return None

        def world_to_screen(self, world_pos, mvp, width, height):
            """3D 월드 좌표를 2D 화면 좌표로 변환"""
            try:
                from PyQt5.QtGui import QVector4D
                
                # 3D 좌표를 동차 좌표로 변환
                clip = mvp.map(QVector4D(world_pos[0], world_pos[1], world_pos[2], 1.0))
                
                if clip.w() == 0:
                    return None
                    
                # NDC 변환
                ndc_x = clip.x() / clip.w()
                ndc_y = clip.y() / clip.w()
                ndc_z = clip.z() / clip.w()
                
                # 화면 뒤쪽이면 무시
                if ndc_z < -1 or ndc_z > 1:
                    return None
                    
                # 화면 좌표로 변환
                screen_x = (ndc_x + 1) * width / 2
                screen_y = (1 - ndc_y) * height / 2
                
                return (screen_x, screen_y)
                
            except:
                return None

        def point_to_line_distance_2d(self, point, line_start, line_end):
            """2D에서 점과 선분 사이의 최단 거리 계산"""
            try:
                px, py = point
                x1, y1 = line_start
                x2, y2 = line_end
                
                # 선분의 길이 제곱
                line_length_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
                
                if line_length_sq == 0:
                    # 점과 점 사이의 거리
                    return ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5
                
                # 선분 위의 가장 가까운 점 찾기
                t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_length_sq))
                
                # 가장 가까운 점의 좌표
                closest_x = x1 + t * (x2 - x1)
                closest_y = y1 + t * (y2 - y1)
                
                # 거리 계산
                distance = ((px - closest_x) ** 2 + (py - closest_y) ** 2) ** 0.5
                return distance
                
            except:
                return float('inf') 
            
            # ✅ 여기에 거리 측정 관련 메서드들을 추가하세요!

        def toggle_distance_mode(self):
            """거리 측정 모드 토글"""
            if not hasattr(self, 'distance_mode'):
                self.distance_mode = False
                self.first_node = None
                self.second_node = None
                self.temp_line = None
            
            self.distance_mode = not self.distance_mode
            
            if self.distance_mode:
                print("📏 거리 측정 모드 활성화")
                self.mode_label.setText("모드: 거리 측정 (첫 번째 노드를 클릭하세요)")
                self.mode_label.setStyleSheet("color: #2196F3; font-size: 11px; margin: 5px;")
                self.distance_btn.setText("모드 해제")
                self.first_node = None
                self.second_node = None
                # 거리 결과 초기화
                self.distance_result_label.setText("측정 대기 중...")
                self.distance_result_label.setStyleSheet("color: #888; font-size: 11px; margin: 5px;")
            else:
                print("⚪ 일반 모드로 복귀")
                self.mode_label.setText("모드: 일반")
                self.mode_label.setStyleSheet("color: #aaa; font-size: 11px; margin: 5px;")
                self.distance_btn.setText("거리 측정")
                self.clear_temp_line()
                # 선택 해제
                if self.first_node:
                    self.first_node.set_selected(False)
                if self.second_node:
                    self.second_node.set_selected(False)
                self.update_scene()


        def handle_distance_mode_click(self, event):
            """거리 측정 모드에서의 클릭 처리"""
            # 클릭한 위치에서 가장 가까운 노드 찾기
            clicked_node = self.find_closest_node_to_click(event.pos())
            
            if clicked_node is None:
                print("❌ 노드를 클릭해주세요")
                self.status_bar.showMessage("노드를 클릭해주세요", 2000)
                return
            
            if self.first_node is None:
                # 첫 번째 노드 선택
                self.first_node = clicked_node
                self.first_node.set_selected(True)
                print(f"📍 첫 번째 노드 선택: {self.first_node.number}")
                self.mode_label.setText("모드: 거리 측정 (두 번째 노드를 클릭하세요)")
                self.status_bar.showMessage(f"첫 번째 노드: {self.first_node.number}", 2000)
                self.update_scene()
                
            elif self.second_node is None:
                # 두 번째 노드 선택
                self.second_node = clicked_node
                self.second_node.set_selected(True)
                print(f"📍 두 번째 노드 선택: {self.second_node.number}")
                
                # 거리 계산
                distance = self.calculate_distance(self.first_node, self.second_node)
                print(f"📏 측정된 거리: {distance:.2f}m")
                
                # UI 업데이트
                self.distance_result_label.setText(f"거리: {distance:.2f}m")
                self.distance_result_label.setStyleSheet("color: #4CAF50; font-size: 11px; margin: 5px; font-weight: bold;")
                
                # 임시 라인 그리기
                self.draw_temp_line(self.first_node.position, self.second_node.position)
                
                # 사용자 입력 거리로 노드 생성
                self.insert_node_btn.setEnabled(True)
                
                # 상태바 업데이트
                self.status_bar.showMessage(
                    f"노드 {self.first_node.number} → {self.second_node.number}: {distance:.2f}m", 
                    5000
                )
                
                
                


        def calculate_distance(self, node1, node2):
            """두 노드 사이의 거리 계산"""
            import numpy as np
            pos1 = np.array(node1.position)
            pos2 = np.array(node2.position)
            return np.linalg.norm(pos2 - pos1)


        def create_node_at_distance(self):
            """지정된 거리에 노드 생성"""
            try:
                # 사용자 입력 거리 가져오기
                target_distance = float(self.distance_input.text() or "5.2")
                print(f"🎯 목표 거리: {target_distance}m")
                
                # 방향 벡터 계산
                import numpy as np
                pos1 = np.array(self.first_node.position)
                pos2 = np.array(self.second_node.position)
                direction = pos2 - pos1
                current_distance = np.linalg.norm(direction)
                
                if current_distance == 0:
                    print("❌ 두 노드가 같은 위치에 있습니다")
                    self.status_bar.showMessage("두 노드가 같은 위치에 있습니다", 3000)
                    return
                
                # 정규화된 방향 벡터
                unit_direction = direction / current_distance
                
                # 새 노드 위치 계산 (첫 번째 노드로부터 target_distance만큼)
                new_position = pos1 + unit_direction * target_distance
                
                # 새 노드 생성
                new_number = self.editor.node_manager.get_next_number()
                from src.data_structures import DataPoint, Node3D
                new_datapoint = DataPoint(
                    new_number, 
                    new_position[0], 
                    new_position[1], 
                    new_position[2]
                )
                new_node = Node3D(new_datapoint)
                
                # 씬에 추가
                self.editor.scene.nodes.append(new_node)
                
                print(f"✅ 새 노드 생성: {new_number} at ({new_position[0]:.2f}, {new_position[1]:.2f}, {new_position[2]:.2f})")
                
                # 시각적 표시
                new_node.set_selected(True)
                self.editor.scene.selected_nodes.add(new_node)
                
                # 결과 표시
                self.distance_result_label.setText(
                    f"✅ 노드 {new_number} 생성됨\n"
                    f"거리: {target_distance}m\n"
                    f"위치: ({new_position[0]:.1f}, {new_position[1]:.1f}, {new_position[2]:.1f})"
                )
                self.distance_result_label.setStyleSheet("color: #4CAF50; font-size: 11px; margin: 5px;")
                
                # 새 노드를 가리키는 화살표 추가 (선택사항)
                self.draw_temp_line(self.first_node.position, new_position)
                
                # 씬 업데이트
                self.update_scene()
                
                # 3초 후 모드 리셋
                QtCore.QTimer.singleShot(3000, self.reset_distance_mode)
                
            except ValueError:
                print("❌ 올바른 거리를 입력하세요")
                self.distance_result_label.setText("❌ 올바른 거리를 입력하세요")
                self.distance_result_label.setStyleSheet("color: #f44336; font-size: 11px; margin: 5px;")


        def reset_distance_mode(self):
            """거리 측정 모드 리셋"""
            if self.first_node:
                self.first_node.set_selected(False)
            if self.second_node:
                self.second_node.set_selected(False)
            
            self.first_node = None
            self.second_node = None
            self.clear_temp_line()
            self.mode_label.setText("모드: 거리 측정 (첫 번째 노드를 클릭하세요)")
            self.distance_result_label.setText("측정 대기 중...")
            self.distance_result_label.setStyleSheet("color: #888; font-size: 11px; margin: 5px;")
            self.update_scene()
            

        def draw_temp_line(self, pos1, pos2):
            """임시 측정 라인 그리기"""
            self.clear_temp_line()
            
            import numpy as np
            self.temp_line = gl.GLLinePlotItem(
                pos=np.array([pos1, pos2]),
                color=(0, 1, 1, 0.8),  # 청록색
                width=3
            )
            self.gl_widget.addItem(self.temp_line)


        def clear_temp_line(self):
            """임시 라인 제거"""
            if hasattr(self, 'temp_line') and self.temp_line:
                self.gl_widget.removeItem(self.temp_line)
                self.temp_line = None


        def find_closest_node_to_click(self, mouse_pos):
            """마우스 클릭 위치에서 가장 가까운 노드 찾기"""
            try:
                mvp = self.gl_widget.projectionMatrix() * self.gl_widget.viewMatrix()
                width, height = self.gl_widget.width(), self.gl_widget.height()
                
                mouse_x = mouse_pos.x()
                mouse_y = mouse_pos.y()
                
                closest_node = None
                min_distance = float('inf')
                detection_radius = 20  # 픽셀 단위
                
                for node in self.editor.scene.nodes:
                    # 보이지 않는 노드는 스킵
                    if not getattr(node, 'is_visible', True):
                        continue
                        
                    # 노드를 화면 좌표로 변환
                    screen_pos = self.world_to_screen(node.position, mvp, width, height)
                    
                    if screen_pos is None:
                        continue
                    
                    # 마우스와의 거리 계산
                    distance = ((mouse_x - screen_pos[0])**2 + (mouse_y - screen_pos[1])**2)**0.5
                    
                    if distance < detection_radius and distance < min_distance:
                        min_distance = distance
                        closest_node = node
                
                if closest_node:
                    print(f"✅ 가장 가까운 노드: {closest_node.number} (거리: {min_distance:.1f}px)")
                    
                return closest_node
            
            except Exception as e:  # ✅ 이 부분 추가!
                print(f"❌ 노드 검색 오류: {e}")
                return None
            
            # ✅ 여기에 추가! (find_closest_node_to_click 메서드 다음)
        def insert_node_at_distance(self):
            """사용자가 삽입 버튼을 클릭했을 때만 노드 생성"""
            print("🔘🔘🔘 노드 삽입 버튼 클릭됨! 🔘🔘🔘")  # 이게 출력되는지 확인!
            
            if not hasattr(self, 'first_node') or not hasattr(self, 'second_node'):
                print("❌ 속성이 없음")
                return
                
            if not self.first_node or not self.second_node:
                print("❌ 먼저 두 노드를 선택하세요")
                print(f"   first_node: {self.first_node}")
                print(f"   second_node: {self.second_node}")
                return
            
            # 거리 입력값 확인
            try:
                target_distance = float(self.distance_input.text() or "0")
                if target_distance <= 0:
                    self.status_bar.showMessage("올바른 거리를 입력하세요 (0보다 큰 값)", 3000)
                    return
            except ValueError:
                self.status_bar.showMessage("올바른 숫자를 입력하세요", 3000)
                return
            
            print(f"🎯 노드 삽입 시작 - 목표 거리: {target_distance}m")
            
            # 노드 생성 함수 호출
            self.create_node_at_distance()
            
            # 삽입 버튼 비활성화 (중복 클릭 방지)
            self.insert_node_btn.setEnabled(False)

        # ✅ 여기에 패턴 인식 메서드들 추가!

        def learn_pattern(self):
            """선택된 노드들의 패턴 학습"""
            selected = list(self.editor.scene.selected_nodes)
            
            if len(selected) < 2:
                self.pattern_info_label.setText("❌ 2개 이상의 노드를 선택하세요")
                self.pattern_info_label.setStyleSheet("color: #f44336; font-size: 11px; margin: 5px;")
                return
            
            print(f"🤖 패턴 학습 시작: {len(selected)}개 노드")
            
            # 노드들을 번호 순으로 정렬
            selected.sort(key=lambda n: n.number)
            
            # 위치 데이터 추출
            positions = np.array([node.position for node in selected])
            
            # 패턴 분석
            pattern_type, pattern_data = self.analyze_pattern(positions)
            
            # 패턴 저장
            self.learned_pattern = {
                'type': pattern_type,
                'data': pattern_data,
                'positions': positions,
                'nodes': selected
            }
            
            # UI 업데이트
            if pattern_type == 'linear':
                direction = pattern_data['direction']
                spacing = pattern_data['spacing']
                self.pattern_info_label.setText(
                    f"✅ 선형 패턴 감지!\n"
                    f"방향: ({direction[0]:.2f}, {direction[1]:.2f}, {direction[2]:.2f})\n"
                    f"간격: {spacing:.2f}m"
                )
            elif pattern_type == 'grid':
                self.pattern_info_label.setText(
                    f"✅ 격자 패턴 감지!\n"
                    f"X 간격: {pattern_data['x_spacing']:.2f}m\n"
                    f"Y 간격: {pattern_data['y_spacing']:.2f}m"
                )
            elif pattern_type == 'circular':
                self.pattern_info_label.setText(
                    f"✅ 원형 패턴 감지!\n"
                    f"중심: ({pattern_data['center'][0]:.1f}, {pattern_data['center'][1]:.1f})\n"
                    f"반경: {pattern_data['radius']:.2f}m"
                )
            else:
                self.pattern_info_label.setText("✅ 패턴 학습 완료!")
            
            self.pattern_info_label.setStyleSheet("color: #4CAF50; font-size: 11px; margin: 5px;")
            self.apply_pattern_btn.setEnabled(True)


        def analyze_pattern(self, positions):
            """위치 데이터에서 패턴 분석"""
            n = len(positions)
            
            if n < 2:
                return 'none', {}
            
            # 1. 선형 패턴 검사 (일직선상에 배치)
            if n >= 2:
                # 첫 번째와 마지막 점을 잇는 직선에 대한 거리 계산
                line_vec = positions[-1] - positions[0]
                line_len = np.linalg.norm(line_vec)
                
                if line_len > 0:
                    line_dir = line_vec / line_len
                    
                    # 각 점의 직선으로부터의 거리
                    distances = []
                    for i in range(1, n-1):
                        vec_to_point = positions[i] - positions[0]
                        proj_len = np.dot(vec_to_point, line_dir)
                        proj_point = positions[0] + proj_len * line_dir
                        dist = np.linalg.norm(positions[i] - proj_point)
                        distances.append(dist)
                    
                    # 모든 점이 직선 근처에 있으면 선형 패턴
                    if not distances or max(distances) < 0.5:  # 0.5m 오차 허용
                        # 등간격인지 확인
                        spacings = []
                        for i in range(n-1):
                            spacing = np.linalg.norm(positions[i+1] - positions[i])
                            spacings.append(spacing)
                        
                        avg_spacing = np.mean(spacings)
                        
                        return 'linear', {
                            'direction': line_dir,
                            'spacing': avg_spacing,
                            'start': positions[0],
                            'end': positions[-1]
                        }
            
            # 2. 격자 패턴 검사 (2D 그리드)
            x_coords = positions[:, 0]
            y_coords = positions[:, 1]
            z_coords = positions[:, 2]
            
            # Z 좌표가 거의 같으면 2D 그리드 가능성
            if np.std(z_coords) < 0.5:
                unique_x = np.unique(np.round(x_coords, 1))
                unique_y = np.unique(np.round(y_coords, 1))
                
                if len(unique_x) > 1 and len(unique_y) > 1:
                    x_spacing = np.mean(np.diff(sorted(unique_x)))
                    y_spacing = np.mean(np.diff(sorted(unique_y)))
                    
                    return 'grid', {
                        'x_spacing': x_spacing,
                        'y_spacing': y_spacing,
                        'z_level': np.mean(z_coords),
                        'x_count': len(unique_x),
                        'y_count': len(unique_y)
                    }
            
            # 3. 원형 패턴 검사
            if n >= 4:
                # 2D 평면에서 원형 패턴 검사 (Z 좌표 무시)
                xy_positions = positions[:, :2]
                center = np.mean(xy_positions, axis=0)
                
                # 중심으로부터의 거리
                radii = [np.linalg.norm(pos - center) for pos in xy_positions]
                avg_radius = np.mean(radii)
                radius_std = np.std(radii)
                
                # 반경의 편차가 작으면 원형 패턴
                if radius_std < avg_radius * 0.1:  # 10% 오차 허용
                    # 각도 계산
                    angles = []
                    for pos in xy_positions:
                        vec = pos - center
                        angle = np.arctan2(vec[1], vec[0])
                        angles.append(angle)
                    
                    return 'circular', {
                        'center': np.append(center, np.mean(z_coords)),
                        'radius': avg_radius,
                        'angles': sorted(angles),
                        'z_level': np.mean(z_coords)
                    }
            
            # 4. 일반 패턴
            return 'general', {
                'positions': positions
            }


        def apply_pattern(self):
            """학습된 패턴을 적용하여 새 노드 생성"""
            if not hasattr(self, 'learned_pattern'):
                self.pattern_info_label.setText("❌ 먼저 패턴을 학습하세요")
                return
            
            count = self.copy_count_input.value()
            pattern = self.learned_pattern
            print(f"🔄 패턴 적용: {pattern['type']} 패턴으로 {count}개 복사")
            
            new_nodes = []
            
            if pattern['type'] == 'linear':
                # 선형 패턴 적용
                direction = pattern['data']['direction']
                spacing = pattern['data']['spacing']
                last_pos = pattern['positions'][-1]
                
                for i in range(count):
                    new_pos = last_pos + direction * spacing * (i + 1)
                    new_node = self.create_node_at_position(new_pos)
                    if new_node:
                        new_nodes.append(new_node)
            
            elif pattern['type'] == 'grid':
                # 격자 패턴 적용
                x_spacing = pattern['data']['x_spacing']
                y_spacing = pattern['data']['y_spacing']
                z_level = pattern['data']['z_level']
                
                # 현재 그리드의 경계 찾기
                positions = pattern['positions']
                max_x = np.max(positions[:, 0])
                min_y = np.min(positions[:, 1])
                
                # 다음 행에 노드 추가
                for i in range(count):
                    x = min_x = np.min(positions[:, 0])
                    y = min_y - y_spacing * (i + 1)
                    
                    for j in range(pattern['data']['x_count']):
                        new_pos = np.array([x, y, z_level])
                        new_node = self.create_node_at_position(new_pos)
                        if new_node:
                            new_nodes.append(new_node)
                        x += x_spacing
            
            elif pattern['type'] == 'circular':
                # 원형 패턴 적용
                center = pattern['data']['center']
                radius = pattern['data']['radius']
                angles = pattern['data']['angles']
                
                # 각도 간격 계산
                if len(angles) > 1:
                    angle_step = angles[1] - angles[0]
                else:
                    angle_step = 2 * np.pi / 8  # 기본값: 45도
                
                last_angle = angles[-1]
                
                for i in range(count):
                    angle = last_angle + angle_step * (i + 1)
                    x = center[0] + radius * np.cos(angle)
                    y = center[1] + radius * np.sin(angle)
                    z = center[2]
                    
                    new_pos = np.array([x, y, z])
                    new_node = self.create_node_at_position(new_pos)
                    if new_node:
                        new_nodes.append(new_node)
            
            # 결과 표시
            if new_nodes:
                self.pattern_info_label.setText(
                    f"✅ {len(new_nodes)}개 노드 생성 완료!\n"
                    f"패턴: {pattern['type']}"
                )
                self.pattern_info_label.setStyleSheet("color: #4CAF50; font-size: 11px; margin: 5px;")
                
                # 새 노드들 선택
                self.editor.scene.clear_selection()
                for node in new_nodes:
                    node.set_selected(True)
                    self.editor.scene.selected_nodes.add(node)
                
                self.update_scene()
                self.update_status()
            else:
                self.pattern_info_label.setText("❌ 노드 생성 실패")
                          
        def create_node_at_position(self, position):
            """지정된 위치에 노드 생성"""
            try:
                new_number = self.editor.node_manager.get_next_number()
                from src.data_structures import DataPoint, Node3D
                
                datapoint = DataPoint(
                    new_number,
                    position[0],
                    position[1],
                    position[2]
                )
                new_node = Node3D(datapoint)
                
                # 씬에 추가
                self.editor.scene.nodes.append(new_node)
                
                print(f"✅ 노드 {new_number} 생성: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})")
                return new_node
                
            except Exception as e:
                print(f"❌ 노드 생성 오류: {e}")
                return None
            
        def create_node_at_position_safe(self, position, tolerance=0.1):
            """지정된 위치에 노드 생성 (중복 체크 포함)"""
            try:
                # 1. 기존 노드 중 같은 위치에 있는지 확인
                for existing_node in self.editor.scene.nodes:
                    dist = np.linalg.norm(
                        np.array(existing_node.position) - np.array(position)
                    )
                    
                    if dist < tolerance:
                        print(f"🔄 기존 노드 {existing_node.number} 재사용 (거리: {dist:.3f}m)")
                        return existing_node
                
                # 2. 새 노드 생성
                new_number = self.editor.node_manager.get_next_number()
                from src.data_structures import DataPoint, Node3D
                
                datapoint = DataPoint(
                    new_number,
                    position[0],
                    position[1],
                    position[2]
                )
                new_node = Node3D(datapoint)
                
                # 3. 생성된 노드는 원본이 아님
                new_node.is_original = False
                new_node.is_protected = False
                
                # 씬에 추가
                self.editor.scene.nodes.append(new_node)
                
                print(f"✅ 새 노드 {new_number} 생성: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})")
                return new_node
                
            except Exception as e:
                print(f"❌ 노드 생성 오류: {e}")
                return None
            
        def create_midpoint_on_edge_safe(self, nodes, plane, edge, min_u, max_u, min_v, max_v, fixed_coord):
            """특정 변의 중점에 노드 생성 (중복 체크 포함)"""
            tolerance = 0.1
            
            # 변에 해당하는 노드들 찾기
            edge_nodes = []
            
            for node in nodes:
                if plane == 'XY':
                    u, v = node.position[0], node.position[1]
                elif plane == 'XZ':
                    u, v = node.position[0], node.position[2]
                else:  # YZ
                    u, v = node.position[1], node.position[2]
                
                # 어느 변에 속하는지 확인
                if edge == 'bottom' and abs(v - min_v) < tolerance:
                    edge_nodes.append(node)
                elif edge == 'top' and abs(v - max_v) < tolerance:
                    edge_nodes.append(node)
                elif edge == 'left' and abs(u - min_u) < tolerance:
                    edge_nodes.append(node)
                elif edge == 'right' and abs(u - max_u) < tolerance:
                    edge_nodes.append(node)
            
            # 변에 2개의 노드가 있으면 중점 계산
            if len(edge_nodes) == 2:
                mid_x = (edge_nodes[0].position[0] + edge_nodes[1].position[0]) / 2
                mid_y = (edge_nodes[0].position[1] + edge_nodes[1].position[1]) / 2
                mid_z = (edge_nodes[0].position[2] + edge_nodes[1].position[2]) / 2
                
                # 중복 체크를 포함한 노드 생성
                new_node = self.create_node_at_position_safe([mid_x, mid_y, mid_z])
                print(f"📍 {edge} 변 중점 처리: 노드 {new_node.number if new_node else 'Failed'}")
                return new_node
            
            return None
        
        def create_paner_line_safe(self, start_node, end_node):
            """PANER 타입 라인 생성 (중복 체크 포함)"""
            from src.data_structures import Line3D, LineType
            
            # 이미 존재하는 라인인지 체크
            for existing_line in self.editor.scene.lines:
                # 같은 노드를 연결하는 라인이 있는지 확인
                if ((existing_line.start_node == start_node and 
                    existing_line.end_node == end_node) or
                    (existing_line.start_node == end_node and 
                    existing_line.end_node == start_node)):
                    
                    print(f"⚠️ PANER 라인이 이미 존재: {start_node.number} - {end_node.number}")
                    return None
            
            # 새 라인 생성
            line = Line3D(start_node, end_node, LineType.PANER)
            self.editor.scene.lines.append(line)
            print(f"✅ PANER 라인 생성: {start_node.number} - {end_node.number}")
            
            return line
            
        def create_cross_connection(self):
            """선택된 4개 노드를 십자 형태로 PANER 연결"""
            selected = list(self.editor.scene.selected_nodes)
            
            if len(selected) != 4:
                self.intersection_info_label.setText("❌ 정확히 4개 노드를 선택하세요")
                return
            
            # 1. 노드들의 중심점 계산
            positions = [node.position for node in selected]
            center = self.calculate_center_point(positions)
            
            # 2. 교차점에 새 노드 생성
            intersection_node = self.create_node_at_position(center)
            
            # 3. 각 노드에서 중심점으로 PANER 라인 연결
            for node in selected:
                self.create_paner_line(node, intersection_node)
            
            # 4. UI 업데이트
            self.intersection_info_label.setText(
                f"✅ 십자 연결 완료!\n"
                f"교차점 노드: {intersection_node.number}\n"
                f"PANER 라인: 4개 생성"
            )
            
            self.update_scene()
            self.update_status()

        def calculate_center_point(self, positions):
            """여러 점의 중심점 계산"""
            import numpy as np
            positions_array = np.array(positions)
            return np.mean(positions_array, axis=0)

        def create_paner_line(self, start_node, end_node):
            """PANER 타입 라인 생성"""
            from src.data_structures import Line3D, LineType
            
            line = Line3D(start_node, end_node, LineType.PANER)
            self.editor.scene.lines.append(line)
            
            return line
        
        def set_selected_as_exterior_group(self):
            """선택된 노드와 라인을 외장 그룹(Group 5)으로 설정"""
            selected_nodes = list(self.editor.scene.selected_nodes)
            
            print(f"🔍 선택된 노드 수: {len(selected_nodes)}")  # 디버그
            
            if not selected_nodes:
                self.status_bar.showMessage("❌ 노드를 선택하세요", 2000)
                return
            
            # 외장 그룹 ID (Group 5 = index 4)
            EXTERIOR_GROUP_ID = 4
            
            # 선택된 노드들을 외장 그룹으로 설정
            changed_nodes = 0  # 카운터 추가
            for node in selected_nodes:
                # 기존 그룹 확인
                old_group = getattr(node, 'group_id', None)
                node.group_id = EXTERIOR_GROUP_ID
                changed_nodes += 1
                print(f"🏢 노드 {node.number}: {old_group} → 외장 그룹")
            
            print(f"✅ 총 {changed_nodes}개 노드가 Group 5로 변경됨")
            
            # 선택된 노드들과 연결된 라인도 확인
            updated_lines = 0
            if hasattr(self.editor.scene, 'lines'):
                print(f"🔍 총 라인 수: {len(self.editor.scene.lines)}")
                
                # 선택된 노드들의 위치를 미리 계산
                selected_positions = {tuple(node.position): node for node in selected_nodes}
                
                for line in self.editor.scene.lines:
                    # 라인의 시작점과 끝점이 선택된 노드의 위치와 일치하는지 확인
                    start_pos_tuple = tuple(line.start_pos)
                    end_pos_tuple = tuple(line.end_pos)
                    
                    start_in_selected = start_pos_tuple in selected_positions
                    end_in_selected = end_pos_tuple in selected_positions
                    
                    # 양쪽 다 선택된 경우만 외장 그룹으로
                    if start_in_selected and end_in_selected:
                        # 라인의 그룹 설정
                        if not hasattr(line, 'group_ids'):
                            line.group_ids = set()
                        line.group_ids.add(EXTERIOR_GROUP_ID)
                        updated_lines += 1
                        
                        # 디버그 출력
                        start_node = selected_positions[start_pos_tuple]
                        end_node = selected_positions[end_pos_tuple]
                        print(f"📏 라인 추가: 노드 {start_node.number} - 노드 {end_node.number}")
            
            # 상태바 업데이트
            self.status_bar.showMessage(
                f"외장 그룹(Group 5) 설정: 노드 {len(selected_nodes)}개, 라인 {updated_lines}개", 
                3000
            )
            
            self.update_scene()
            self.update_status()
                
        def select_by_coordinates(self):
            """체크된 좌표 기준으로 노드/라인 선택"""
            # 현재 선택된 노드가 있는지 확인
            if not self.editor.scene.selected_nodes:
                self.status_bar.showMessage("먼저 기준이 될 노드를 하나 선택하세요", 3000)
                return
            
            # 기준 노드 (첫 번째 선택된 노드)
            reference_node = list(self.editor.scene.selected_nodes)[0]
            ref_pos = reference_node.position
            
            # 어떤 좌표를 고정할지 확인
            fix_x = self.x_coord_checkbox.isChecked()
            fix_y = self.y_coord_checkbox.isChecked()
            fix_z = self.z_coord_checkbox.isChecked()
            
            if not (fix_x or fix_y or fix_z):
                self.status_bar.showMessage("X, Y, Z 중 하나 이상을 체크하세요", 3000)
                return
            
            # 허용 오차
            tolerance = 0.1  # 기본값, 나중에 설정 가능하게 만들 수 있음
            
            # 기존 선택 유지하면서 추가
            selected_count = 0
            
            # 노드 선택
            for node in self.editor.scene.nodes:
                if node in self.editor.scene.selected_nodes:
                    continue  # 이미 선택된 노드는 스킵
                    
                # 좌표 비교
                match = True
                if fix_x and abs(node.position[0] - ref_pos[0]) > tolerance:
                    match = False
                if fix_y and abs(node.position[1] - ref_pos[1]) > tolerance:
                    match = False
                if fix_z and abs(node.position[2] - ref_pos[2]) > tolerance:
                    match = False
                
                if match:
                    node.set_selected(True)
                    self.editor.scene.selected_nodes.add(node)
                    selected_count += 1
            
            # 라인 선택 (양 끝점이 모두 조건에 맞는 경우)
            selected_line_count = 0
            if hasattr(self.editor.scene, 'lines'):
                for line in self.editor.scene.lines:
                    # 라인의 양 끝점 체크
                    start_match = True
                    end_match = True
                    
                    if fix_x:
                        if abs(line.start_pos[0] - ref_pos[0]) > tolerance:
                            start_match = False
                        if abs(line.end_pos[0] - ref_pos[0]) > tolerance:
                            end_match = False
                    if fix_y:
                        if abs(line.start_pos[1] - ref_pos[1]) > tolerance:
                            start_match = False
                        if abs(line.end_pos[1] - ref_pos[1]) > tolerance:
                            end_match = False
                    if fix_z:
                        if abs(line.start_pos[2] - ref_pos[2]) > tolerance:
                            start_match = False
                        if abs(line.end_pos[2] - ref_pos[2]) > tolerance:
                            end_match = False
                    
                    # 양 끝점이 모두 조건에 맞으면 선택
                    if start_match and end_match:
                        if not hasattr(line, 'is_selected'):
                            line.is_selected = False
                        line.is_selected = True
                        selected_line_count += 1
            
            # 결과 표시
            coord_str = []
            if fix_x: coord_str.append(f"X={ref_pos[0]:.1f}")
            if fix_y: coord_str.append(f"Y={ref_pos[1]:.1f}")
            if fix_z: coord_str.append(f"Z={ref_pos[2]:.1f}")
            
            self.status_bar.showMessage(
                f"좌표 선택 완료 ({', '.join(coord_str)}): "
                f"노드 {selected_count}개, 라인 {selected_line_count}개 추가 선택", 
                5000
            )
            
            # 화면 업데이트
            self.update_scene()
            self.update_status()
            
        # ✅ 줌 관련 메서드들 추가
        def toggle_zoom_mode(self):
            """줌 모드 토글"""
            self.zoom_mode = self.zoom_mode_action.isChecked()
            
            # 다른 모드 해제
            if self.zoom_mode:
                self.selection_mode = False
                self.selection_mode_action.setChecked(False)
                self.setCursor(QtCore.Qt.CrossCursor)
                self.status_bar.showMessage("🔍 줌 모드 - 드래그하여 영역 확대", 2000)
            else:
                self.setCursor(QtCore.Qt.ArrowCursor)
                self.status_bar.showMessage("줌 모드 해제", 2000)

        def start_zoom_rect(self, pos):
            """줌 영역 표시 시작"""
            self.status_bar.showMessage(f"줌 영역 선택 중... 시작: ({pos.x()}, {pos.y()})")

        def update_zoom_rect(self, pos):
            """줌 영역 업데이트"""
            if self.zoom_start:
                width = abs(pos.x() - self.zoom_start.x())
                height = abs(pos.y() - self.zoom_start.y())
                self.status_bar.showMessage(f"줌 영역: {width} x {height}")

        def finish_zoom(self, end_pos):
            """줌 실행"""
            if not self.zoom_start:
                return
            
            # 드래그한 영역의 3D 좌표 계산
            bounds_min, bounds_max = self.calculate_zoom_bounds(self.zoom_start, end_pos)
            
            if bounds_min is None or bounds_max is None:
                self.status_bar.showMessage("줌 영역이 너무 작습니다", 2000)
                return
            
            # 선택된 영역으로 카메라 이동
            self.zoom_to_bounds(bounds_min, bounds_max)
            
            # 줌 모드 해제
            self.zoom_mode_action.setChecked(False)
            self.toggle_zoom_mode()

        def calculate_zoom_bounds(self, start_pos, end_pos):
            """화면 좌표를 3D 공간 경계로 변환"""
            mvp = self.gl_widget.projectionMatrix() * self.gl_widget.viewMatrix()
            width, height = self.gl_widget.width(), self.gl_widget.height()
            
            # 화면 좌표 경계
            min_x = min(start_pos.x(), end_pos.x())
            max_x = max(start_pos.x(), end_pos.x())
            min_y = min(start_pos.y(), end_pos.y())
            max_y = max(start_pos.y(), end_pos.y())
            
            # 너무 작은 영역 체크
            if abs(max_x - min_x) < 10 or abs(max_y - min_y) < 10:
                return None, None
            
            # 드래그 영역의 화면 비율 계산
            drag_width_ratio = (max_x - min_x) / width
            drag_height_ratio = (max_y - min_y) / height
            
            # 영역 내 노드들 찾기
            nodes_in_region = []
            
            for node in self.editor.scene.nodes:
                screen_pos = self.world_to_screen(node.position, mvp, width, height)
                if screen_pos is None:
                    continue
                    
                if min_x <= screen_pos[0] <= max_x and min_y <= screen_pos[1] <= max_y:
                    nodes_in_region.append(node)
            
            if not nodes_in_region:
                # 노드가 없어도 대략적인 영역 계산
                # 현재 뷰의 중심과 거리를 기준으로 추정
                current_center = self.gl_widget.opts['center']
                current_distance = self.gl_widget.opts.get('distance', 100)
                
                # 드래그 영역의 중심 (화면 좌표)
                drag_center_x = (min_x + max_x) / 2
                drag_center_y = (min_y + max_y) / 2
                
                # 화면 중심으로부터의 오프셋 비율
                offset_x = (drag_center_x - width/2) / width
                offset_y = (drag_center_y - height/2) / height
                
                # 새로운 중심 추정
                import numpy as np
                estimated_center = np.array([
                    current_center.x() + offset_x * current_distance,
                    current_center.y() - offset_y * current_distance,  # Y는 반대
                    current_center.z()
                ])
                
                # 추정된 경계
                estimated_size = current_distance * max(drag_width_ratio, drag_height_ratio)
                bounds_min = estimated_center - estimated_size / 2
                bounds_max = estimated_center + estimated_size / 2
                
                return bounds_min, bounds_max
            
            # 3D 경계 계산
            import numpy as np
            positions = np.array([n.position for n in nodes_in_region])
            bounds_min = np.min(positions, axis=0)
            bounds_max = np.max(positions, axis=0)
            
            # 디버그 정보
            print(f"📐 드래그 비율: {drag_width_ratio:.2f} x {drag_height_ratio:.2f}")
            print(f"📍 선택된 노드: {len(nodes_in_region)}개")
            
            return bounds_min, bounds_max

        def zoom_to_bounds(self, bounds_min, bounds_max):
            """지정된 경계로 카메라 줌"""
            import numpy as np
            from PyQt5.QtGui import QVector3D
            
            # 경계의 중심
            center = (bounds_min + bounds_max) / 2
            
            # 경계 상자의 크기
            size = bounds_max - bounds_min
            
            # 화면 비율 고려
            viewport_width = self.gl_widget.width()
            viewport_height = self.gl_widget.height()
            aspect_ratio = viewport_width / viewport_height if viewport_height > 0 else 1
            
            # 뷰 방향에 따른 크기 선택
            current_elevation = self.gl_widget.opts.get('elevation', 30)
            
            if abs(current_elevation) > 80:  # Top 뷰
                # X, Y 평면에서 가장 큰 크기
                view_size = max(size[0] * aspect_ratio, size[1]) 
            else:  # 다른 뷰들
                # 3차원 모두 고려
                view_size = max(size[0] * aspect_ratio, size[1], size[2] * 2)
            
            # ✅ 거리 계산 수정
            distance = view_size * 0.7  # 화면 크기의 70%를 객체가 차지
            
            # ✅ 최소 거리를 제거하거나 작게 설정
            distance = max(distance, 1000.0)  # 100.0 → 10.0으로 변경
            
            # 카메라 설정
            self.gl_widget.opts['center'] = QVector3D(center[0], center[1], center[2])
            self.gl_widget.opts['distance'] = distance
            self.gl_widget.update()
            
            self.status_bar.showMessage(f"영역 확대 완료 (거리: {distance:.1f})", 3000)
            
            # 디버그 정보
            print(f"🔍 줌 정보:")
            print(f"   경계 크기: {size}")
            print(f"   최대 크기: {max(size):.2f}")
            print(f"   뷰 크기: {view_size:.2f}")
            print(f"   카메라 거리: {distance:.2f}")

        def reset_zoom(self):
            """전체 뷰로 리셋"""
            self.fit_to_view()
            self.status_bar.showMessage("전체 뷰로 리셋", 2000)
            
        def create_rectangular_panel(self):
            """선택된 4개 노드로 사각형 패널 생성 (중복 체크 포함)"""
            selected = list(self.editor.scene.selected_nodes)
            
            if len(selected) != 4:
                self.panel_info_label.setText("❌ 정확히 4개의 노드를 선택하세요")
                return
            
            import numpy as np
            
            # 1. 선택된 노드들의 좌표 추출
            positions = np.array([node.position for node in selected])
            
            # 2. X, Y, Z 좌표별로 분석
            x_coords = positions[:, 0]
            y_coords = positions[:, 1]
            z_coords = positions[:, 2]
            
            # 3. 어느 평면에 있는지 판단
            x_range = np.max(x_coords) - np.min(x_coords)
            y_range = np.max(y_coords) - np.min(y_coords)
            z_range = np.max(z_coords) - np.min(z_coords)
            
            print(f"좌표 범위 - X: {x_range:.2f}, Y: {y_range:.2f}, Z: {z_range:.2f}")
            
            # 4. 평면 판단
            if z_range < 0.1:  # XY 평면
                plane = 'XY'
                fixed_coord = np.mean(z_coords)
            elif y_range < 0.1:  # XZ 평면
                plane = 'XZ'
                fixed_coord = np.mean(y_coords)
            else:  # YZ 평면
                plane = 'YZ'
                fixed_coord = np.mean(x_coords)
            
            # 5. 최소/최대 좌표 찾기
            min_u = np.min(positions[:, 0] if plane != 'YZ' else positions[:, 1])
            max_u = np.max(positions[:, 0] if plane != 'YZ' else positions[:, 1])
            min_v = np.min(positions[:, 1] if plane == 'XY' else positions[:, 2])
            max_v = np.max(positions[:, 1] if plane == 'XY' else positions[:, 2])
            
            # 6. 변의 중점 생성 (중복 체크 포함)
            new_nodes = []
            created_nodes = []  # 새로 생성된 노드만 추적
            
            # 아래 변 중점
            bottom_mid = self.create_midpoint_on_edge_safe(
                selected, plane, 'bottom', min_u, max_u, min_v, max_v, fixed_coord
            )
            if bottom_mid:
                new_nodes.append(bottom_mid)
                if not bottom_mid.is_original:  # 새로 생성된 노드인 경우
                    created_nodes.append(bottom_mid)
            
            # 위 변 중점
            top_mid = self.create_midpoint_on_edge_safe(
                selected, plane, 'top', min_u, max_u, min_v, max_v, fixed_coord
            )
            if top_mid:
                new_nodes.append(top_mid)
                if not top_mid.is_original:
                    created_nodes.append(top_mid)
            
            # 왼쪽 변 중점
            left_mid = self.create_midpoint_on_edge_safe(
                selected, plane, 'left', min_u, max_u, min_v, max_v, fixed_coord
            )
            if left_mid:
                new_nodes.append(left_mid)
                if not left_mid.is_original:
                    created_nodes.append(left_mid)
            
            # 오른쪽 변 중점
            right_mid = self.create_midpoint_on_edge_safe(
                selected, plane, 'right', min_u, max_u, min_v, max_v, fixed_coord
            )
            if right_mid:
                new_nodes.append(right_mid)
                if not right_mid.is_original:
                    created_nodes.append(right_mid)
            
            # 7. 중심점 생성 (중복 체크)
            center_x = np.mean(x_coords)
            center_y = np.mean(y_coords)
            center_z = np.mean(z_coords)
            center_node = self.create_node_at_position_safe([center_x, center_y, center_z])
            
            if center_node and not center_node.is_original:
                created_nodes.append(center_node)
            
            # 8. 십자 연결 (중복 체크 포함)
            created_lines = 0
            if len(new_nodes) >= 4:
                # 위-아래 연결
                if bottom_mid and top_mid:
                    if self.create_paner_line_safe(bottom_mid, top_mid):
                        created_lines += 1
                
                # 좌-우 연결
                if left_mid and right_mid:
                    if self.create_paner_line_safe(left_mid, right_mid):
                        created_lines += 1

                # ✅ 여기에 패널 분할 로직 추가!
                # 패널 내부 분할 생성 (사용자가 3x3, 4x4 등을 원할 때)
                if self.panel_divisions_x.value() > 2 or self.panel_divisions_y.value() > 2:
                    # 분할 수 가져오기
                    div_x = self.panel_divisions_x.value()
                    div_y = self.panel_divisions_y.value()
                    
                    print(f"📐 패널 분할: {div_x} x {div_y}")
                    
                    # 패널의 경계 계산
                    min_x, max_x = np.min(x_coords), np.max(x_coords)
                    min_y, max_y = np.min(y_coords), np.max(y_coords)
                    min_z, max_z = np.min(z_coords), np.max(z_coords)
                    
                    # 내부 그리드 노드 생성
                    internal_nodes = []
                    for i in range(1, div_x):
                        for j in range(1, div_y):
                            # 내부 노드 위치 계산
                            if plane == 'XY':
                                x = min_x + (max_x - min_x) * i / div_x
                                y = min_y + (max_y - min_y) * j / div_y
                                z = fixed_coord
                            elif plane == 'XZ':
                                x = min_x + (max_x - min_x) * i / div_x
                                y = fixed_coord
                                z = min_z + (max_z - min_z) * j / div_y
                            else:  # YZ
                                x = fixed_coord
                                y = min_y + (max_y - min_y) * i / div_y
                                z = min_z + (max_z - min_z) * j / div_y
                            
                            # 노드 생성 (중복 체크 포함)
                            node = self.create_node_at_position_safe([x, y, z])
                            if node and not node.is_original:
                                created_nodes.append(node)
                                internal_nodes.append(node)
                    
                    print(f"✅ 내부 그리드 노드 {len(internal_nodes)}개 생성")
                    
                    # TODO: 필요시 내부 노드들을 PANER로 연결
                    # (격자 패턴으로 연결하는 로직 추가 가능)

                # 9. 결과 표시
                self.panel_info_label.setText(
                    f"✅ 패널 생성 완료! ({plane} 평면)\n"
                    f"새 노드: {len(created_nodes)}개\n"
                    f"재사용: {len(new_nodes) - len(created_nodes)}개\n"
                    f"PANER 라인: {created_lines}개"
                )
                self.update_scene()

        def create_midpoint_on_edge(self, nodes, plane, edge, min_u, max_u, min_v, max_v, fixed_coord):
            """특정 변의 중점에 노드 생성"""
            tolerance = 0.1
            
            # 변에 해당하는 노드들 찾기
            edge_nodes = []
            
            for node in nodes:
                if plane == 'XY':
                    u, v = node.position[0], node.position[1]
                elif plane == 'XZ':
                    u, v = node.position[0], node.position[2]
                else:  # YZ
                    u, v = node.position[1], node.position[2]
                
                # 어느 변에 속하는지 확인
                if edge == 'bottom' and abs(v - min_v) < tolerance:
                    edge_nodes.append(node)
                elif edge == 'top' and abs(v - max_v) < tolerance:
                    edge_nodes.append(node)
                elif edge == 'left' and abs(u - min_u) < tolerance:
                    edge_nodes.append(node)
                elif edge == 'right' and abs(u - max_u) < tolerance:
                    edge_nodes.append(node)
            
            # 변에 2개의 노드가 있으면 중점 계산
            if len(edge_nodes) == 2:
                mid_x = (edge_nodes[0].position[0] + edge_nodes[1].position[0]) / 2
                mid_y = (edge_nodes[0].position[1] + edge_nodes[1].position[1]) / 2
                mid_z = (edge_nodes[0].position[2] + edge_nodes[1].position[2]) / 2
                
                new_node = self.create_node_at_position([mid_x, mid_y, mid_z])
                print(f"✅ {edge} 변 중점 노드 생성: {new_node.number if new_node else 'Failed'}")
                return new_node
            
            return None

        def sort_nodes_rectangular(self, nodes):
            """4개 노드를 사각형 순서로 정렬"""
            import numpy as np
            
            # 평면 판별 (모든 노드가 같은 평면에 있는지)
            positions = np.array([n.position for n in nodes])
            
            # 중심점
            center = np.mean(positions, axis=0)
            
            # 각 노드의 각도 계산 (XY 평면 기준)
            angles = []
            for pos in positions:
                dx = pos[0] - center[0]
                dy = pos[1] - center[1]
                angle = np.arctan2(dy, dx)
                angles.append(angle)
            
            # 각도 순으로 정렬
            sorted_indices = np.argsort(angles)
            sorted_nodes = [nodes[i] for i in sorted_indices]
            
            return sorted_nodes

        def create_panel_subdivisions(self, corner_nodes, center_node, divisions):
            """패널을 추가로 분할"""
            # 각 사분면에 대해 분할 수행
            for i in range(4):
                corner1 = corner_nodes[i]
                corner2 = corner_nodes[(i + 1) % 4]
                
                # 사분면의 중간 노드들 생성
                for j in range(1, divisions):
                    # 코너에서 중심으로의 보간
                    t = j / divisions
                    
                    # corner1에서 center로
                    x1 = corner1.position[0] + t * (center_node.position[0] - corner1.position[0])
                    y1 = corner1.position[1] + t * (center_node.position[1] - corner1.position[1])
                    z1 = corner1.position[2] + t * (center_node.position[2] - corner1.position[2])
                    
                    # corner2에서 center로
                    x2 = corner2.position[0] + t * (center_node.position[0] - corner2.position[0])
                    y2 = corner2.position[1] + t * (center_node.position[1] - corner2.position[1])
                    z2 = corner2.position[2] + t * (center_node.position[2] - corner2.position[2])
                    
                    # 두 점 사이의 중점
                    mid_x = (x1 + x2) / 2
                    mid_y = (y1 + y2) / 2
                    mid_z = (z1 + z2) / 2
                    
                    # 노드 생성
                    new_node = self.create_node_at_position([mid_x, mid_y, mid_z])
                    
                    # PANER 라인 연결
                    if new_node and j == divisions // 2:  # 중간 지점만
                        self.create_paner_line(center_node, new_node)
                        
            # ✅ 올바른 들여쓰기 (클래스 안에 있어야 함)
        def open_panel_editor(self):
            """외장 그룹만 별도 창에서 편집"""
            import subprocess
            
            # 먼저 현재 작업 저장 (외장 그룹 포함)
            temp_file = "temp/exterior_group_data.csv"
            
            # temp 폴더 생성
            from pathlib import Path
            Path("temp").mkdir(exist_ok=True)
            
            # Group 5 (외장 그룹)만 필터링해서 저장
            self.save_group_data(temp_file, group_id=4)  # Group 5 = index 4
            
            # 패널 편집기를 별도 프로세스로 실행
            subprocess.Popen([
                sys.executable, 
                "panel_editor.py",  # 별도 파일
                "--input", temp_file
            ])
            
            self.status_bar.showMessage("패널 편집기가 열렸습니다 (외장 그룹)", 3000)

        def save_group_data(self, filepath, group_id):
            """특정 그룹의 데이터만 저장"""
            # Group 5에 속한 노드들만 필터링
            group_nodes = [node for node in self.editor.scene.nodes 
                        if hasattr(node, 'group_id') and node.group_id == group_id]
            
            # CSV로 저장
            import csv
            with open(filepath, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['number', 'x', 'y', 'z', 'group_id'])
                
                for node in group_nodes:
                    writer.writerow([
                        node.number,
                        node.position[0],
                        node.position[1],
                        node.position[2],
                        node.group_id
                    ])
            
            print(f"✅ Group {group_id + 1} 데이터 저장: {len(group_nodes)}개 노드")
            
        def start_panel_mapping(self):
            """패널 맵핑 모드 시작"""
            self.mapping_mode = True
            self.mapped_panels = []
            
            # UI 업데이트
            self.status_bar.showMessage("📐 패널 맵핑 모드 - 4개의 노드를 선택하여 패널을 정의하세요", 5000)
            
            # 맵핑 정보 표시 위젯 생성
            if not hasattr(self, 'mapping_info_dock'):
                self.create_mapping_info_dock()
            self.mapping_info_dock.show()

        def create_mapping_info_dock(self):
            """맵핑 정보 도킹 위젯"""
            from PyQt5.QtWidgets import QDockWidget, QTextEdit
            
            self.mapping_info_dock = QDockWidget("패널 맵핑 정보", self)
            self.mapping_info_text = QTextEdit()
            self.mapping_info_text.setReadOnly(True)
            self.mapping_info_text.setMaximumHeight(200)
            self.mapping_info_dock.setWidget(self.mapping_info_text)
            
            self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.mapping_info_dock)

        def define_rect_panel(self):
            """선택된 4개 노드로 사각형 패널 정의"""
            selected = list(self.editor.scene.selected_nodes)
            
            if len(selected) != 4:
                QtWidgets.QMessageBox.warning(
                    self, '경고',
                    '정확히 4개의 노드를 선택하세요.'
                )
                return
            
            # 패널 추가
            panel_id = self.panel_mapping.add_panel(selected, 'rect', group_id=4)
            
            # 시각적 표시 (선택된 노드들을 다른 색으로)
            for node in selected:
                node.color = np.array([1.0, 0.5, 0.0, 1.0])  # 주황색
            
            # 패널 영역 표시 (옵션)
            self.draw_panel_outline(selected)
            
            # 정보 업데이트
            self.update_mapping_info()
            
            self.status_bar.showMessage(f"✅ 패널 {panel_id} 정의 완료", 3000)

        def draw_panel_outline(self, corner_nodes):
            """패널 외곽선 그리기"""
            # 임시 라인으로 패널 표시
            positions = [node.position for node in corner_nodes]
            
            # 사각형 순서로 정렬 (시계방향)
            sorted_nodes = self.sort_nodes_rectangular(corner_nodes)
            
            # 외곽선 그리기
            for i in range(4):
                start = sorted_nodes[i]
                end = sorted_nodes[(i + 1) % 4]
                
                # 임시 시각적 라인 추가 (실제 라인이 아닌 표시용)
                line = gl.GLLinePlotItem(
                    pos=np.array([start.position, end.position]),
                    color=(1, 0.5, 0, 1),  # 주황색
                    width=3
                )
                self.gl_widget.addItem(line)
                
                # 나중에 지울 수 있도록 저장
                if not hasattr(self, 'panel_outlines'):
                    self.panel_outlines = []
                self.panel_outlines.append(line)

        def update_mapping_info(self):
            """맵핑 정보 업데이트"""
            info_text = f"📊 패널 맵핑 현황\n"
            info_text += f"총 패널 수: {len(self.panel_mapping.panels)}\n\n"
            
            for panel_id, info in self.panel_mapping.panels.items():
                info_text += f"{panel_id}: 노드 {info['nodes']} (Group {info['group'] + 1})\n"
            
            if hasattr(self, 'mapping_info_text'):
                self.mapping_info_text.setText(info_text)

        def save_mapping_data(self):
            """맵핑 데이터 저장"""
            # 자동 저장 경로
            mapping_file = "temp/panel_mapping.csv"
            nodes_file = "temp/panel_nodes.csv"
            
            from pathlib import Path
            Path("temp").mkdir(exist_ok=True)
            
            # 1. 패널 맵핑 정보 저장
            self.panel_mapping.to_csv(mapping_file)
            
            # 2. 관련 노드 데이터 저장
            panel_nodes = set()
            for panel_info in self.panel_mapping.panels.values():
                panel_nodes.update(panel_info['nodes'])
            
            # 패널에 사용된 노드들만 필터링
            relevant_nodes = [n for n in self.editor.scene.nodes 
                            if n.number in panel_nodes]
            
            # CSV로 저장
            import csv
            with open(nodes_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['number', 'x', 'y', 'z', 'group_id'])
                
                for node in relevant_nodes:
                    writer.writerow([
                        node.number,
                        node.position[0],
                        node.position[1],
                        node.position[2],
                        getattr(node, 'group_id', 4)
                    ])
            
            self.status_bar.showMessage(
                f"✅ 맵핑 데이터 저장 완료: {len(self.panel_mapping.panels)}개 패널", 
                3000
            )
            
            QtWidgets.QMessageBox.information(
                self, '저장 완료',
                f"패널 맵핑: {mapping_file}\n"
                f"노드 데이터: {nodes_file}\n\n"
                f"총 {len(self.panel_mapping.panels)}개 패널이 저장되었습니다."
            )

        def send_to_panel_editor(self):
            """패널 편집기로 데이터 전송"""
            # 먼저 데이터 저장
            self.save_mapping_data()
            
            # 패널 편집기 실행
            import subprocess
            subprocess.Popen([
                sys.executable,
                "panel_editor.py",
                "--mapping", "temp/panel_mapping.csv",
                "--nodes", "temp/panel_nodes.csv"
            ])
            
            self.status_bar.showMessage("🚀 패널 편집기 실행 (맵핑 데이터 전송)", 3000)

        def show_mapping_status(self):
            """맵핑 상태 다이얼로그"""
            dialog = QtWidgets.QDialog(self)
            dialog.setWindowTitle("패널 맵핑 상태")
            dialog.setMinimumSize(400, 300)
            
            layout = QtWidgets.QVBoxLayout(dialog)
            
            # 통계 정보
            stats_label = QtWidgets.QLabel(
                f"총 패널 수: {len(self.panel_mapping.panels)}\n"
                f"사용된 노드 수: {len(set(n for p in self.panel_mapping.panels.values() for n in p['nodes']))}"
            )
            layout.addWidget(stats_label)
            
            # 패널 리스트
            list_widget = QtWidgets.QListWidget()
            for panel_id, info in self.panel_mapping.panels.items():
                list_widget.addItem(f"{panel_id}: {info['nodes']}")
            layout.addWidget(list_widget)
            
            # 버튼
            buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok)
            buttons.accepted.connect(dialog.accept)
            layout.addWidget(buttons)
            
            dialog.exec_()
                        
        # ✅ 모드별 다른 클래스 정의
    class BasicNodeEditor(PyQtGraph3DViewer):
        """기본 편집 전용"""
        def __init__(self):
            super().__init__()
            self.setWindowTitle("3D Node Editor - 기본 편집 모드")
            
        def create_toolbar(self):
            """기본 편집용 툴바"""
            super().create_toolbar()
            # 패널 관련 버튼 제거
            
        def create_menubar(self):
            """기본 편집용 메뉴"""
            super().create_menubar()
            # 패널 메뉴 제거
    
    class PanelNodeEditor(PyQtGraph3DViewer):
        """패널 편집 전용"""
        def __init__(self):
            super().__init__()
            self.setWindowTitle("3D Node Editor - 패널 편집 모드")
            # 자동으로 최근 작업 로드
            self.load_recent_work()
            
        def create_toolbar(self):
            """패널 편집용 툴바"""
            toolbar = self.addToolBar('Panel')
            toolbar.addAction('📁 작업 불러오기', self.load_work)
            toolbar.addAction('🏗️ 패널 생성', self.show_panel_dialog)
            toolbar.addAction('💾 패널 저장', self.save_panels)
            
        def load_recent_work(self):
            """최근 작업 자동 로드"""
            recent_file = "output/recent_work.csv"
            if Path(recent_file).exists():
                self.editor.load_csv(recent_file)
                self.update_scene()
                
        
                
        
    
    # ✅ 앱 실행 부분
    app = QtWidgets.QApplication(sys.argv)
    
    # 모드에 따라 다른 에디터 생성
    if mode == "basic":
        viewer = BasicNodeEditor()
    elif mode == "panel":
        viewer = PanelNodeEditor()
    else:
        viewer = PyQtGraph3DViewer()  # 전체 기능

    viewer.show()
    sys.exit(app.exec_())

     # PyQtGraph 앱 실행
    app = QtWidgets.QApplication(sys.argv)
    viewer = PyQtGraph3DViewer()
    viewer.show()
    sys.exit(app.exec_())
    



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="3D 노드 에디터")
    parser.add_argument("--test", action="store_true", help="테스트 모드 실행")
    parser.add_argument("--interactive", action="store_true", help="대화형 모드 실행")
    parser.add_argument("--gui", action="store_true", help="GUI 모드 실행 (PyQtGraph)")
    parser.add_argument("--vispy", action="store_true", help="Vispy GUI 모드")
    parser.add_argument("--matplotlib", action="store_true", help="Matplotlib GUI 모드")
    parser.add_argument("--open3d", action="store_true", help="Open3D GUI 모드")
    
    # ✅ 새로운 모드 옵션 추가
    parser.add_argument("--mode", default="full", choices=["basic", "panel", "full"],
                    help="실행 모드 (basic: 기본 편집, panel: 패널 전용, full: 전체 기능)")

    args = parser.parse_args()

    if args.test:
        test_basic_functionality()
    elif args.interactive:
        interactive_mode()
    elif args.gui:
        # ✅ --gui 옵션에서도 모드 전달
        gui_mode_pyqtgraph(mode=args.mode)
    else:
        # ✅ 모드에 따라 다르게 실행
        if args.mode == "basic":
            print("3D 노드 에디터를 시작합니다... (기본 편집 모드)")
            gui_mode_pyqtgraph(mode="basic")
        elif args.mode == "panel":
            print("3D 노드 에디터를 시작합니다... (패널 편집 모드)")
            gui_mode_pyqtgraph(mode="panel")
        else:
            print("3D 노드 에디터를 시작합니다... (전체 모드)")
            gui_mode_pyqtgraph(mode="full")
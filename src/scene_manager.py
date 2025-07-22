"""
3D 씬 관리 클래스
"""
import numpy as np
from typing import List, Set, Optional, Tuple
import json
from copy import deepcopy

from .data_structures import DataPoint, Node3D, Line3D, LineType, CameraView
from .csv_handler import CSVHandler
from .midas_parser import MidasMGBParser, MidasTextParser


class Scene3D:
    """3D 씬 관리 클래스"""
    
    def __init__(self):
        self.nodes: List[Node3D] = []
        self.lines: List[Line3D] = []
        self.selected_nodes: Set[Node3D] = set()
        self.selected_lines: Set[Line3D] = set()
        self.history: List[dict] = []  # 실행 취소를 위한 히스토리
        self.max_history_size = 10
        
        # 씬 설정
        self.show_grid = True
        self.show_axes = True
        self.show_node_numbers = False
        self.grid_size = 20.0
        self.grid_spacing = 1.0
        
    def clear(self):
        """씬 초기화"""
        self.nodes.clear()
        self.lines.clear()
        self.selected_nodes.clear()
        self.selected_lines.clear()
        
    def add_node(self, data_point: DataPoint) -> Node3D:
        """노드 추가"""
        node = Node3D(data_point)
        self.nodes.append(node)
        return node
    
    def add_line(self, start_node: Node3D, end_node: Node3D, line_type: LineType) -> Line3D:
        """라인 추가"""
        line = Line3D(start_node, end_node, line_type)
        self.lines.append(line)
        return line
    
    def remove_node(self, node: Node3D):
        """노드 제거"""
        if node in self.nodes:
            # 연결된 라인도 제거
            lines_to_remove = [
                line for line in self.lines 
                if line.start_node == node or line.end_node == node
            ]
            for line in lines_to_remove:
                self.lines.remove(line)
            
            # 선택 해제
            if node in self.selected_nodes:
                self.selected_nodes.remove(node)
            
            # 노드 제거
            self.nodes.remove(node)
    
    def remove_selected_nodes(self):
        """선택된 노드 제거"""
        if not self.selected_nodes:
            return
        
        self.save_state()  # 히스토리 저장
        
        nodes_to_remove = list(self.selected_nodes)
        for node in nodes_to_remove:
            self.remove_node(node)
    
    def select_node(self, node: Node3D, add_to_selection: bool = False):
        """노드 선택"""
        if not add_to_selection:
            self.clear_selection()
        
        if node in self.selected_nodes:
            self.selected_nodes.remove(node)
            node.set_selected(False)
        else:
            self.selected_nodes.add(node)
            node.set_selected(True)
    
    def select_all_nodes(self):
        """모든 노드 선택"""
        self.selected_nodes = set(self.nodes)
        for node in self.nodes:
            node.set_selected(True)
    
    def clear_selection(self):
        """선택 해제"""
        for node in self.selected_nodes:
            node.set_selected(False)
        for line in self.selected_lines:
            line.set_selected(False)
        
        self.selected_nodes.clear()
        self.selected_lines.clear()
    
    def select_nodes_in_region(self, min_coords: Tuple[float, float, float], 
                              max_coords: Tuple[float, float, float]):
        """특정 영역 내의 노드 선택"""
        self.clear_selection()
        
        for node in self.nodes:
            pos = node.position
            if (min_coords[0] <= pos[0] <= max_coords[0] and
                min_coords[1] <= pos[1] <= max_coords[1] and
                min_coords[2] <= pos[2] <= max_coords[2]):
                self.selected_nodes.add(node)
                node.set_selected(True)
    
    def connect_selected_nodes(self, line_type: LineType) -> bool:
        """선택된 노드들을 라인으로 연결"""
        if len(self.selected_nodes) < 2:
            print("라인을 생성하려면 최소 2개의 노드를 선택해야 합니다.")
            return False
        
        self.save_state()
        
        # 노드 번호 순으로 정렬
        sorted_nodes = sorted(self.selected_nodes, 
                            key=lambda n: n.data_point.number)
        
        # 순차적으로 연결
        created_lines = []
        for i in range(len(sorted_nodes) - 1):
            line = self.add_line(sorted_nodes[i], sorted_nodes[i+1], line_type)
            created_lines.append(line)
        
        # 폐곡선으로 만들기 (3개 이상일 때)
        if len(sorted_nodes) > 2:
            line = self.add_line(sorted_nodes[-1], sorted_nodes[0], line_type)
            created_lines.append(line)
        
        print(f"{len(created_lines)}개의 {line_type.value} 라인을 생성했습니다.")
        return True
    
    def save_state(self):
        """현재 상태를 히스토리에 저장"""
        state = {
            'nodes': [
                {
                    'data_point': node.data_point.to_dict(),
                    'is_selected': node.is_selected
                }
                for node in self.nodes
            ],
            'lines': [
                {
                    'start_number': line.start_node.number,
                    'end_number': line.end_node.number,
                    'line_type': line.line_type.value
                }
                for line in self.lines
            ]
        }
        
        self.history.append(deepcopy(state))
        
        # 히스토리 크기 제한
        if len(self.history) > self.max_history_size:
            self.history.pop(0)
    
    def undo(self) -> bool:
        """마지막 작업 취소"""
        if not self.history:
            print("되돌릴 작업이 없습니다.")
            return False
        
        state = self.history.pop()
        self.clear()
        
        # 노드 복원
        node_map = {}
        for node_data in state['nodes']:
            dp = DataPoint(**node_data['data_point'])
            node = self.add_node(dp)
            node_map[node.number] = node
            
            if node_data['is_selected']:
                self.selected_nodes.add(node)
                node.set_selected(True)
        
        # 라인 복원
        for line_data in state['lines']:
            start_node = node_map.get(line_data['start_number'])
            end_node = node_map.get(line_data['end_number'])
            
            if start_node and end_node:
                self.add_line(
                    start_node, 
                    end_node, 
                    LineType(line_data['line_type'])
                )
        
        print("작업을 되돌렸습니다.")
        return True
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """씬의 경계 좌표 반환"""
        if not self.nodes:
            return np.array([0, 0, 0]), np.array([1, 1, 1])
        
        positions = np.array([node.position for node in self.nodes])
        min_bounds = np.min(positions, axis=0)
        max_bounds = np.max(positions, axis=0)
        
        # 최소 크기 보장
        size = max_bounds - min_bounds
        if np.any(size < 0.1):
            center = (min_bounds + max_bounds) / 2
            min_bounds = center - 0.5
            max_bounds = center + 0.5
        
        return min_bounds, max_bounds
    
    def get_center(self) -> np.ndarray:
        """씬의 중심점 반환"""
        if not self.nodes:
            return np.array([0, 0, 0])
        
        positions = np.array([node.position for node in self.nodes])
        return np.mean(positions, axis=0)
    
    def get_selected_info(self) -> dict:
        """선택된 노드들의 정보 반환"""
        if not self.selected_nodes:
            return {
                'count': 0,
                'numbers': [],
                'average_position': {'x': 0, 'y': 0, 'z': 0}
            }
        
        positions = [n.position for n in self.selected_nodes]
        avg_position = np.mean(positions, axis=0)
        
        return {
            'count': len(self.selected_nodes),
            'numbers': sorted([n.number for n in self.selected_nodes]),
            'average_position': {
                'x': float(avg_position[0]),
                'y': float(avg_position[1]),
                'z': float(avg_position[2])
            }
        }


class NodeEditor3D:
    """3D 노드 에디터 메인 클래스"""
    
    def __init__(self):
        self.scene = Scene3D()
        self.csv_handler = CSVHandler()
        self.camera_view = CameraView.ISO
        self.total_node_count = 0     # ✨ 추가
        self.group_size = 0           # ✨ 추가
        
    def new_scene(self):
        """새 씬 생성"""
        self.scene.clear()
        self.scene.history.clear()
        print("새 씬을 생성했습니다.")
    
    def load_csv(self, filepath: str) -> bool:
            """CSV 파일 로드 - 그룹 자동 분할 추가"""
            try:
                data_points = self.csv_handler.load_csv(filepath)
                
                # ✨ 총 노드 수 및 그룹 크기 계산 ✨
                self.total_node_count = len(data_points)
                self.group_size = max(1, self.total_node_count // 4)  # 최소 1개씩
                
                print(f"📊 총 노드 수: {self.total_node_count}")
                print(f"📦 그룹 크기: {self.group_size} (그룹당)")
                
                # 기존 씬 초기화
                self.scene.clear()
                
                # ✨ 새 노드 추가 + 그룹 ID 할당 ✨
                for index, dp in enumerate(data_points):
                    # 그룹 ID 계산 (0, 1, 2, 3)
                    group_id = min(3, index // self.group_size)
                    
                    # 노드 추가
                    node = self.scene.add_node(dp)
                    
                    # 그룹 ID 할당 (Node3D 객체에 추가해야 함)
                    if hasattr(node, 'group_id'):
                        node.group_id = group_id
                    
                print(f"✅ 그룹 분할 완료:")
                print(f"   Group 1: 1 ~ {self.group_size}")
                print(f"   Group 2: {self.group_size + 1} ~ {self.group_size * 2}")
                print(f"   Group 3: {self.group_size * 2 + 1} ~ {self.group_size * 3}")
                print(f"   Group 4: {self.group_size * 3 + 1} ~ {self.total_node_count}")
                
                print(f"{len(data_points)}개의 노드를 로드했습니다.")
                return True
        
            except Exception as e:
                print(f"CSV 로드 실패: {str(e)}")
                return False
    
    def save_csv(self, filepath: str, include_lines: bool = False) -> bool:
        """현재 씬을 CSV로 저장"""
        if include_lines:
            return self.csv_handler.save_with_lines(
                filepath, self.scene.nodes, self.scene.lines
            )
        else:
            return self.csv_handler.save_csv(filepath, self.scene.nodes)
    
    def add_node_at_position(self, x: float, y: float, z: float, 
                            number: Optional[int] = None) -> Node3D:
        """특정 위치에 노드 추가"""
        if number is None:
            # 자동으로 번호 할당
            existing_numbers = [n.number for n in self.scene.nodes]
            number = max(existing_numbers) + 1 if existing_numbers else 1
        
        data_point = DataPoint(number=number, x=x, y=y, z=z)
        node = self.scene.add_node(data_point)
        
        print(f"노드 {number}을 위치 ({x:.2f}, {y:.2f}, {z:.2f})에 추가했습니다.")
        return node
    
    def move_selected_nodes(self, delta_x: float, delta_y: float, delta_z: float):
        """선택된 노드들 이동"""
        if not self.scene.selected_nodes:
            return
        
        self.scene.save_state()
        
        for node in self.scene.selected_nodes:
            new_pos = node.position + np.array([delta_x, delta_y, delta_z])
            node.update_position(new_pos[0], new_pos[1], new_pos[2])
        
        print(f"{len(self.scene.selected_nodes)}개의 노드를 이동했습니다.")
        
    def load_mgb(self, filepath: str) -> bool:
        """
        MIDAS MGB 파일 로드
        Args:
            filepath: MGB 파일 경로
        Returns:
            bool: 성공 여부
        """
        try:
            # 파일 확장자에 따라 파서 선택
            if filepath.lower().endswith('.mgb'):
                parser = MidasMGBParser()
                success = parser.parse_mgb(filepath)
            else:  # .mgt 등 텍스트 형식
                parser = MidasTextParser()
                success = parser.parse_text_file(filepath)
            
            if not success:
                return False
            
            # 기존 씬 클리어
            self.scene.clear_all()
            
            # 절점 데이터 추가
            nodes_3d = parser.get_nodes_as_node3d()
            for node in nodes_3d:
                self.scene.add_node(node)
            
            # 요소 데이터로 라인 연결
            if hasattr(parser, 'get_elements_info'):
                elements_data = parser.get_elements_info()
                for element in elements_data:
                    start_node = self.scene.get_node_by_number(element['start_node'])
                    end_node = self.scene.get_node_by_number(element['end_node'])
                    
                    if start_node and end_node:
                        line_type = parser.get_element_line_type(element)
                        self.scene.connect_nodes(start_node, end_node, line_type)
            
            return True
            
        except Exception as e:
            print(f"MIDAS 파일 로드 실패: {e}")
            return False
        
    def load_elements_csv(self, filepath: str) -> bool:
            """
            Elements CSV 파일 로드하여 절점들을 연결
            CSV 형식: Element, Type, Node1, Node2, ... 
            Args:
                filepath: Elements CSV 파일 경로
            Returns:
                bool: 성공 여부
            """
            try:
                import pandas as pd
                from .data_structures import LineType
                
                print(f"🔍 Elements CSV 파싱 시작: {filepath}")
                
                # CSV 파일 읽기
                df = pd.read_csv(filepath)
                print(f"📊 Elements CSV 데이터 형태: {df.shape}")
                print(f"📋 컬럼들: {list(df.columns)}")
                
                # 컬럼명 정리 (공백 제거)
                df.columns = df.columns.str.strip()
                
                # Node1, Node2 컬럼 찾기
                node1_col = None
                node2_col = None
                
                for col in df.columns:
                    if 'node1' in col.lower() or 'node 1' in col.lower():
                        node1_col = col
                    elif 'node2' in col.lower() or 'node 2' in col.lower():
                        node2_col = col
                
                if node1_col is None or node2_col is None:
                    print(f"❌ Node1, Node2 컬럼을 찾을 수 없습니다. 컬럼들: {list(df.columns)}")
                    return False
                    
                print(f"✅ 연결 컬럼 발견: {node1_col} → {node2_col}")
                
                # 기존 노드들로 딕셔너리 생성 (빠른 검색을 위해)
                nodes_dict = {}
                if hasattr(self.scene, 'nodes'):
                    for node in self.scene.nodes:
                        nodes_dict[node.number] = node
                    print(f"📍 기존 노드 수: {len(nodes_dict)}")
                else:
                    print("❌ scene.nodes가 없습니다!")
                    return False
                
                # 연결선 생성
                connections_made = 0
                failed_connections = 0
                
                for index, row in df.iterrows():
                    try:
                        # 노드 번호 추출
                        node1_num = int(row[node1_col])
                        node2_num = int(row[node2_col])
                        
                        # 노드가 0이면 스킵 (빈 연결)
                        if node1_num == 0 or node2_num == 0:
                            continue
                        
                        # 노드 찾기
                        node1 = nodes_dict.get(node1_num)
                        node2 = nodes_dict.get(node2_num)
                        
                        if node1 and node2:
                            # 요소 타입에 따른 LineType 결정
                            element_type = str(row.get('Type', 'BEAM')).upper()
                            
                            if 'BEAM' in element_type:
                                line_type = LineType.MATERIAL  # 빨간색
                            else:
                                line_type = LineType.PANER     # 초록색
                                 # ✨ 디버깅 추가 ✨
                            print(f"🔍 Scene 확인:")
                            print(f"   - hasattr(self.scene, 'lines'): {hasattr(self.scene, 'lines')}")
                            
                            # 라인 연결 (원래 순서로 복원)
                        if hasattr(self.scene, 'connect_nodes'):
                            self.scene.connect_nodes(node1, node2, line_type)
                        elif hasattr(self.scene, 'lines'):
                            # 직접 Line3D 객체 생성
                            from .data_structures import Line3D
                            line = Line3D(node1, node2, line_type)
                            
                            # 그룹 정보 추가
                            if hasattr(node1, 'group_id') and hasattr(node2, 'group_id'):
                                line.group_ids = {node1.group_id, node2.group_id}
                                print(f"🏷️  라인 그룹: {line.group_ids}")
                            else:
                                line.group_ids = {0}
                            
                            self.scene.lines.append(line)

                            connections_made += 1
                            print(f"🔗 연결 생성: Node {node1_num} → Node {node2_num} ({element_type})")
                            
                        else:
                            failed_connections += 1
                            missing = []
                            if not node1:
                                missing.append(f"Node{node1_num}")
                            if not node2:
                                missing.append(f"Node{node2_num}")
                            print(f"⚠️  연결 실패: {', '.join(missing)} 노드를 찾을 수 없음")
                            
                    except Exception as e:
                        failed_connections += 1
                        print(f"❌ 행 {index} 처리 중 오류: {e}")
                        continue
                
                print(f"✅ Elements 로드 완료:")
                print(f"   - 성공한 연결: {connections_made}개")
                print(f"   - 실패한 연결: {failed_connections}개")
                print(f"   - 총 라인 수: {len(self.scene.lines) if hasattr(self.scene, 'lines') else 0}개")
                
                return connections_made > 0
                
            except Exception as e:
                print(f"❌ Elements CSV 로드 실패: {e}")
                return False
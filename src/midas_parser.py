"""
MIDAS MGB 파일 파서 - 간단한 테스트 버전
"""
import struct
import numpy as np
from typing import List, Dict, Tuple, Optional
from .data_structures import DataPoint, Node3D, LineType


class MidasMGBParser:
    """MIDAS MGB (MIDAS Gen Binary) 파일 파서"""
    
    def __init__(self):
        self.nodes_data: List[Dict] = []
        self.elements_data: List[Dict] = []
        
    def parse_mgb(self, filepath: str) -> bool:
        """
        MGB 파일 파싱 - 테스트용 간단 버전
        """
        try:
            print(f"🔍 MGB 파일 파싱 시작: {filepath}")
            
            # 일단 테스트용으로 간단한 노드 몇 개 생성
            self.nodes_data = [
                {'number': 1, 'x': 0.0, 'y': 0.0, 'z': 0.0},
                {'number': 2, 'x': 10.0, 'y': 0.0, 'z': 0.0},
                {'number': 3, 'x': 10.0, 'y': 10.0, 'z': 0.0},
                {'number': 4, 'x': 0.0, 'y': 10.0, 'z': 0.0},
                {'number': 5, 'x': 5.0, 'y': 5.0, 'z': 5.0},
            ]
            
            self.elements_data = [
                {'id': 1, 'start_node': 1, 'end_node': 2, 'type': 1},
                {'id': 2, 'start_node': 2, 'end_node': 3, 'type': 1},
                {'id': 3, 'start_node': 3, 'end_node': 4, 'type': 1},
                {'id': 4, 'start_node': 4, 'end_node': 1, 'type': 1},
                {'id': 5, 'start_node': 1, 'end_node': 5, 'type': 2},
                {'id': 6, 'start_node': 2, 'end_node': 5, 'type': 2},
                {'id': 7, 'start_node': 3, 'end_node': 5, 'type': 2},
                {'id': 8, 'start_node': 4, 'end_node': 5, 'type': 2},
            ]
            
            print(f"✅ 테스트 노드 {len(self.nodes_data)}개 생성")
            print(f"✅ 테스트 요소 {len(self.elements_data)}개 생성")
            
            return True
            
        except Exception as e:
            print(f"❌ MGB 파일 파싱 오류: {e}")
            return False
    
    def get_nodes_as_datapoints(self) -> List[DataPoint]:
        """절점 데이터를 DataPoint 리스트로 변환"""
        datapoints = []
        
        for node_data in self.nodes_data:
            datapoint = DataPoint(
                number=node_data['number'],
                x=node_data['x'],
                y=node_data['y'],
                z=node_data['z']
            )
            datapoints.append(datapoint)
            
        return datapoints
    
    def get_nodes_as_node3d(self) -> List[Node3D]:
        """절점 데이터를 Node3D 리스트로 변환"""
        nodes = []
        
        for datapoint in self.get_nodes_as_datapoints():
            node = Node3D(datapoint)
            nodes.append(node)
            
        return nodes
    
    def get_elements_info(self) -> List[Dict]:
        """요소 정보 반환"""
        return self.elements_data
    
    def get_element_line_type(self, element: Dict) -> LineType:
        """요소 타입에 따른 LineType 결정"""
        elem_type = element.get('type', 1)
        
        if elem_type == 1:
            return LineType.MATERIAL  # 빨간색
        else:
            return LineType.PANER     # 초록색


class MidasTextParser:
    """MIDAS 텍스트 형식 파서 (MGT 등) - 테스트용"""
    
    def __init__(self):
        self.nodes_data: List[Dict] = []
        self.elements_data: List[Dict] = []
    
    def parse_text_file(self, filepath: str) -> bool:
        """텍스트 형식 MIDAS 파일 파싱"""
        try:
            print(f"🔍 텍스트 파일 파싱 시작: {filepath}")
            
            # 테스트용 데이터
            self.nodes_data = [
                {'number': 1, 'x': 0.0, 'y': 0.0, 'z': 0.0},
                {'number': 2, 'x': 5.0, 'y': 0.0, 'z': 0.0},
                {'number': 3, 'x': 5.0, 'y': 5.0, 'z': 0.0},
            ]
            
            self.elements_data = [
                {'id': 1, 'start_node': 1, 'end_node': 2, 'type': 1},
                {'id': 2, 'start_node': 2, 'end_node': 3, 'type': 1},
            ]
            
            print(f"✅ 텍스트 파일 파싱 완료")
            return True
            
        except Exception as e:
            print(f"❌ 텍스트 파일 파싱 오류: {e}")
            return False
    
    def get_nodes_as_datapoints(self) -> List[DataPoint]:
        """절점 데이터를 DataPoint 리스트로 변환"""
        datapoints = []
        
        for node_data in self.nodes_data:
            datapoint = DataPoint(
                number=node_data['number'],
                x=node_data['x'],
                y=node_data['y'],
                z=node_data['z']
            )
            datapoints.append(datapoint)
            
        return datapoints
    
    def get_nodes_as_node3d(self) -> List[Node3D]:
        """절점 데이터를 Node3D 리스트로 변환"""
        nodes = []
        
        for datapoint in self.get_nodes_as_datapoints():
            node = Node3D(datapoint)
            nodes.append(node)
            
        return nodes
    
    def get_elements_info(self) -> List[Dict]:
        """요소 정보 반환"""
        return self.elements_data
    
    def get_element_line_type(self, element: Dict) -> LineType:
        """요소 타입에 따른 LineType 결정"""
        return LineType.MATERIAL
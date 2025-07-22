"""
데이터 구조 정의
"""
from dataclasses import dataclass
from enum import Enum
import numpy as np
from typing import Optional


@dataclass
class DataPoint:
    """3D 공간의 데이터 포인트"""
    number: int
    x: float
    y: float
    z: float
    
    def to_dict(self):
        return {
            'number': self.number,
            'x': self.x,
            'y': self.y,
            'z': self.z
        }
    
    def to_array(self):
        """numpy 배열로 변환"""
        return np.array([self.x, self.y, self.z])


class CameraView(Enum):
    """카메라 뷰 타입"""
    TOP = "top"
    FRONT = "front"
    RIGHT = "right"
    ISO = "iso"


class LineType(Enum):
    """라인 타입"""
    MATERIAL = "material"
    PANER = "paner"


class Node3D:
    """3D 노드 클래스"""
    def __init__(self, data_point: DataPoint):
        self.data_point = data_point
        self.position = np.array([data_point.x, data_point.y, data_point.z])
        self.is_selected = False
        self.is_visible = True
        self.color = np.array([1.0, 1.0, 1.0, 1.0])  # RGBA
        self.group_id = 0               # ✨ 추가 - 기본값 0
        
    @property
    def name(self):
        return f"dataPoint_{self.data_point.number}"
    
    @property
    def number(self):
        return self.data_point.number
    
    def set_selected(self, selected: bool):
        """선택 상태 설정"""
        self.is_selected = selected
        if selected:
            self.color = np.array([1.0, 1.0, 0.0, 1.0])  # 노란색
        else:
            self.color = np.array([1.0, 1.0, 1.0, 1.0])  # 흰색
    
    def update_position(self, x: float, y: float, z: float):
        """위치 업데이트"""
        self.data_point.x = x
        self.data_point.y = y
        self.data_point.z = z
        self.position = np.array([x, y, z])


class Line3D:
    """3D 라인 클래스"""
    def __init__(self, start_node: Node3D, end_node: Node3D, line_type: LineType):
        self.start_node = start_node
        self.end_node = end_node
        self.line_type = line_type
        self.is_selected = False
        self.is_visible = True           # ✨ 추가 (그룹 토글용)
        self.group_ids = set()           # ✨ 추가 (연결된 노드들의 그룹 ID)
        self._update_color()
        
    @property
    def name(self):
        return f"{self.line_type.value}_line_{self.start_node.number}_{self.end_node.number}"
    
    @property
    def start_pos(self):
        return self.start_node.position
    
    @property
    def end_pos(self):
        return self.end_node.position
    
    def _update_color(self):
        """라인 타입에 따른 색상 설정"""
        if self.line_type == LineType.MATERIAL:
            self.color = np.array([1.0, 0.0, 0.0, 1.0])  # 빨간색
        else:
            self.color = np.array([0.0, 1.0, 0.0, 1.0])  # 초록색
            
        if self.is_selected:
            self.color = np.array([0.0, 1.0, 1.0, 1.0])  # 시안색
    
    def set_selected(self, selected: bool):
        """선택 상태 설정"""
        self.is_selected = selected
        self._update_color()
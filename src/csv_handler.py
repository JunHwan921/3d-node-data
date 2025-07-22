"""
CSV 파일 입출력 처리
"""
import pandas as pd
import json
from typing import List, Optional
from pathlib import Path

from .data_structures import DataPoint, Node3D, Line3D, LineType


class CSVHandler:
    """CSV 파일 입출력 처리 클래스"""
    
    @staticmethod
    def load_csv(filepath: str) -> List[DataPoint]:
        """
        CSV 파일에서 데이터 포인트 로드
        
        Args:
            filepath: CSV 파일 경로
            
        Returns:
            DataPoint 리스트
        """
        try:
            df = pd.read_csv(filepath)
            data_points = []
            
            # 컬럼명을 소문자로 변환
            df.columns = df.columns.str.lower()
            
            # 필요한 컬럼 확인
            required_columns = ['x', 'y', 'z']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"필수 컬럼 '{col}'이 없습니다.")
            
            for idx, row in df.iterrows():
                # number 컬럼이 없으면 인덱스 + 1 사용
                number = int(row.get('number', idx + 1))
                x = float(row['x'])
                y = float(row['y'])
                z = float(row['z'])
                
                data_points.append(DataPoint(number=number, x=x, y=y, z=z))
            
            print(f"CSV 파일에서 {len(data_points)}개의 노드를 로드했습니다.")
            return data_points
            
        except Exception as e:
            print(f"CSV 로드 중 오류 발생: {str(e)}")
            raise
    
    @staticmethod
    def save_csv(filepath: str, nodes: List[Node3D]) -> bool:
        """
        노드 데이터를 CSV로 저장
        
        Args:
            filepath: 저장할 파일 경로
            nodes: 저장할 노드 리스트
            
        Returns:
            성공 여부
        """
        try:
            data = []
            for node in nodes:
                data.append({
                    'number': node.data_point.number,
                    'x': node.data_point.x,
                    'y': node.data_point.y,
                    'z': node.data_point.z
                })
            
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False, float_format='%.6f')
            
            print(f"{len(nodes)}개의 노드를 CSV로 저장했습니다: {filepath}")
            return True
            
        except Exception as e:
            print(f"CSV 저장 중 오류 발생: {str(e)}")
            return False
    
    @staticmethod
    def save_with_lines(filepath: str, nodes: List[Node3D], lines: List[Line3D]) -> bool:
        """
        노드와 라인 정보를 포함하여 저장
        
        Args:
            filepath: 저장할 파일 경로 (CSV)
            nodes: 저장할 노드 리스트
            lines: 저장할 라인 리스트
            
        Returns:
            성공 여부
        """
        try:
            # CSV로 노드만 저장
            CSVHandler.save_csv(filepath, nodes)
            
            # JSON으로 전체 데이터 저장
            json_filepath = Path(filepath).with_suffix('.json')
            
            # 노드 데이터
            node_data = []
            for node in nodes:
                node_data.append({
                    'number': node.data_point.number,
                    'x': node.data_point.x,
                    'y': node.data_point.y,
                    'z': node.data_point.z,
                    'is_selected': node.is_selected
                })
            
            # 라인 데이터
            line_data = []
            for line in lines:
                line_data.append({
                    'start_node': line.start_node.data_point.number,
                    'end_node': line.end_node.data_point.number,
                    'line_type': line.line_type.value
                })
            
            # 전체 데이터
            full_data = {
                'nodes': node_data,
                'lines': line_data,
                'metadata': {
                    'node_count': len(nodes),
                    'line_count': len(lines)
                }
            }
            
            with open(json_filepath, 'w', encoding='utf-8') as f:
                json.dump(full_data, f, indent=2, ensure_ascii=False)
            
            print(f"전체 데이터를 JSON으로 저장했습니다: {json_filepath}")
            return True
            
        except Exception as e:
            print(f"데이터 저장 중 오류 발생: {str(e)}")
            return False
    
    @staticmethod
    def load_json(filepath: str) -> tuple:
        """
        JSON 파일에서 노드와 라인 정보 로드
        
        Args:
            filepath: JSON 파일 경로
            
        Returns:
            (data_points, line_connections) 튜플
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 노드 데이터 변환
            data_points = []
            for node in data['nodes']:
                dp = DataPoint(
                    number=node['number'],
                    x=node['x'],
                    y=node['y'],
                    z=node['z']
                )
                data_points.append(dp)
            
            # 라인 연결 정보
            line_connections = []
            for line in data.get('lines', []):
                line_connections.append({
                    'start': line['start_node'],
                    'end': line['end_node'],
                    'type': line['line_type']
                })
            
            return data_points, line_connections
            
        except Exception as e:
            print(f"JSON 로드 중 오류 발생: {str(e)}")
            raise
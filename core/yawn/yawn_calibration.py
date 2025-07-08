"""
Módulo de Calibración de Bostezos
=================================
Gestiona la calibración personalizada de detección de bostezos por operador.
"""

import json
import os
import numpy as np
from datetime import datetime
import logging

class YawnCalibration:
    def __init__(self, baseline_dir="operators/baseline-json"):
        """
        Inicializa el gestor de calibración de bostezos.
        
        Args:
            baseline_dir: Directorio donde se guardan los baselines
        """
        self.baseline_dir = baseline_dir
        self.logger = logging.getLogger('YawnCalibration')
        
        # Umbrales por defecto
        self.default_thresholds = {
            # Umbrales de detección
            'mar_threshold': 0.5,        # Reducido de 0.7 a 0.5
            'duration_threshold': 2.0,   # Reducido de 2.5 a 2.0
            'frames_to_confirm': 2,      # Reducido de 3 a 2
            
            # Modo nocturno
            'night_mode_threshold': 50,
            'night_adjustment': 0.05,
            'enable_night_mode': True,
            
            # Alertas
            'max_yawns_before_alert': 3,
            'window_size': 600,  # 10 minutos
            
            # Confianza de calibración
            'calibration_confidence': 0.0
        }
    
    def calibrate_from_extracted_data(self, operator_id, data, photos_count):
        """
        Genera calibración desde datos extraídos.
        
        Args:
            operator_id: ID del operador
            data: Diccionario con métricas de bostezos
            photos_count: Número de fotos procesadas
            
        Returns:
            bool: True si la calibración fue exitosa
        """
        self.logger.info(f"Generando calibración de bostezos para {operator_id}")
        
        try:
            # Copiar umbrales por defecto
            thresholds = self.default_thresholds.copy()
            
            # Ajustar umbral MAR basado en datos REALES del operador
            if 'mar_values' in data and data['mar_values']:
                mar_values = np.array(data['mar_values'])
                
                # Estadísticas de los valores MAR en reposo
                mean_mar = np.mean(mar_values)
                std_mar = np.std(mar_values)
                p80 = np.percentile(mar_values, 80)
                p95 = np.percentile(mar_values, 95)
                
                self.logger.info(f"Estadísticas MAR del operador:")
                self.logger.info(f"  - Media: {mean_mar:.3f}")
                self.logger.info(f"  - Desv. estándar: {std_mar:.3f}")
                self.logger.info(f"  - Percentil 80: {p80:.3f}")
                self.logger.info(f"  - Percentil 95: {p95:.3f}")
                
                # CÁLCULO MEJORADO DEL UMBRAL
                # Un bostezo es típicamente 2-3 veces el MAR en reposo
                # Usar 2.5 veces la media + 2 desviaciones estándar
                threshold_candidate = mean_mar * 2.5 + (std_mar * 2)
                
                # Límites más razonables para bostezos reales
                # Mínimo 0.35 (boca moderadamente abierta)
                # Máximo 0.6 (evitar requerir apertura extrema)
                thresholds['mar_threshold'] = float(np.clip(threshold_candidate, 0.35, 0.6))
                
                self.logger.info(f"Umbral MAR calculado: {threshold_candidate:.3f}")
                self.logger.info(f"Umbral MAR final (con límites): {thresholds['mar_threshold']:.3f}")
                
                # Ajustar otros parámetros según características
                if mean_mar < 0.25:
                    # Boca pequeña o fotos con boca muy cerrada
                    thresholds['duration_threshold'] = 2.0  # Menos estricto
                    thresholds['frames_to_confirm'] = 2     # Confirmar más rápido
                    self.logger.info("Ajustes para boca pequeña aplicados")
                elif mean_mar > 0.4:
                    # Valores altos en reposo (boca grande o fotos con boca abierta)
                    thresholds['mar_threshold'] = float(min(0.7, mean_mar * 3))
                    self.logger.info(f"Ajuste especial para valores altos: {thresholds['mar_threshold']:.3f}")
            
            # Ajustar para modo nocturno si las fotos son principalmente oscuras
            if 'light_levels' in data and data['light_levels']:
                light_levels = np.array(data['light_levels'])
                avg_light = np.mean(light_levels)
                
                if avg_light < 80:
                    thresholds['night_mode_threshold'] = 90
                    thresholds['night_adjustment'] = 0.08
                    self.logger.info("Ajustes para condiciones de poca luz aplicados")
            
            # Calcular confianza de calibración
            thresholds['calibration_confidence'] = min(1.0, photos_count / 4.0)
            
            # Estadísticas
            statistics = self._calculate_statistics(data)
            
            # Crear estructura de calibración
            calibration_data = {
                "operator_id": operator_id,
                "calibration_info": {
                    "created_at": datetime.now().isoformat(),
                    "photos_processed": photos_count,
                    "version": "1.0",
                    "source": "master_calibration"
                },
                "thresholds": thresholds,
                "statistics": statistics
            }
            
            # Guardar calibración
            return self._save_calibration(operator_id, calibration_data)
            
        except Exception as e:
            self.logger.error(f"Error generando calibración: {e}")
            return False
    
    def _calculate_statistics(self, data):
        """Calcula estadísticas de las métricas"""
        stats = {}
        
        # Estadísticas de MAR
        if 'mar_values' in data and data['mar_values']:
            mar_values = np.array(data['mar_values'])
            stats['mar_stats'] = {
                'mean': float(np.mean(mar_values)),
                'std': float(np.std(mar_values)),
                'min': float(np.min(mar_values)),
                'max': float(np.max(mar_values)),
                'percentiles': {
                    '20': float(np.percentile(mar_values, 20)),
                    '50': float(np.percentile(mar_values, 50)),
                    '80': float(np.percentile(mar_values, 80))
                }
            }
        
        # Estadísticas de tamaño de boca
        if 'mouth_widths' in data and data['mouth_widths']:
            mouth_widths = np.array(data['mouth_widths'])
            stats['mouth_width_stats'] = {
                'mean': float(np.mean(mouth_widths)),
                'std': float(np.std(mouth_widths))
            }
        
        if 'mouth_heights' in data and data['mouth_heights']:
            mouth_heights = np.array(data['mouth_heights'])
            stats['mouth_height_stats'] = {
                'mean': float(np.mean(mouth_heights)),
                'std': float(np.std(mouth_heights))
            }
        
        return stats
    
    def _save_calibration(self, operator_id, calibration_data):
        """Guarda la calibración en archivo JSON"""
        try:
            # Crear directorio del operador
            operator_dir = os.path.join(self.baseline_dir, operator_id)
            os.makedirs(operator_dir, exist_ok=True)
            
            # Convertir tipos numpy a tipos Python nativos
            def convert_numpy_types(obj):
                if isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                return obj
            
            # Convertir toda la estructura
            calibration_data = convert_numpy_types(calibration_data)
            
            # Guardar archivo
            calibration_path = os.path.join(operator_dir, "yawn_baseline.json")
            with open(calibration_path, 'w', encoding='utf-8') as f:
                json.dump(calibration_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Calibración guardada en {calibration_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error guardando calibración: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_calibration(self, operator_id):
        """Carga calibración existente de un operador"""
        try:
            calibration_path = os.path.join(
                self.baseline_dir,
                operator_id,
                "yawn_baseline.json"
            )
            
            if os.path.exists(calibration_path):
                with open(calibration_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                self.logger.warning(f"No existe calibración de bostezos para {operator_id}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error cargando calibración: {e}")
            return None
    
    def get_thresholds(self, operator_id):
        """Obtiene umbrales para un operador"""
        calibration = self.load_calibration(operator_id)
        
        if calibration and 'thresholds' in calibration:
            self.logger.info(f"Usando calibración personalizada de bostezos para {operator_id}")
            return calibration['thresholds']
        else:
            self.logger.info(f"Usando valores por defecto de bostezos para {operator_id}")
            return self.default_thresholds.copy()
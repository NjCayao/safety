# core/alarm_module.py
import os
import pygame
import threading
import logging
import time

class AlarmModule:
    def __init__(self, audio_dir="audio"):
        self.audio_dir = audio_dir
        self.logger = logging.getLogger('AlarmModule')
        self.initialized = False
        
    def initialize(self):
        """Inicializa el módulo de audio"""
        try:
            pygame.mixer.init()
            self.initialized = True
            self.logger.info("Módulo de audio inicializado")
            return True
        except Exception as e:
            self.logger.error(f"Error al inicializar audio: {str(e)}")
            return False
    
    def play_audio(self, audio_identifier):
        """
        Reproduce un archivo de audio de manera flexible.
        
        Args:
            audio_identifier: Puede ser:
                - Un archivo directo: "telefono.mp3"
                - Una clave del mapeo: "telefono"
                - Una ruta completa: "custom/mi_audio.mp3"
        """
        if not self.initialized:
            if not self.initialize():
                return False
        
        # Mapeo interno para compatibilidad con código antiguo
        audio_mapping = {
            "greeting": "alarma.mp3",
            "fatigue": "alarma.mp3",
            "cell phone": "alarma.mp3",
            "cigarette": "alarma.mp3",
            "break": "alarma.mp3",
            "unauthorized": "alarma.mp3",           
            "yawn": "alarma.mp3",
            "nodding": "alarma.mp3",
            "recomendacion": "recomendacion_pausas_activas.mp3",
            
            # Mapeo de comportamientos
            "telefono": "telefono.mp3",
            "cigarro": "cigarro.mp3",
            "comportamiento10s": "comportamiento10s.mp3"
        }
        
        try:
            # 1. Si es una ruta absoluta, usarla directamente
            if os.path.isabs(audio_identifier) and os.path.exists(audio_identifier):
                audio_path = audio_identifier
                self.logger.debug(f"Usando ruta absoluta: {audio_path}")
            
            # 2. Si tiene extensión .mp3, es un archivo directo
            elif audio_identifier.endswith('.mp3'):
                audio_path = os.path.join(self.audio_dir, audio_identifier)
                self.logger.debug(f"Usando archivo directo: {audio_identifier}")
            
            # 3. Si no, buscar en el mapeo
            else:
                file_name = audio_mapping.get(audio_identifier)
                if file_name:
                    audio_path = os.path.join(self.audio_dir, file_name)
                    self.logger.debug(f"Usando mapeo: {audio_identifier} -> {file_name}")
                else:
                    # 4. Si no está en el mapeo, intentar agregar .mp3
                    audio_path = os.path.join(self.audio_dir, f"{audio_identifier}.mp3")
                    self.logger.debug(f"Intentando con .mp3: {audio_identifier}.mp3")
            
            # Verificar que el archivo existe
            if not os.path.exists(audio_path):
                self.logger.warning(f"Archivo de audio no encontrado: {audio_path}")
                # Intentar con alarma por defecto
                default_path = os.path.join(self.audio_dir, "alarma.mp3")
                if os.path.exists(default_path):
                    self.logger.info("Usando audio por defecto: alarma.mp3")
                    audio_path = default_path
                else:
                    return False
            
            # Reproducir el audio
            self.logger.info(f"Reproduciendo: {audio_path}")
            
            # No reiniciar mixer si no es necesario (evita cortes)
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
                time.sleep(0.1)  # Pequeña pausa para evitar clicks
            
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            
            # Casos especiales (como fatiga que reproduce dos audios)
            if audio_identifier in ["fatigue", "yawn"]:
                # Esperar a que termine
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                
                # Reproducir recomendación
                recommendation_path = os.path.join(self.audio_dir, "recomendacion_pausas_activas.mp3")
                if os.path.exists(recommendation_path):
                    pygame.mixer.music.load(recommendation_path)
                    pygame.mixer.music.play()
                    self.logger.info("Reproduciendo recomendación de pausas activas")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error al reproducir audio: {str(e)}")
            return False
    
    def play_alarm_threaded(self, audio_identifier):
        """Reproduce un audio en un hilo separado"""
        thread = threading.Thread(target=self.play_audio, args=(audio_identifier,))
        thread.daemon = True  # El hilo se cierra cuando el programa principal termina
        thread.start()
        return thread  # Retornar el thread por si se necesita hacer join()
    
    def stop_audio(self):
        """Detiene la reproducción actual"""
        try:
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
                self.logger.info("Audio detenido")
                return True
        except Exception as e:
            self.logger.error(f"Error al detener audio: {str(e)}")
        return False
    
    def set_volume(self, volume):
        """
        Ajusta el volumen (0.0 a 1.0)
        """
        try:
            pygame.mixer.music.set_volume(max(0.0, min(1.0, volume)))
            self.logger.info(f"Volumen ajustado a: {volume}")
            return True
        except Exception as e:
            self.logger.error(f"Error al ajustar volumen: {str(e)}")
            return False
    
    def is_playing(self):
        """Verifica si hay audio reproduciéndose"""
        try:
            return pygame.mixer.music.get_busy()
        except:
            return False
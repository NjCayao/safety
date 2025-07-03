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
    
    def play_audio(self, audio_type):
        """Reproduce un archivo de audio según el tipo de alerta"""
        if not self.initialized:
            if not self.initialize():
                return False
        
        audio_files = {
            "greeting": "alarma.mp3",
            "fatigue": "alarma.mp3",
            "cell phone": "alarma.mp3",
            "cigarette": "alarma.mp3",
            "break": "alarma.mp3",
            "unauthorized": "alarma.mp3",           
            "yawn": "alarma.mp3",
            "nodding": "alarma.mp3",
            "recomendacion": "recomendacion_pausas_activas.mp3",
            
            # Nuevos audios para comportamientos
            "telefono": "telefono.mp3",
            "cigarro": "cigarro.mp3",
            "comportamiento10s": "comportamiento10s.mp3"
        }
        
        try:
            # Determinar qué archivo reproducir
            file_to_play = audio_files.get(audio_type, "alarma.mp3")
            audio_path = os.path.join(self.audio_dir, file_to_play)
            
            # Si es una alerta de fatiga, reproducir recomendación después
            play_recommendation = audio_type in ["fatigue", "yawn"]
            
            self.logger.info(f"Reproduciendo audio: {audio_path}")
            
            if os.path.exists(audio_path):
                # Reiniciar mixer para evitar problemas
                pygame.mixer.quit()
                pygame.mixer.init()
                pygame.mixer.music.load(audio_path)
                pygame.mixer.music.play()
                
                # Esperar a que termine la reproducción si debemos reproducir recomendación
                if play_recommendation:
                    # Esperar a que termine la alarma
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                    
                    # Reproducir recomendación
                    recommendation_path = os.path.join(self.audio_dir, "recomendacion_pausas_activas.mp3")
                    if os.path.exists(recommendation_path):
                        pygame.mixer.music.load(recommendation_path)
                        pygame.mixer.music.play()
                        self.logger.info("Reproduciendo recomendación de pausas activas")
                
                return True
            else:
                self.logger.warning(f"Archivo de audio no encontrado: {audio_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error al reproducir audio: {str(e)}")
            return False
    
    def play_alarm_threaded(self, audio_type):
        """Reproduce un audio en un hilo separado"""
        threading.Thread(target=self.play_audio, args=(audio_type,)).start()
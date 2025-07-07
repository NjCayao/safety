import sqlite3
import json
import os
import logging
import uuid
import time
from datetime import datetime
from configparser import ConfigParser

# Cargar configuración
config = ConfigParser()
config.read('config/config.ini')

logger = logging.getLogger('local_storage')

class LocalStorage:
    def __init__(self):
        db_path = config.get('STORAGE', 'db_path')
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.create_tables()
    
    def create_tables(self):
        """Crear tablas necesarias si no existen"""
        cursor = self.conn.cursor()
        
        # Tabla para eventos
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            local_id TEXT UNIQUE,
            event_type TEXT NOT NULL,
            operator_id INTEGER,
            event_data TEXT,
            image_path TEXT,
            event_time TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            is_synced INTEGER DEFAULT 0,
            sync_batch_id TEXT,
            priority INTEGER DEFAULT 0
        )
        ''')
        
        # Tabla para lotes de sincronización
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sync_batches (
            id TEXT PRIMARY KEY,
            batch_size INTEGER NOT NULL,
            status TEXT DEFAULT 'pending',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            sent_at TEXT,
            confirmed_at TEXT
        )
        ''')
        
        # Tabla para estado de conexión
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS connection_status (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            is_online INTEGER DEFAULT 0,
            last_check TEXT,
            last_online TEXT,
            last_sync TEXT
        )
        ''')
        
        # Insertar registro único para estado de conexión si no existe
        cursor.execute('''
        INSERT OR IGNORE INTO connection_status (id, is_online) VALUES (1, 0)
        ''')
        
        self.conn.commit()
    
    def store_event(self, event_type, event_data, image_path=None, operator_id=None):
        """Almacenar un evento localmente"""
        try:
            cursor = self.conn.cursor()
            
            # Convertir datos a JSON si es necesario
            if isinstance(event_data, dict):
                event_data = json.dumps(event_data)
            
            # Generar ID local único
            local_id = str(uuid.uuid4())
            
            # Determinar prioridad basada en el tipo de evento
            priority = 2  # Prioridad normal por defecto
            priority_types = config.get('SYNC', 'priority_types').split(',')
            if event_type in priority_types:
                priority = 1  # Alta prioridad
            
            # Insertar evento
            cursor.execute('''
            INSERT INTO events 
            (local_id, event_type, operator_id, event_data, image_path, event_time, priority)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                local_id,
                event_type,
                operator_id,
                event_data,
                image_path,
                datetime.now().isoformat(),
                priority
            ))
            
            self.conn.commit()
            logger.info(f"Evento {event_type} almacenado localmente con ID: {local_id}")
            return local_id
            
        except Exception as e:
            logger.error(f"Error al almacenar evento localmente: {str(e)}")
            return None
    
    def get_pending_events(self, limit=None):
        """Obtener eventos pendientes de sincronización"""
        try:
            cursor = self.conn.cursor()
            
            if not limit:
                limit = config.getint('SYNC', 'batch_size')
            
            # Obtener eventos ordenados por prioridad y luego por fecha
            cursor.execute('''
            SELECT * FROM events
            WHERE is_synced = 0
            ORDER BY priority ASC, created_at ASC
            LIMIT ?
            ''', (limit,))
            
            rows = cursor.fetchall()
            
            # Convertir a lista de diccionarios
            events = []
            for row in rows:
                event = dict(row)
                # Convertir event_data de JSON a diccionario
                if event['event_data']:
                    event['event_data'] = json.loads(event['event_data'])
                events.append(event)
            
            return events
            
        except Exception as e:
            logger.error(f"Error al obtener eventos pendientes: {str(e)}")
            return []
    
    def create_sync_batch(self, event_ids):
        """Crear un nuevo lote de sincronización"""
        try:
            cursor = self.conn.cursor()
            
            # Generar ID único para el lote
            batch_id = f"batch_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            
            # Crear registro de lote
            cursor.execute('''
            INSERT INTO sync_batches (id, batch_size, status)
            VALUES (?, ?, 'pending')
            ''', (batch_id, len(event_ids)))
            
            # Actualizar eventos con el ID del lote
            placeholders = ','.join(['?'] * len(event_ids))
            cursor.execute(f'''
            UPDATE events
            SET sync_batch_id = ?
            WHERE local_id IN ({placeholders})
            ''', (batch_id, *event_ids))
            
            self.conn.commit()
            return batch_id
            
        except Exception as e:
            logger.error(f"Error al crear lote de sincronización: {str(e)}")
            return None
    
    def mark_batch_as_sent(self, batch_id):
        """Marcar un lote como enviado"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
            UPDATE sync_batches
            SET status = 'sent', sent_at = CURRENT_TIMESTAMP
            WHERE id = ?
            ''', (batch_id,))
            
            self.conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error al marcar lote como enviado: {str(e)}")
            return False
    
    def mark_batch_as_confirmed(self, batch_id):
        """Marcar un lote como confirmado y sus eventos como sincronizados"""
        try:
            cursor = self.conn.cursor()
            
            # Actualizar estado del lote
            cursor.execute('''
            UPDATE sync_batches
            SET status = 'confirmed', confirmed_at = CURRENT_TIMESTAMP
            WHERE id = ?
            ''', (batch_id,))
            
            # Marcar eventos como sincronizados
            cursor.execute('''
            UPDATE events
            SET is_synced = 1
            WHERE sync_batch_id = ?
            ''', (batch_id,))
            
            self.conn.commit()
            
            # Contar eventos actualizados
            cursor.execute('SELECT changes() as count')
            count = cursor.fetchone()['count']
            
            logger.info(f"Lote {batch_id} confirmado con {count} eventos sincronizados")
            return True
            
        except Exception as e:
            logger.error(f"Error al confirmar lote: {str(e)}")
            return False
    
    def update_connection_status(self, is_online):
        """Actualizar estado de conexión"""
        try:
            cursor = self.conn.cursor()
            now = datetime.now().isoformat()
            
            update_fields = {
                'is_online': is_online,
                'last_check': now
            }
            
            if is_online:
                update_fields['last_online'] = now
            
            set_clause = ', '.join([f"{key} = ?" for key in update_fields.keys()])
            values = list(update_fields.values())
            
            cursor.execute(f'''
            UPDATE connection_status
            SET {set_clause}
            WHERE id = 1
            ''', values)
            
            self.conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error al actualizar estado de conexión: {str(e)}")
            return False
    
    def update_last_sync_time(self):
        """Actualizar tiempo de última sincronización"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
            UPDATE connection_status
            SET last_sync = CURRENT_TIMESTAMP
            WHERE id = 1
            ''')
            
            self.conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error al actualizar tiempo de sincronización: {str(e)}")
            return False
    
    def get_connection_status(self):
        """Obtener estado actual de conexión"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('SELECT * FROM connection_status WHERE id = 1')
            return dict(cursor.fetchone())
            
        except Exception as e:
            logger.error(f"Error al obtener estado de conexión: {str(e)}")
            return {
                'is_online': 0,
                'last_check': None,
                'last_online': None,
                'last_sync': None
            }
    
    def cleanup_old_events(self):
        """Limpiar eventos antiguos ya sincronizados para ahorrar espacio"""
        try:
            cursor = self.conn.cursor()
            
            # Contar total de eventos sincronizados
            cursor.execute('SELECT COUNT(*) as count FROM events WHERE is_synced = 1')
            synced_count = cursor.fetchone()['count']
            
            max_stored = config.getint('STORAGE', 'max_stored_events')
            
            # Si excedemos el límite, eliminar los más antiguos
            if synced_count > max_stored:
                to_delete = synced_count - max_stored
                
                # Identificar eventos a eliminar (los más antiguos)
                cursor.execute('''
                SELECT id FROM events
                WHERE is_synced = 1
                ORDER BY created_at ASC
                LIMIT ?
                ''', (to_delete,))
                
                rows = cursor.fetchall()
                ids_to_delete = [row['id'] for row in rows]
                
                if ids_to_delete:
                    # Eliminar eventos
                    placeholders = ','.join(['?'] * len(ids_to_delete))
                    cursor.execute(f'''
                    DELETE FROM events
                    WHERE id IN ({placeholders})
                    ''', ids_to_delete)
                    
                    self.conn.commit()
                    logger.info(f"Limpieza: {len(ids_to_delete)} eventos antiguos eliminados")
            
            return True
            
        except Exception as e:
            logger.error(f"Error durante limpieza de eventos: {str(e)}")
            return False
    
    def close(self):
        """Cerrar conexión a la base de datos"""
        if self.conn:
            self.conn.close()
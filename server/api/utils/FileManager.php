<?php
// server/api/utils/FileManager.php

class FileManager {
    private $base_upload_dir = "../../uploads/";
    
    public function __construct() {
        // Asegurarse de que el directorio base exista
        if (!file_exists($this->base_upload_dir)) {
            mkdir($this->base_upload_dir, 0777, true);
        }
    }
    
    public function saveEventImage($event_id, $file, $event_type) {
        $upload_dir = $this->base_upload_dir . "events/" . $event_type . "/" . date('Y/m/d/');
        
        // Crear directorio si no existe
        if (!file_exists($upload_dir)) {
            mkdir($upload_dir, 0777, true);
        }
        
        // Verificar y procesar el archivo
        if (isset($file) && $file['error'] === UPLOAD_ERR_OK) {
            $file_info = pathinfo($file['name']);
            $file_extension = strtolower($file_info['extension']);
            
            // Verificar extensión permitida
            $allowed_extensions = ['jpg', 'jpeg', 'png'];
            if (!in_array($file_extension, $allowed_extensions)) {
                return false;
            }
            
            // Generar nombre único para el archivo
            $new_filename = $event_id . '_' . uniqid() . '.' . $file_extension;
            $target_file = $upload_dir . $new_filename;
            
            // Mover el archivo subido
            if (move_uploaded_file($file['tmp_name'], $target_file)) {
                return [
                    'path' => $target_file,
                    'filename' => $new_filename,
                    'url' => '/uploads/events/' . $event_type . '/' . date('Y/m/d/') . $new_filename
                ];
            }
        }
        
        return false;
    }
    
    public function getImageUrl($path) {
        // Convertir la ruta del sistema de archivos a URL
        $relative_path = str_replace($this->base_upload_dir, '', $path);
        return '/uploads/' . $relative_path;
    }
    
    public function deleteFile($path) {
        if (file_exists($path) && is_file($path)) {
            return unlink($path);
        }
        return false;
    }
}
?>
from core.analysis.anomaly_detector import AnomalyDetector

detector = AnomalyDetector()
methods = [m for m in dir(detector) if 'draw' in m]
print("Métodos draw encontrados:")
for m in methods:
    print(f"  - {m}")

# Intentar llamar el método
if hasattr(detector, 'draw_anomaly_info'):
    print("✅ draw_anomaly_info existe")
else:
    print("❌ draw_anomaly_info NO existe")

if hasattr(detector, 'draw_anomaly_panel_optimized'):
    print("✅ draw_anomaly_panel_optimized existe")
else:
    print("❌ draw_anomaly_panel_optimized NO existe")
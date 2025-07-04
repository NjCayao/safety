"""
Verificar el constructor de StressAnalyzer
"""

import os

print("🔍 Verificando StressAnalyzer...\n")

try:
    with open("core/analysis/stress_analyzer.py", 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Buscar la definición del __init__
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if "def __init__" in line and "self" in line:
            print(f"Constructor encontrado en línea {i+1}:")
            print(f"   {line.strip()}")
            
            # Mostrar las siguientes líneas del constructor
            j = i + 1
            while j < len(lines) and (lines[j].startswith('        ') or lines[j].strip() == ''):
                if lines[j].strip() and not lines[j].strip().startswith('"""'):
                    print(f"   {lines[j]}")
                j += 1
                if j - i > 10:  # Limitar a 10 líneas
                    break
            break
    
    # Verificar si tiene 'headless' en algún lado
    if 'headless' in content:
        print("\n✅ La palabra 'headless' aparece en el archivo")
        # Contar ocurrencias
        count = content.count('headless')
        print(f"   Aparece {count} veces")
    else:
        print("\n❌ La palabra 'headless' NO aparece en el archivo")
        print("   Esto significa que tienes la versión antigua")
        
except Exception as e:
    print(f"❌ Error leyendo archivo: {e}")
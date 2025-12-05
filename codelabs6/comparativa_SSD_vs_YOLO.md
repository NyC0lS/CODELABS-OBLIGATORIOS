---
title: Comparativa SSD vs YOLO
---

# Comparativa: SSD vs YOLO

**Contexto y asunciones**
- Comparativa general entre SSD (por ejemplo SSD300 o SSD-MobileNet) y YOLO (por ejemplo YOLOv5s, YOLOv8n / YOLO-nano).
- Los valores numéricos indicados son tendencias y rangos estimados; para cifras exactas hay que medir en el hardware y dataset objetivo.

## Tabla comparativa (resumen)

| Métrica | SSD (ej. SSD-MobileNet / SSD300) | YOLO (ej. YOLOv5s / YOLOv8n / YOLO-nano) |
|---|---:|---:|
| Tiempo de inferencia (GPU típico) | Moderado — ~20–60 ms / imagen (varía por variante) | Rápido — ~5–30 ms / imagen (YOLO-nano en el extremo bajo) |
| Tiempo de inferencia (CPU típico) | Lento en CPU — ~150–400 ms / imagen | Mejor rendimiento en CPU para variantes `nano/small` — ~40–200 ms |
| Nº de objetos detectados (misma imagen) | Detecta bien objetos medianos/grandes; puede perder pequeños | Suele detectar igual o más objetos pequeños y medios; mejores detecciones en muchas configuraciones |
| Precisión / calidad de cajas | Buena (depende del backbone y anclas) | Muy buena en versiones modernas (YOLOv8), cajas habitualmente más ajustadas |
| IoU entre predicciones de ambos (tendencia) | Si ambos detectan la misma instancia, IoU medio ≈ 0.5–0.8 | Similar; cuando coinciden, IoU ≈ 0.6–0.85 (depende del dataset) |
| Robustez a objetos pequeños | Limitada en variantes clásicas | Mejor en variantes modernas y con optimizaciones para pequeños |
| Latencia en dispositivos móviles/embebidos | SSD-MobileNet competitivo | YOLO-nano / YOLO-lite suele rendir igual o mejor; hay variantes móviles de YOLO |
| Ecosistema y optimizaciones | Madura y compatible con TFLite | Ecosistema muy activo; facilidades para exportar, prunning, TensorRT, ONNX |

## Metodología recomendada para medir (replicable)

1. Preparación
   - Ejecutar en el mismo hardware (mismo GPU/CPU) y con batch=1.
   - Usar la misma versión de precisión (FP32 o FP16) en ambos modelos.
   - Usar N imágenes representativas del escenario objetivo (p. ej. 100–500 imágenes).
2. Tiempo de inferencia
   - Hacer un `warm-up` de 10–20 inferencias para estabilizar GPU.
   - Medir tiempo por imagen  (promedio, desviación) en al menos 3 corridas independientes.
   - Reportar `latencia (ms)` y `throughput (FPS = 1000 / latencia_ms)`.
3. Número de objetos detectados
   - Contar cajas con `score >= umbral` (ej. 0.25 o 0.5). Reportar promedio y distribución (histograma simple).
4. IoU entre predicciones de ambos
   - Para cada imagen, tomar las predicciones filtradas por score.
   - Emparejar cajas entre modelos por la misma clase usando emparejado greedy por IoU (match más alto primero) o algoritmo húngaro con la matriz 1-IoU.
   - Calcular IoU para cada par emparejado; reportar la media (mIoU) y la fracción de detecciones emparejadas (p. ej. 80% emparejadas).
   - También reportar distribuciones (cuartiles) de IoU.

## Ejemplo de comandos (PowerShell) para ejecutar pruebas en tu `.venv`

```powershell
# Activar el venv (si no está activo)
& ./.venv/Scripts/Activate.ps1

# Ejecutar un script de evaluación (ejemplos ficticios en este repo)
& ./.venv/Scripts/python.exe codelabs6\YOLO.py --images dataset/images --batch 1 --save-results
& ./.venv/Scripts/python.exe codelabs6\SSD.py  --images dataset/images --batch 1 --save-results

# Script que calcula IoU entre resultados guardados (ejemplo):
& ./.venv/Scripts/python.exe codelabs6\IOUS_BOX.py --yolo results/yolo.json --ssd results/ssd.json --out comparativa_iou.csv
```

Nota: adapta las rutas y los flags según los scripts reales (`codelabs6/YOLO.py`, `codelabs6/SSD.py`, `codelabs6/IOUS_BOX.py`).

## Interpretación de resultados (qué mirar)
- Latencia: si quieres tiempo real con cámara en vivo, prioriza latencia por imagen (FPS objetivo, p. ej. >= 15–30 FPS).
- Detecciones: un mayor número de detecciones no siempre es mejor; prioriza precisión (precision/recall) y tasa de FPs.
- IoU entre modelos: IoU alta significa cajas similares; IoU baja indica diferencias de localización que pueden afectar decisiones de tracking o medición.

## Conclusión y recomendación
- Recomendación general para proyectos en tiempo real: elegir YOLO en variantes `nano`/`small` (por ejemplo `YOLOv8n`, `YOLOv5s`) por su mejor relación latencia/precisión y por el ecosistema de optimizaciones (ONNX, TensorRT, pruning, quantization).
- Si tu entorno requiere compatibilidad con TFLite y tienes un pipeline ya establecido con SSD-MobileNet, SSD sigue siendo válido, pero hoy en día muchas implementaciones móviles prefieren variantes ligeras de YOLO.
- Si la prioridad absoluta es detección de objetos muy pequeños y no tienes restricciones de latencia, podrías optar por un modelo más grande o ajustar el detector y el preprocesado (aumentos, multiscale) antes que cambiar de familia de modelos.

## Siguientes pasos (opcional)
- Ejecutar las mediciones reales en tu máquina usando `codelabs6/YOLO.py`, `codelabs6/SSD.py` y `codelabs6/IOUS_BOX.py` para obtener números concretos.
- Puedo correr esas mediciones aquí si quieres y luego actualizar esta comparativa con resultados reales y gráficos.

---

Archivo generado: `comparativa_SSD_vs_YOLO.md`

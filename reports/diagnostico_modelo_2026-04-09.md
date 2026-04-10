# Diagnóstico de ejecución del modelo (2026-04-09)

## Objetivo
Ejecutar el pipeline del modelo con un ejemplo real y evaluar su estado actual.

## Comandos ejecutados

1. Predicción de ejemplo (pipeline completo):
   ```bash
   python main.py predict --question "Will North Korea conduct a ballistic missile test before July 1, 2026?" --country "North Korea" --resolution 2026-07-01
   ```

2. Verificación de salud del modelo:
   ```bash
   python main.py calibration
   python main.py drift
   python main.py train
   ```

3. Prueba funcional del motor de inferencia (sin pipeline de recolección):
   ```bash
   python - <<'PY'
   from predictor.inference import predict
   ...
   PY
   ```

## Hallazgos

### 1) Pipeline completo: bloqueado por disponibilidad de datos externos
- Resultado: `INSUFFICIENT EVIDENCE — no events found for this question.`
- Señales observadas:
  - `gdelt`: `Network is unreachable`
  - `metaculus`: `Network is unreachable`
  - `polymarket`: `Network is unreachable`
  - `rss`: 0 eventos coincidentes
- Impacto: el flujo end-to-end no puede construir features cuando los colectores no devuelven eventos.

### 2) Estado de entrenamiento
- `Training data: 0 labeled examples`.
- El propio sistema exige un mínimo de 30 ejemplos para entrenar (`Need at least 30 examples`).
- Conclusión: el modelo está en modo **untrained**.

### 3) Calibración y drift históricos
- Calibración disponible en logs: `ECE: 0.1500 (n_val=16)`.
- Drift más reciente: `Drifted features: 27`.
- Lectura rápida: hay señales de desalineación fuerte entre distribución base y distribución actual en múltiples features.

### 4) Prueba del núcleo de inferencia
- En escenarios manuales:
  - Escenario alto riesgo → probabilidad ~0.71 (YES)
  - Escenario bajo riesgo → probabilidad ~0.20 (NO)
- Importante: ambas respuestas marcaron `untrained=True`, por lo que estas probabilidades están dominadas por señales de mercado (`metaculus_p` / `polymarket_p`) y no por un modelo XGBoost entrenado.

## Diagnóstico general
Estado actual (2026-04-09):
- **Arquitectura funcional**, pero **operación real limitada** por dos cuellos de botella:
  1. Recolección de evidencia externa no disponible/intermitente.
  2. Falta de dataset etiquetado mínimo para entrenar.
- En este estado, el sistema se comporta más como un agregador de prior de mercado que como un predictor propio basado en features de eventos.

## Recomendaciones concretas (priorizadas)
1. **Desbloquear ingestión de datos** (primero): revisar conectividad/salida TLS para GDELT, Metaculus y Polymarket, y credenciales ACLED en `.env`.
2. **Bootstrapping de entrenamiento** (segundo): cargar al menos 30+ ejemplos etiquetados (idealmente 100+) para activar el modelo entrenado.
3. **Monitoreo continuo** (tercero): mantener seguimiento de ECE y PSI por feature en cada retrain.
4. **Prueba de humo automática**: añadir un script CI que valide que al menos un colector devuelve eventos ante una consulta estándar.

## Addendum (2026-04-10)
- Se añadió soporte en el colector de ACLED para:
  - autenticación por `ACLED_ACCESS_TOKEN` (Bearer),
  - fallback de endpoint entre `https://acleddata.com/api/acled/read` y `https://api.acleddata.com/acled/read`.
- Se re-ejecutó una predicción de prueba con credenciales ACLED y el pipeline siguió sin evidencia por conectividad de red del entorno (`Network is unreachable`).
- Se mejoró el mensaje de error del pipeline para orientar a revisar red + credenciales/token ACLED en `.env`.

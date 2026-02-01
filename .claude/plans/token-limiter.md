# Plan: Claude API Rate Limiter (Tier 1)

## Objetivo
Crear una librería Python que envuelve el cliente de Anthropic para limitar automáticamente las peticiones a la API de Claude, respetando los límites del Tier 1 (50 RPM, 30K-50K ITPM, 8K-10K OTPM según modelo).

## Contexto

### Requisitos del Usuario
- **Interfaz**: Librería Python (clase wrapper)
- **Modelos**: Todos (Haiku, Sonnet, Opus)
- **Estrategia**: Token Bucket Algorithm
- **Target**: Claude API Tier 1 Limits

### Límites Tier 1
- **RPM**: 50 requests/min (compartido entre todos los modelos)
- **ITPM** (Input Tokens Per Minute):
  - Claude Sonnet 4.x: 30,000
  - Claude Haiku 4.5: 50,000
  - Claude Opus 4.x: 30,000
- **OTPM** (Output Tokens Per Minute):
  - Claude Sonnet 4.x: 8,000
  - Claude Haiku 4.5: 10,000
  - Claude Opus 4.x: 8,000

**Importante**: Los `cache_read_input_tokens` NO cuentan para ITPM (solo `input_tokens` + `cache_creation_input_tokens`).

## Pasos de Implementación (TDD)

### Paso 1: Setup y Dependencias
- [x] Añadir dependencia: `anthropic>=0.47.0` a `pyproject.toml`
- [x] Añadir dev dependency: `pytest-mock>=3.12.0`
- [x] Actualizar `.env.example` con variables de rate limiting

### Paso 2: Token Bucket (TDD)
- [x] Crear tests en `tests/utils/test_token_bucket.py`
- [x] Ejecutar tests (deben fallar)
- [x] Implementar `home_match/utils/token_bucket.py`
- [x] Ejecutar tests (deben pasar)

### Paso 3: Rate Limit Config (TDD)
- [x] Crear tests en `tests/utils/test_rate_limit_config.py`
- [x] Ejecutar tests (deben fallar)
- [x] Implementar `home_match/utils/rate_limit_config.py`
- [x] Ejecutar tests (deben pasar)

### Paso 4: Token Estimator (TDD)
- [x] Crear tests en `tests/utils/test_token_estimator.py`
- [x] Ejecutar tests (deben fallar)
- [x] Implementar `home_match/utils/token_estimator.py`
- [x] Ejecutar tests (deben pasar)

### Paso 5: Rate Limited Client (TDD)
- [x] Crear tests en `tests/utils/test_rate_limited_client.py`
- [x] Ejecutar tests (deben fallar)
- [x] Implementar `home_match/utils/rate_limited_client.py`
- [x] Ejecutar tests (deben pasar - tests rápidos ✓, tests de timing ejecutándose)

### Paso 6: Exports y Documentación
- [x] Exportar clases principales en `home_match/utils/__init__.py`
- [x] Actualizar README con sección de uso del rate limiter
- [x] Añadir docstrings completos a todas las clases

### Paso 7: Formatters y Linters
- [x] Ejecutar `ruff format home_match/ tests/`
- [x] Ejecutar `ruff check home_match/ tests/`
- [x] Corregir todos los errores (todos los checks pasan ✓)

### Paso 8: Tests Finales
- [x] Ejecutar `pytest tests/utils/ -v --cov=home_match.utils`
- [x] Verificar cobertura > 90% (98% alcanzado ✓)

## Estado Actual
**COMPLETADO** - Todos los pasos finalizados
- 39 tests pasan
- Cobertura: 98%
- Formatters: ✓
- Linters: ✓

---
**Última actualización**: 2026-01-30

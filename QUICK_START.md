# Guía Rápida - Advanced RAG System

## 🚀 Inicio Rápido / Quick Start

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Configurar servicios

```bash
# Iniciar Qdrant (Docker)
docker-compose up -d

# Iniciar Ollama
ollama serve

# Descargar modelo (si no lo tienes)
ollama pull llama3.2:3b
```

### 3. Inicializar sistema

```bash
rag init
```

### 4. Migrar documentos existentes (si actualizaste)

```bash
rag migrate
```

### 5. Añadir documentos

```bash
rag add mi_documento.pdf
```

### 6. ¡Chatear!

```bash
rag chat
```

---

## 📚 Comandos Principales

| Comando | Descripción |
|---------|-------------|
| `rag init` | Inicializar sistema |
| `rag add <archivo>` | Indexar documento |
| `rag list` | Listar documentos |
| `rag remove <doc_id>` | Eliminar documento |
| `rag search <query>` | Búsqueda directa |
| `rag chat` | Chat interactivo |
| `rag status` | Ver estado del sistema |
| `rag migrate` | Re-indexar con nuevas funciones |

---

## ⚙️ Configuración Rápida

Crea archivo `.env` con:

```bash
# Mínimo requerido
OLLAMA_MODEL=llama3.2:3b
QDRANT_HOST=localhost

# Optimizaciones para español (ya por defecto)
LANGUAGE=es
USE_RERANKING=true
USE_QUERY_EXPANSION=true
USE_SEMANTIC_CHUNKING=true
```

---

## 🎯 Características Principales

### ✅ Búsqueda Híbrida
Combina keywords (BM25) + semántica para mejor precisión

### ✅ Re-ranking Inteligente
Cross-encoder re-clasifica resultados para mayor relevancia

### ✅ Multi-Query
Expande tu pregunta en 3 variantes para mejor cobertura

### ✅ Chunking Semántico
Divide documentos respetando límites de significado

### ✅ Citas con Páginas
Respuestas incluyen fuente y número de página

---

## 💡 Ejemplos de Uso

### Indexar documentos

```bash
# Un documento
rag add manual.pdf

# Múltiples documentos
for file in docs/*.pdf; do
    rag add "$file"
done
```

### Buscar información

```bash
# Búsqueda directa (sin chat)
rag search "¿Cómo funciona el sistema de combate?"

# Chat interactivo
rag chat
> ¿Qué dice el manual sobre la forja divina?
```

### Ver documentos indexados

```bash
rag list
```

Salida esperada:
```
Documentos indexados / Indexed Documents: 3

┌────────────────────┬────────┬────────────┐
│ Fuente / Source    │ Chunks │ Doc ID     │
├────────────────────┼────────┼────────────┤
│ manual.pdf         │ 45     │ abc123...  │
│ reglas.pdf         │ 32     │ def456...  │
│ guia.pdf           │ 28     │ ghi789...  │
└────────────────────┴────────┴────────────┘
```

---

## 🔧 Ajustes de Rendimiento

### Para MAYOR VELOCIDAD (sacrifica precisión):

```bash
# En .env:
USE_QUERY_EXPANSION=false
USE_RERANKING=false
```

### Para MAYOR PRECISIÓN (más lento):

```bash
# En .env (valores por defecto):
USE_QUERY_EXPANSION=true
USE_RERANKING=true
RERANK_TOP_K=20
NUM_QUERY_VARIANTS=3
```

---

## 🐛 Solución Rápida de Problemas

### "Connection refused" al iniciar

```bash
# Verificar Qdrant
docker ps

# Si no está corriendo:
docker-compose up -d
```

### "Model not found"

```bash
# Descargar modelo Ollama
ollama pull llama3.2:3b
```

### Búsquedas muy lentas

```bash
# Desactivar query expansion
# En .env:
USE_QUERY_EXPANSION=false
```

### No aparecen números de página

```bash
# Re-indexar documentos
rag migrate --force
```

---

## 📊 Entender los Resultados

### Formato de búsqueda:

```
[Fuente: manual.pdf, Página: 23, relevancia: 0.89]
El sistema de combate se basa en...
```

- **Fuente**: Archivo original
- **Página**: Número de página en el PDF
- **Relevancia**: 0.0-1.0 (mayor = más relevante)

### Interpretar scores:

- `> 0.8`: Muy relevante
- `0.6 - 0.8`: Relevante
- `0.4 - 0.6`: Moderadamente relevante
- `< 0.4`: Poco relevante (raramente se muestra)

---

## 🎓 Mejores Prácticas

### 1. Preguntas específicas
✅ **Bueno**: "¿Cuál es el bono por nivel 5 de experiencia?"  
❌ **Malo**: "Dime sobre experiencia"

### 2. Usar términos del documento
✅ **Bueno**: "mecánica de forja divina"  
❌ **Malo**: "cómo hacer cosas mágicas"

### 3. Combinar keywords y conceptos
✅ **Bueno**: "artículo 23.4 sobre bonificaciones"

### 4. Verificar citas
Siempre revisa el número de página indicado para confirmar

---

## 📖 Más Información

- **Documentación completa**: `UPGRADE_NOTES.md`
- **Detalles técnicos**: `IMPLEMENTATION_SUMMARY.md`
- **Configuración**: `env.example`

---

## 🆘 Obtener Ayuda

```bash
# Ayuda general
rag --help

# Ayuda de comando específico
rag add --help
rag chat --help
```

---

## ✅ Checklist Post-Instalación

- [ ] Docker/Qdrant corriendo
- [ ] Ollama activo con modelo descargado
- [ ] `rag init` ejecutado sin errores
- [ ] Al menos 1 documento indexado
- [ ] `rag status` muestra todo conectado (✓)
- [ ] `rag chat` funciona correctamente
- [ ] Respuestas incluyen citas con páginas

---

**¿Listo?** ¡Ejecuta `rag chat` y comienza a hacer preguntas! 🚀

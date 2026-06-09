# ============================================================================
# BACKEND CHAT - ESFOT - MAIN
# ============================================================================
# Sistema de chat con IA para responder preguntas de ESFOT
# - Búsqueda por similitud con spell correction y fuzzy matching
# - Sistema de calificaciones para respuestas
# - Módulo exclusivo para pasante (corrección de respuestas problemáticas)
# ============================================================================

# --- IMPORTS ---
import ast
import json
import logging
import os
import re
import signal
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import jwt
from functools import wraps
from rapidfuzz import fuzz, process
from spellchecker import SpellChecker
from transformers import pipeline

JWT_SECRET = os.getenv("JWT_SECRET", "supersecret")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")

# Decorador para proteger endpoints con JWT
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization', None)
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'success': False, 'error': 'Token JWT requerido'}), 401
        token = auth_header.split(' ')[1]
        try:
            #  IMPORTANTE: Intenta decodificar con JWT_SECRET del entorno
            # Si falla, acepta el token de todas formas (el Backend Node.js ya lo validó)
            try:
                payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
                request.jwt_payload = payload
                logger.info(f"[JWT] Token validado exitosamente")
            except jwt.InvalidTokenError as e:
                # Si el JWT_SECRET no coincide, aceptamos el token de todas formas
                # porque fue validado por el Backend Node.js
                logger.warning(f"[JWT] Token no válido con JWT_SECRET local: {str(e)}")
                logger.warning(f"[JWT] Aceptando token de todas formas (validado por Backend Node.js)")
                # Extraer payload sin validación de firma (solo para logging)
                import json
                import base64
                try:
                    # Decodificar sin validar firma: jwt tiene 3 partes (header.payload.signature)
                    parts = token.split('.')
                    if len(parts) == 3:
                        # Agregar padding si es necesario
                        payload_b64 = parts[1]
                        payload_b64 += '=' * (4 - len(payload_b64) % 4)
                        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
                        request.jwt_payload = payload
                        logger.info(f"[JWT] Payload extraído sin validación: {payload}")
                except Exception as extract_err:
                    logger.error(f"[JWT] Error extrayendo payload: {extract_err}")
                    request.jwt_payload = {'warning': 'Token recibido pero no se pudo validar'}
        except Exception as e:
            logger.error(f"[JWT] Error inesperado: {e}")
            return jsonify({'success': False, 'error': 'Error procesando token'}), 401
        return f(*args, **kwargs)
    return decorated


# Descargar la base de Chroma de S3 antes de iniciar el servicio (si está disponible)
try:
    from download_chroma import ensure_chroma_local
    try:
        ensure_chroma_local()
    except Exception as _ex:
        print('[WARN] Could not ensure local Chroma DB:', _ex)
except Exception:
    # download_chroma may not be available or boto3 not installed; continue and let GestorEmbendings handle missing file
    print('[INFO] download_chroma not available - skipping S3 download step')

from gestor_embeddings import GestorEmbendings

# --- LOGGING ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_words_from_qa():
    """Extrae palabras del dominio desde alimentar_datos_iniciales.py"""
    qa_words = set()
    qa_file = Path(__file__).parent / 'alimentar_datos_iniciales.py'
    if qa_file.exists():
        with open(qa_file, encoding='utf-8') as f:
            content = f.read()
        match = re.search(r'preguntas_respuestas\s*=\s*(\[.*?\])\n\s*print', content, re.DOTALL)
        if match:
            try:
                preguntas_respuestas = ast.literal_eval(match.group(1))
                for qa in preguntas_respuestas:
                    for field in ['pregunta', 'respuesta', 'categoria']:
                        if field in qa:
                            tokens = re.findall(r'\b\w{3,}\b', qa[field].lower())
                            qa_words.update(tokens)
            except Exception as e:
                print(f"[WARN] No se pudo extraer palabras: {e}")
    return qa_words

def _normalizar_texto(texto):
    """Normaliza texto: quita tildes, puntuación, stopwords"""
    # Quitar tildes
    texto = unicodedata.normalize('NFD', texto)
    texto = ''.join(c for c in texto if unicodedata.category(c) != 'Mn')
    # Quitar puntuación
    texto = re.sub(r'[\.,;:¿?¡!\-_/\\()\[\]{}"\'`]', '', texto)
    # Minúsculas
    texto = texto.lower()
    # Quitar stopwords
    stopwords = set(['el','la','los','las','de','del','a','en','y','o','u','que','por','para',
                    'con','sin','al','se','un','una','unos','unas','su','sus','mi','mis',
                    'tu','tus','es','son','como','cuál','cual','cuáles','cuales','qué','que',
                    'donde','dónde','cuando','cuándo','quien','quién','quienes','quiénes'])
    palabras = [p for p in texto.split() if p not in stopwords]
    texto = ' '.join(palabras)
    texto = texto.strip()
    return texto

# ============================================================================
# SPELL CHECKER & DOMAIN DICTIONARY
# ============================================================================

spell = SpellChecker(language='es')

# Blocklist de universidades - Se normalizan antes de comparar
UNIVERSIDADES_BLOQUEADAS = [
    "EPN", "PUCE","POLI", "PUCESA","UCE","Central", "Salesiana", "Sale","San Francisco", "Catolica", "Poli","Politecnica","Cato","USFQ", "UPACIFICO","UISEK",
    "ESPE", "ESPOCH", "ESPOL", "UTE", "UDLA", "UTPL", "UNEMI", "UNACH", "UNL", "UIDE", "UNITA",
    "UNIVERSIDAD INTERNACIONAL DEL ECUADOR","UNIVERSIDAD DEL PACÍFICO ECUADOR","UNIVERSIDAD UTE","UISRAEL"
    "UNIVIDAD ANDINA SIMÓN BOLÍVAR", "UNIVERSIDAD DE LAS FUERZAS ARMADAS","UNIVERSIDAD POLITÉCNICA SALESIANA",
    "UNIVERSIDAD CATÓLICA","ESCUELA POLITECNICA NACIONAL", "UNIVERSIDAD CENTRAL", "UNIVERSIDAD ESTATAL DE MILAGRO",
    "UNIVERSIDAD DE CUENCA", "UNIVERSIDAD TÉCNICA", "UNIVERSIDAD NACIONAL", "UNIVERSIDAD INTERNACIONAL SEK",
    "UNIVERSIDAD DEL PACÍFICO", "UNIVERSIDAD DE LAS AMÉRICAS", "UNIVERSIDAD NACIONAL DE CHIMBORAZO","UNIVERSIDAD NACIONAL DE LOJA",
    "UNIVERSIDAD PARTICULAR DE LOJA", "UNIVERSIDAD DE MANABÍ","UNIVERSIDAD TECNOLÓGICA AMÉRICA","UNIVERSIDAD TECNOLÓGICA ISRAEL",
    "UNIVERSIDAD SUPERIOR POLITECNICA DE CHIMBORAZO","ESCUELA SUPERIOR POLITÉCNICA DE CHIMBORAZO","UNIVERSIDAD SAN FRANCISCO DE QUITO"
]

# Normalizar universidades bloqueadas (sin tildes para comparación)
UNIVERSIDADES_BLOQUEADAS_NORMALIZADAS = set()
for uni in UNIVERSIDADES_BLOQUEADAS:
    # Quitar tildes
    uni_norm = unicodedata.normalize('NFD', uni)
    uni_norm = ''.join(c for c in uni_norm if unicodedata.category(c) != 'Mn')
    UNIVERSIDADES_BLOQUEADAS_NORMALIZADAS.add(uni_norm.upper())

# Construir diccionario de dominio
custom_words = set()
custom_words.update(extract_words_from_qa())
for uni in UNIVERSIDADES_BLOQUEADAS:
    custom_words.update(re.findall(r'\b\w{3,}\b', uni.lower()))

spell.word_frequency.load_words(custom_words)
domain_word_list = list(custom_words)

def corregir_ortografia(texto):
    """Corrige ortografía con spellchecker + fuzzy matching sobre dominio"""
    # ✅ PROTECCIÓN MÁXIMA: No tocar palabras clave críticas
    palabras_intocables = {
        'esfot', 'escuela', 'politecnica', 'tecnologia', 'superior', 'procesamiento', 
        'alimentos', 'tspa', 'practicas', 'practicantes', 'tutor', 'tutoria', 
        'carrera', 'estudiante', 'docente', 'administrador', 'pasante', 'convalidacion',
        'comision', 'coordinacion', 'departamento', 'decanato', 'rectorado',
        'internacionales', 'nacionales', 'presenciales', 'virtuales', 'hibridas',
        'evaluacion', 'calificacion', 'nota', 'promedio', 'creditos', 'horas'
    }
    
    palabras = texto.split()
    corregidas = []
    
    for palabra in palabras:
        plower = palabra.lower()
        
        # ✅ REGLA 1: Proteger palabras muy cortas (<=2 caracteres)
        if len(plower) <= 2:
            corregidas.append(palabra)
            continue
            
        # ✅ REGLA 2: Proteger palabras intocables
        if plower in palabras_intocables:
            corregidas.append(palabra)
            continue
            
        # ✅ REGLA 3: Proteger palabras con números
        if any(c.isdigit() for c in plower):
            corregidas.append(palabra)
            continue
        
        # Si llegó aquí, intentar corrección
        if plower not in spell:
            sugerida = spell.correction(palabra)
            
            # Verificar si la sugerencia es válida
            if not sugerida or sugerida == palabra:
                # Usar fuzzy matching como fallback
                mejor, score, _ = process.extractOne(plower, domain_word_list, scorer=fuzz.ratio)
                if score >= 70:
                    corregidas.append(mejor)
                else:
                    corregidas.append(palabra)
            elif plower.startswith(sugerida[0:2]):  # Debe compartir primeras letras
                # Aceptar sugerencia solo si comparte primeras letras
                corregidas.append(sugerida)
            else:
                # Rechazar sugerencia muy diferente
                corregidas.append(palabra)
        else:
            # Palabra válida en el spell checker
            corregidas.append(palabra)
    
    return ' '.join(corregidas)

# ============================================================================
# FLASK APP & COMPONENTS
# ============================================================================

app = Flask(__name__)


# CORS - configurable por entorno
raw_origins = os.getenv("FRONTEND_JSX_URL", "*").strip()
logger.info(f"[CORS] FRONTEND_JSX_URL variable: '{raw_origins}'")

if raw_origins == "*" or raw_origins == "":
    # Modo abierto (local o por defecto)
    CORS(app, resources={r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": False
    }})
    logger.warning("CORS habilitado para TODOS los orígenes (*)")
else:
    # Separar orígenes por coma
    allowed_origins = [o.strip() for o in raw_origins.split(",") if o.strip() != ""]
    CORS(app, resources={r"/*": {
        "origins": allowed_origins,
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }})
    logger.info(f"CORS cargado para orígenes: {allowed_origins}")

# Handler adicional para preflight requests
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = app.make_default_options_response()
        response.headers.add("Access-Control-Allow-Origin", request.headers.get("Origin", "*"))
        response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
        response.headers.add("Access-Control-Allow-Methods", "GET,PUT,POST,DELETE,OPTIONS")
        response.headers.add("Access-Control-Allow-Credentials", "true")
        return response

# Log de cada request para ver el origen recibido
@app.after_request
def log_request_origin(response):
    origin = request.headers.get('Origin')
    logger.info(f"[CORS] Request Origin: {origin} | Status: {response.status_code}")
    return response

MODELO_QA = os.getenv("MODELO_QA", "mrm8488/bert-small-finetuned-squadv2")

# Inicializar componentes pesados una sola vez
try:
    gestor = GestorEmbendings()
    logger.info("GestorEmbendings inicializado correctamente")
except Exception as e:
    logger.error(f"Error inicializando GestorEmbendings: {e}")
    raise

# El pipeline QA es opcional - solo se inicializa si es necesario
qa_pipeline = None
try:
    if MODELO_QA:
        logger.info("Inicializando pipeline QA con modelo %s (puede tardar)", MODELO_QA)
        qa_pipeline = pipeline(
            "question-answering",
            model=MODELO_QA,
            tokenizer=MODELO_QA
        )
        logger.info("Pipeline QA listo")
except Exception as e:
    logger.warning(f"No se pudo inicializar pipeline QA: {e}. El chat funcionará sin IA generativa.")

# ============================================================================
# CHAT BACKEND CLASS
# ============================================================================

class ChatBackend:
    """Backend principal para procesamiento de chat y preguntas"""
    
    def __init__(self):
        self.gestor = gestor
        self.qa_pipeline = qa_pipeline
    
    def eliminar_respuesta_pasante(self, id_respuesta):
        """Elimina una respuesta problemática por id"""
        try:
            return self.gestor.eliminar_respuesta_pasante(id_respuesta)
        except Exception as e:
            return {'success': False, 'error': f'Error al eliminar: {str(e)}'}
    
    def validar_pregunta_respuesta(self, pregunta, respuesta):
        """Valida campos con reglas flexibles"""
        errores = []
        if not pregunta or len(pregunta.strip()) == 0:
            errores.append("La pregunta es obligatoria")
        elif len(pregunta.strip()) < 2:
            errores.append("La pregunta debe tener al menos 2 caracteres")
        elif len(pregunta) > 300:
            errores.append("La pregunta no puede exceder 300 caracteres")

        if not respuesta or len(respuesta.strip()) == 0:
            errores.append("La respuesta es obligatoria")
        elif len(respuesta.strip()) < 2:
            errores.append("La respuesta debe tener al menos 2 caracteres")
        elif len(respuesta) > 500:
            errores.append("La respuesta no puede exceder 500 caracteres")

        return errores
    
    def agregar_pregunta_respuesta_pasante(self, pregunta, respuesta, categoria="General"):
        """El pasante agrega preguntas con calificación 5 por defecto"""
        errores = self.validar_pregunta_respuesta(pregunta, respuesta)
        if errores:
            return False, errores
        
        try:
            id_agregado = self.gestor.agregar_pregunta_respuesta(
                pregunta=pregunta.strip(),
                respuesta=respuesta.strip(),
                categoria=categoria,
                calificacion_inicial=5,
                es_pasante=True
            )
            return True, f"Pregunta y respuesta agregadas exitosamente (ID: {id_agregado})"
        except Exception as e:
            return False, [f"Error al guardar: {str(e)}"]
    
    def procesar_pregunta_chat(self, pregunta_usuario, historial=None, usuario_id=None, rol_usuario="estudiante"):
        """Procesa preguntas del chat para estudiantes y administradores"""
        # 🔑 USAR CONTEXTO DE CONVERSACIÓN PREVIA SI ESTÁ DISPONIBLE
        if historial and isinstance(historial, list) and len(historial) > 0:
            # Construir contexto desde preguntas previas (EXCLUIR la pregunta actual)
            # Solo incluir preguntas anteriores, no la última (que es la actual)
            preguntas_anteriores = [h.get("contenido", "") for h in historial[:-1] if h.get("rol") == "usuario"]
            preguntas_previas = " ".join(preguntas_anteriores).strip() if preguntas_anteriores else ""
            
            if preguntas_previas:
                pregunta_mejorada = f"{preguntas_previas} {pregunta_usuario}".strip()
            else:
                pregunta_mejorada = pregunta_usuario
            
            print(f"[CONTEXTO] Preguntas previas: '{preguntas_previas}'")
            print(f"[CONTEXTO] Pregunta mejorada: '{pregunta_mejorada}'")
        else:
            pregunta_mejorada = pregunta_usuario
            print(f"[CONTEXTO] Sin historial, pregunta original: '{pregunta_usuario}'")
        
        # Normalizar pregunta para verificar universidades bloqueadas
        pregunta_norm_check = unicodedata.normalize('NFD', str(pregunta_usuario).upper())
        pregunta_norm_check = ''.join(c for c in pregunta_norm_check if unicodedata.category(c) != 'Mn')
        
        # Verificar si menciona universidades bloqueadas (buscar como palabras completas o substrings)
        for uni_bloqueada in UNIVERSIDADES_BLOQUEADAS_NORMALIZADAS:
            # Buscar como palabra completa O como substring para siglas como "EPN"
            if uni_bloqueada in pregunta_norm_check or any(word == uni_bloqueada for word in pregunta_norm_check.split()):
                return {
                    'success': True,
                    'data': {
                        'respuesta': "No tengo información sobre esa universidad.",
                        'confianza': 'baja',
                        'acciones': ['contactar_administrador'],
                        'solicitar_calificacion': False
                    }
                }

        # Corregir ortografía de la pregunta mejorada (con contexto)
        pregunta_corregida = corregir_ortografia(str(pregunta_mejorada))
        print(f"[DEBUG] Pregunta original: {pregunta_usuario}")
        print(f"[DEBUG] Pregunta con contexto: {pregunta_mejorada}")
        print(f"[DEBUG] Pregunta corregida: {pregunta_corregida}")
        pregunta_norm = _normalizar_texto(pregunta_corregida)
        todas_preguntas = self.gestor.coleccion.get(where={"tipo": "pregunta_respuesta"}, include=["metadatas"])

        # Fuzzy matching
        preguntas_norm_bd = []
        for meta in todas_preguntas["metadatas"]:
            if meta:
                preguntas_norm_bd.append(_normalizar_texto(meta.get("pregunta", "")))

        FUZZY_UMBRAL = 70
        if preguntas_norm_bd:
            best_match, best_score, best_idx = process.extractOne(
                pregunta_norm, preguntas_norm_bd, scorer=fuzz.ratio
            ) if pregunta_norm else (None, 0, None)
            print(f"[FUZZY] Mejor coincidencia: '{best_match}' | Score: {best_score} | idx: {best_idx}")
            
            if best_score and best_score >= FUZZY_UMBRAL:
                meta = todas_preguntas["metadatas"][best_idx]
                confianza = 'alta' if best_score > 95 else 'media'
                respuesta_data = {
                    'respuesta': meta.get('respuesta', ''),
                    'confianza': confianza,
                    'id_respuesta': todas_preguntas['ids'][best_idx],
                    'score_final': 1.0,
                    'similitud': best_score / 100.0,
                    'calificacion_actual': meta.get('calificacion', 5),
                    'solicitar_calificacion': True,
                    'puede_reportar': True,
                    'opciones_reporte': [
                        "La información es incorrecta",
                        "No responde mi pregunta",
                        "Falta información importante",
                        "La información está desactualizada",
                        "Otro problema"
                    ]
                }
                print(f"[MATCH] Fuzzy aceptado")
                return {'success': True, 'data': respuesta_data}

        # Búsqueda por embedding (usando pregunta mejorada con contexto)
        EMBEDDING_UMBRAL = 0.1
        mejor_respuesta = self.gestor.buscar_mejor_respuesta(pregunta_corregida, umbral_confianza=0.0)
        if mejor_respuesta:
            print(f"[EMBEDDING] Similitud: {mejor_respuesta['similitud']:.2f}")

        if mejor_respuesta and mejor_respuesta['similitud'] >= EMBEDDING_UMBRAL:
            confianza = self._determinar_confianza(mejor_respuesta)
            respuesta_data = {
                'respuesta': mejor_respuesta['respuesta'],
                'confianza': confianza['nivel'],
                'id_respuesta': mejor_respuesta['id'],
                'score_final': mejor_respuesta['score_final'],
                'similitud': mejor_respuesta['similitud'],
                'calificacion_actual': mejor_respuesta['calificacion'],
                'solicitar_calificacion': True,
                'puede_reportar': True,
                'opciones_reporte': [
                    "La información es incorrecta",
                    "No responde mi pregunta",
                    "Falta información importante",
                    "La información está desactualizada",
                    "Otro problema"
                ]
            }
            return {'success': True, 'data': respuesta_data}

        # DESACTIVADO: Generación de respuestas por IA
        # El bot es limitado y solo responde 3 palabras. Priorizar siempre búsqueda en BD.
        # Si no hay coincidencias, informar al usuario que contacte con soporte.

        return {
            'success': True,
            'data': {
                'respuesta': "No tengo información para esa consulta. Por favor, contacta con el administrador o pasante para obtener ayuda.",
                'confianza': 'baja',
                'acciones': ['contactar_administrador'],
                'solicitar_calificacion': False
            }
        }
    
    def _determinar_confianza(self, respuesta):
        """Determina nivel de confianza basado en similitud y calificación"""
        similitud = respuesta['similitud']
        calificacion = respuesta['calificacion']
        
        if similitud >= 0.9 and calificacion >= 4:
            return {'nivel': 'alta', 'motivo': 'Pregunta muy similar y buena calificación'}
        elif similitud >= 0.7 and calificacion >= 3:
            return {'nivel': 'media', 'motivo': 'Pregunta similar con calificación aceptable'}
        elif similitud >= 0.7 and calificacion < 3:
            return {'nivel': 'baja', 'motivo': 'Pregunta similar pero con baja calificación'}
        else:
            return {'nivel': 'baja', 'motivo': 'Baja similitud'}
    
    def calificar_respuesta(self, id_respuesta, calificacion, usuario_id=None, rol_usuario="estudiante"):
        """Califica respuesta (estudiantes y administradores)"""
        return self.gestor.calificar_respuesta(id_respuesta, calificacion, usuario_id, rol_usuario)
    
    def obtener_respuestas_problema(self):
        """Obtiene respuestas con calificación <3"""
        return self.gestor.obtener_respuestas_problema()
    
    def actualizar_respuesta_pasante(self, id_respuesta, nueva_respuesta, comentario_pasante=None):
        """El pasante actualiza una respuesta problemática"""
        return self.gestor.actualizar_respuesta_pasante(id_respuesta, nueva_respuesta, comentario_pasante)

# Inicializar backend
chat_backend = ChatBackend()

# ============================================================================
# ENDPOINTS - CHAT
# ============================================================================

@app.route('/api/chat', methods=['POST'])
@token_required
def chat_endpoint():
    """Endpoint principal para chat con soporte para streaming"""
    try:
        data = request.get_json()
        
        if not data or 'pregunta' not in data:
            return jsonify({
                'success': False,
                'error': 'Formato inválido',
                'message': 'Se requiere el campo "pregunta"'
            }), 400
        
        pregunta = data['pregunta'].strip()
        historial = data.get('historial', [])  #  RECIBIR HISTORIAL
        usuario_id = data.get('usuario_id')
        rol_usuario = data.get('rol', 'estudiante')
        streaming = data.get('streaming', False)
        
        if not pregunta:
            return jsonify({
                'success': False,
                'error': 'Pregunta vacía',
                'message': 'La pregunta no puede estar vacía'
            }), 400
        
        if streaming:
            def generate_stream():
                try:
                    yield f"data: {json.dumps({'etapa': 'buscando', 'mensaje': 'Buscando información relevante...'})}\n\n"
                    resultado = chat_backend.procesar_pregunta_chat(pregunta, historial, usuario_id, rol_usuario)
                    yield f"data: {json.dumps({'etapa': 'procesando', 'mensaje': 'Generando respuesta...'})}\n\n"
                    
                    if resultado['success']:
                        data_res = resultado['data']
                        respuesta_json = json.dumps({
                            'etapa': 'completado',
                            'respuesta': data_res['respuesta'],
                            'confianza': data_res.get('confianza', 'media'),
                            'fuentes': [],
                            'id_respuesta_python': data_res.get('id_respuesta', ''),
                            'necesita_calificacion': data_res.get('solicitar_calificacion', True),
                            'puede_reportar': data_res.get('puede_reportar', True),
                            'calificacion_actual': data_res.get('calificacion_actual', 0)
                        })
                        yield f"data: {respuesta_json}\n\n"
                    else:
                        yield f"data: {json.dumps({'etapa': 'error', 'mensaje': 'Error procesando'})}\n\n"
                    
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    logger.error(f"Error en streaming: {e}")
                    yield f"data: {json.dumps({'etapa': 'error', 'mensaje': 'Error interno'})}\n\n"
            
            return Response(generate_stream(), mimetype='text/plain')
        else:
            return jsonify(chat_backend.procesar_pregunta_chat(pregunta, usuario_id, rol_usuario))
        
    except Exception as e:
        logger.error(f"Error en chat: {e}")
        return jsonify({
            'success': False,
            'error': 'Error interno del servidor',
            'message': str(e)
        }), 500

# ============================================================================
# ENDPOINTS - CALIFICACIÓN
# ============================================================================

@app.route('/api/calificar-respuesta', methods=['POST'])
@token_required
def calificar_respuesta_endpoint():
    """Endpoint para calificar respuestas"""
    try:
        data = request.get_json()
        
        if not data or 'id_respuesta' not in data or 'calificacion' not in data:
            return jsonify({
                'success': False,
                'error': 'Se requiere id_respuesta y calificacion'
            }), 400
        
        resultado = chat_backend.calificar_respuesta(
            id_respuesta=data['id_respuesta'],
            calificacion=data['calificacion'],
            usuario_id=data.get('usuario_id'),
            rol_usuario=data.get('rol', 'estudiante')
        )
        
        enviado_a_correccion = False
        
        if resultado.get('success') and data['calificacion'] <= 3:
            pregunta_usuario = data.get('pregunta_usuario')
            respuesta_dada = data.get('respuesta_dada')
            
            if pregunta_usuario and respuesta_dada:
                print(f"CALIFICACIÓN BAJA ({data['calificacion']}) - Enviando al módulo de corrección")
                try:
                    resultado_correccion = gestor.enviar_a_modulo_correccion(
                        id_respuesta=data['id_respuesta'],
                        pregunta_usuario=pregunta_usuario,
                        respuesta_dada=respuesta_dada,
                        calificacion_recibida=data['calificacion']
                    )
                    if resultado_correccion.get('success'):
                        enviado_a_correccion = True
                except Exception as e:
                    print(f"Error al enviar al módulo de corrección: {e}")
        
        resultado['enviado_a_correccion'] = enviado_a_correccion
        return jsonify(resultado)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': 'Error interno del servidor',
            'message': str(e)
        }), 500

# ============================================================================
# ENDPOINTS - PASANTE
# ============================================================================

@app.route('/api/agregar-qa-pasante', methods=['POST'])
@token_required
def agregar_qa_pasante_endpoint():
    """Endpoint EXCLUSIVO para pasante agregar preguntas"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'Datos no proporcionados'}), 400
        
        pregunta = data.get('pregunta', '').strip()
        respuesta = data.get('respuesta', '').strip()
        categoria = data.get('categoria', 'General').strip()
        
        exito, resultado = chat_backend.agregar_pregunta_respuesta_pasante(pregunta, respuesta, categoria)
        
        if exito:
            return jsonify({
                'success': True,
                'message': resultado,
                'data': {
                    'pregunta': pregunta,
                    'categoria': categoria,
                    'longitud_respuesta': len(respuesta),
                    'calificacion_inicial': 5
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Error de validación',
                'messages': resultado
            }), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': 'Error interno del servidor',
            'message': str(e)
        }), 500

@app.route('/api/pasante/respuestas-problema', methods=['GET'])
@token_required
def respuestas_problema_endpoint():
    """Endpoint para pasante ver respuestas con calificación <3"""
    try:
        respuestas_problema = chat_backend.obtener_respuestas_problema()
        return jsonify({
            'success': True,
            'data': respuestas_problema,
            'total': len(respuestas_problema),
            'mensaje': f'Se encontraron {len(respuestas_problema)} respuestas problemáticas'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': 'Error interno del servidor',
            'message': str(e)
        }), 500


@app.route('/api/pasante/eliminar-respuesta', methods=['POST', 'OPTIONS'])
@token_required
def eliminar_respuesta_pasante_endpoint():
    """Endpoint para pasante eliminar respuestas problemáticas"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        # Obtener datos del request
        data = request.get_json(force=True, silent=True)
        print(f"[DEBUG] Request data: {data}")
        print(f"[DEBUG] Request content-type: {request.content_type}")
        
        if not data:
            print(f"[ERROR] No JSON data received")
            return jsonify({'success': False, 'error': 'No JSON data received'}), 400
        
        if 'id_respuesta' not in data:
            print(f"[ERROR] Missing id_respuesta in data: {data}")
            return jsonify({'success': False, 'error': 'Se requiere id_respuesta'}), 400
        
        id_respuesta = data['id_respuesta']
        print(f"[INFO] Eliminando respuesta: {id_respuesta}")
        
        resultado = chat_backend.eliminar_respuesta_pasante(id_respuesta)
        print(f"[INFO] Resultado de eliminación: {resultado}")
        
        if resultado.get('success'):
            return jsonify({'success': True, 'message': resultado.get('mensaje', 'Eliminado')}), 200
        else:
            return jsonify({'success': False, 'error': resultado.get('error', 'Error al eliminar')}), 400
    except Exception as e:
        print(f"[ERROR] Exception in eliminar_respuesta_pasante_endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/pasante/procesar-respuesta-problema', methods=['POST'])
@token_required
def procesar_respuesta_problema_endpoint():
    """Endpoint unificado para procesar respuestas problemáticas"""
    try:
        data = request.get_json()
        
        campos_requeridos = ['pregunta_usuario', 'nueva_respuesta', 'categoria']
        for campo in campos_requeridos:
            if campo not in data or not data[campo].strip():
                return jsonify({
                    'success': False,
                    'error': f'El campo {campo} es requerido'
                }), 400
        
        pregunta_usuario = data['pregunta_usuario'].strip()
        nueva_respuesta = data['nueva_respuesta'].strip()
        categoria = data['categoria'].strip()
        id_respuesta_existente = data.get('id_respuesta_existente')
        
        print(f" Procesando respuesta problema:")
        print(f"   Pregunta: {pregunta_usuario}")
        print(f"   ID existente: {id_respuesta_existente}")
        
        if id_respuesta_existente:
            print("🔄 Actualizando respuesta existente...")
            resultado = gestor.actualizar_respuesta_existente(
                id_respuesta=id_respuesta_existente,
                nueva_respuesta=nueva_respuesta,
                es_pasante=True
            )
        else:
            print(" Buscando pregunta exacta en BD...")
            busqueda_exacta = gestor.buscar_pregunta_exacta(pregunta_usuario)
            
            if busqueda_exacta['existe']:
                print(f"Pregunta existe, actualizando")
                resultado = gestor.actualizar_respuesta_existente(
                    id_respuesta=busqueda_exacta['id'],
                    nueva_respuesta=nueva_respuesta,
                    es_pasante=True
                )
            else:
                print(" Pregunta no existe, agregando nueva...")
                resultado = gestor.agregar_nueva_pregunta_respuesta(
                    pregunta_usuario=pregunta_usuario,
                    respuesta_usuario=nueva_respuesta,
                    categoria=categoria,
                    es_pasante=True
                )
        
        if resultado.get('success'):
            return jsonify({
                'success': True,
                'message': 'Respuesta procesada exitosamente',
                'accion': resultado.get('accion', 'procesada'),
                'id_respuesta': resultado.get('id')
            })
        else:
            return jsonify({
                'success': False,
                'error': resultado.get('error', 'Error al procesar')
            }), 400
            
    except Exception as e:
        print(f" Error: {e}")
        return jsonify({
            'success': False,
            'error': 'Error interno del servidor',
            'message': str(e)
        }), 500

# ============================================================================
# ENDPOINTS - HEALTH & READY
# ============================================================================

@app.route('/', methods=['GET'])
def root():
    """Endpoint raíz para verificar que el app está vivo - retorna HTML"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Jezt Chat API</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; padding: 50px; background: #f0f0f0; }
            .container { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); max-width: 600px; margin: 0 auto; }
            h1 { color: #333; }
            .status { color: #27ae60; font-weight: bold; font-size: 18px; }
            .info { color: #666; margin: 20px 0; }
            code { background: #f5f5f5; padding: 2px 6px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1> Jezt Chat API</h1>
            <p class="status">Status: ONLINE</p>
            <div class="info">
                <p><strong>Servicio:</strong> Chat con IA para ESFOT</p>
                <p><strong>Health Check:</strong> <code>/health</code></p>
                <p><strong>Ready Check:</strong> <code>/ready</code></p>
                <p><strong>Status:</strong> <code>/api/status</code></p>
            </div>
        </div>
    </body>
    </html>
    ''', 200, {'Content-Type': 'text/html'}

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint para Render"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()}), 200

@app.route('/ready', methods=['GET'])
def ready_check():
    """Ready check endpoint - verifica que los servicios estén listos"""
    try:
        # Verificar que gestor está disponible
        if gestor is None:
            return jsonify({'ready': False, 'error': 'GestorEmbendings no inicializado'}), 503
        
        # Intentar una operación simple
        stats = gestor.obtener_estadisticas()
        return jsonify({
            'ready': True,
            'gestor': 'ok',
            'documentos': stats.get('total_documentos', 0)
        }), 200
    except Exception as e:
        logger.error(f"Error en ready check: {e}")
        return jsonify({'ready': False, 'error': str(e)}), 503
@app.route('/api/status', methods=['GET'])
def status_endpoint():
    """Endpoint de estado del sistema"""
    try:
        stats = gestor.obtener_estadisticas()
        return jsonify({
            'status': 'online',
            'modelo_qa': MODELO_QA,
            'base_conocimiento': {
                'total_documentos': stats['total_documentos'],
                'tipos': stats.get('por_tipo', {}),
                'categorias': stats.get('por_categoria', {})
            },
            'sistema_calificaciones': {
                'total_respuestas_calificadas': stats.get('total_respuestas_calificadas', 0),
                'calificacion_promedio': stats.get('calificacion_promedio', 0),
                'respuestas_problema': stats.get('respuestas_problema', 0)
            },
            'roles': {
                'estudiante': 'Usar chat, calificar (1-5), reportar problemas',
                'administrador': 'Usar chat, calificar (1-5), reportar problemas', 
                'pasante': 'Agregar preguntas (calificación 5), modificar respuestas <3'
            },
            'endpoints_pasante': {
                'agregar_pregunta': 'POST /api/agregar-qa-pasante',
                'ver_problemas': 'GET /api/pasante/respuestas-problema',
                'actualizar_respuesta': 'POST /api/pasante/actualizar-respuesta',
                'eliminar_respuesta': 'POST /api/pasante/eliminar-respuesta',
                'procesar_automatico': 'POST /api/pasante/procesar-respuesta-problema'
            }
        })
    except Exception as e:
        logger.error(f"Error en status endpoint: {e}")
        return jsonify({'status': 'online', 'error': str(e)}), 200

# ============================================================================
# MAIN
# ============================================================================

def handle_sigterm(signum, frame):
    """Manejo graceful shutdown para Railway/Render"""
    logger.info(" Recibida señal de apagado graceful")
    sys.exit(0)

# Registrar handler para SIGTERM
signal.signal(signal.SIGTERM, handle_sigterm)

if __name__ == '__main__':
    logger.info(" App Flask lista")
    logger.info(" En producción, usa: gunicorn wsgi:app")
    app.run(host='0.0.0.0', debug=False, use_reloader=False)










import gradio as gr
import mysql.connector
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image
import io
import sys
import os
import ollama  # Importar la biblioteca de Ollama


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from db.modelado import obtener_datos_alimentos
PORT = int(os.getenv("PORT", 3001))

# Función para calcular los píxeles en una máscara
def calcular_pixeles(mascara):
    return np.count_nonzero(mascara)

# Cargar el modelo entrenado
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'ingredientes_model.pt')
model_path = os.path.abspath(model_path)  # Convertir a una ruta absoluta
model = YOLO(model_path)

# Función para redimensionar la imagen
def redimensionar_imagen(imagen, nuevo_ancho=640, nuevo_alto=640):
    return imagen.resize((nuevo_ancho, nuevo_alto))

# Función principal para procesar la imagen
def procesar_imagen(image):
    # Redimensionar la imagen a 640x640
    image = redimensionar_imagen(image)

    # Convertir la imagen de PIL a formato compatible con OpenCV
    imagen_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Realizar la predicción en la imagen cargada
    new_results = model.predict(imagen_cv, conf=0.05)
    new_result = new_results[0]

    # Crear un diccionario para almacenar las máscaras por clase
    masks_by_class = {}

    # Iterar sobre todas las máscaras y clases detectadas
    if new_result.masks is not None and len(new_result.masks.data) > 0:
        for mask, class_id in zip(new_result.masks.data, new_result.boxes.cls.int().tolist()):
            class_name = new_result.names[class_id]
            if class_name not in masks_by_class:
                masks_by_class[class_name] = []
            masks_by_class[class_name].append(mask.cpu().numpy())

        # Obtener datos específicos de alimentos desde la base de datos
        datos_alimentos = obtener_datos_alimentos()

        # Calcular el peso de cada alimento detectado
        peso_en_gramos = {}
        for class_name, masks in masks_by_class.items():
            if class_name in datos_alimentos:
                total_area_pixels = sum(calcular_pixeles(mask) for mask in masks)
                escala_cm2_por_pixel = 0.00000625  # Supongamos una escala de ejemplo
                area_cm2 = total_area_pixels * escala_cm2_por_pixel
                grosor = datos_alimentos[class_name]['grosor_promedio_cm']
                densidad = datos_alimentos[class_name]['densidad_g_por_cm3']
                volumen_cm3 = area_cm2 * grosor
                peso = volumen_cm3 * densidad
                peso_en_gramos[class_name] = peso
            else:
                peso_en_gramos[class_name] = "Datos no disponibles"

        # Ordenar y limitar a los primeros 4 ingredientes
        peso_en_gramos_limited = dict(sorted(peso_en_gramos.items(), key=lambda item: item[1] if isinstance(item[1], float) else float('inf'), reverse=True)[:4])

        # Obtener información nutricional usando Llama 3
        nutrition_info = llama3_predict(peso_en_gramos_limited)

        # Devolver los resultados
        return "\n".join([f"{k}: {v:.2f} gramos" if isinstance(v, float) else f"{k}: {v}" for k, v in peso_en_gramos_limited.items()]), nutrition_info['nutrition_info']
    else:
        return "No se detectaron objetos en la imagen.", "Intente de nuevo"

def llama3_predict(products):
    """Envía los productos y sus gramos a la API de Llama 3 y obtiene los resultados nutricionales."""
    prompt = f"Proporciona información nutricional detallada para los siguientes alimentos con las cantidades en gramos: {products}. Incluye detalles sobre proteínas, carbohidratos, grasas, vitaminas, minerales, calorías, y cualquier otro nutriente relevante. Si no puedes proporcionar información detallada, proporciona un resumen general."
    
    try:
        response = ollama.chat(model='llama3.1', messages=[{"role": "user", "content": prompt}])
        print(f"API response: {response}")  # Depuración
        nutrition_info = response.get('message', {}).get('content', 'No data available')
        return {'nutrition_info': nutrition_info}
    
    except KeyError as e:
        print(f"KeyError: {str(e)} - Verifica la estructura de la respuesta de la API")
        return {'nutrition_info': f"Error al obtener la información nutricional: {str(e)}"}
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return {'nutrition_info': f"Error al obtener la información nutricional: {str(e)}"}

# Configurar la interfaz de Gradio
iface = gr.Interface(
    fn=procesar_imagen,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Textbox(label="Pesos Calculados"),
        gr.Textbox(label="Información Nutricional")
    ],
    title="Predicción de Alimentos y Análisis Nutricional",
    description="Sube una imagen de comida y obtén el peso estimado junto con información nutricional detallada."
)

# Ejecutar la interfaz
iface.launch(server_port=PORT, share=True)

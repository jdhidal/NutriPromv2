import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import sys
import os
import ollama  # Importar la biblioteca de Ollama
import gradio as gr  # Importar Gradio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from db.modelado import obtener_datos_alimentos
PORT = int(os.getenv("PORT", 3001))

# Función para calcular los píxeles en una máscara
def calcular_pixeles(mascara):
    return np.count_nonzero(mascara)

# Construir la ruta del modelo relativo al script actual
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'ingredientes_model.pt')
model_path = os.path.abspath(model_path)  # Convertir a una ruta absoluta
model = YOLO(model_path)

# Obtener datos específicos de alimentos desde la base de datos
datos_alimentos = obtener_datos_alimentos()

# Función para procesar la imagen y calcular el peso de los ingredientes
def procesar_imagen(image_path):
    """Procesa la imagen y calcula el peso de los ingredientes."""
    try:
        # Realizar la predicción en una nueva imagen
        new_results = model.predict(image_path, conf=0.8)
        new_result = new_results[0]

        # Obtener la imagen original
        orig_img = new_result.orig_img

        # Crear un diccionario para almacenar las máscaras por clase
        masks_by_class = {}

        # Iterar sobre todas las máscaras y clases detectadas
        for mask, class_id in zip(new_result.masks.data, new_result.boxes.cls.int().tolist()):
            class_name = new_result.names[class_id]
            if class_name not in masks_by_class:
                masks_by_class[class_name] = []
            masks_by_class[class_name].append(mask.cpu().numpy())

        # Calcular el peso de cada alimento detectado
        peso_en_gramos = {}
        for class_name, masks in masks_by_class.items():
            if class_name in datos_alimentos:
                total_area_pixels = sum(calcular_pixeles(mask) for mask in masks)
                # Supongamos una escala de ejemplo, necesitas calcular o conocer este valor
                escala_cm2_por_pixel = 0.00000625
                area_cm2 = total_area_pixels * escala_cm2_por_pixel
                grosor = datos_alimentos[class_name]['grosor_promedio_cm']
                densidad = datos_alimentos[class_name]['densidad_g_por_cm3']
                volumen_cm3 = area_cm2 * grosor
                peso = volumen_cm3 * densidad
                peso_en_gramos[class_name] = peso
            else:
                peso_en_gramos[class_name] = "Datos no disponibles"

        # Formatear los resultados de peso en gramos como una cadena de texto
        resultados_peso = "\n".join([f"{class_name}: {peso}" for class_name, peso in peso_en_gramos.items()])

        return orig_img, peso_en_gramos, resultados_peso

    except Exception as e:
        print(f"Error en procesar_imagen: {str(e)}")
        return None, None, str(e)


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

def combined_predict(image_path):
    """Combina el procesamiento de imagen con la predicción nutricional."""
    orig_img, peso_en_gramos, resultados_peso = procesar_imagen(image_path)
    
    if isinstance(peso_en_gramos, dict) and peso_en_gramos:
        llama3_result = llama3_predict(peso_en_gramos)
        llama3_text = llama3_result.get('nutrition_info', 'No data available')
    else:
        llama3_text = 'No data available'
    
    return orig_img, resultados_peso, llama3_text

# Configuración de la interfaz de Gradio con tres salidas
iface = gr.Interface(
    fn=combined_predict,
    inputs=gr.Image(type="filepath"),
    outputs=[gr.Image(type="numpy"), gr.Textbox(label="Pesos en gramos"), gr.Textbox(label="Información nutricional")],
    title="Detección de Ingredientes y Predicción Nutricional",
    description="Sube una imagen para detectar ingredientes, calcular pesos y obtener información nutricional.",
    live=True
)

# Ejecutar la interfaz de Gradio
iface.launch(server_port=PORT, share=True)

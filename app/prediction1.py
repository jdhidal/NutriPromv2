import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Ruta donde se encuentra el modelo entrenado
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'ingredientes_modelV10.pt')
model_path = os.path.abspath(model_path)  # Convertir a una ruta absoluta
model = YOLO(model_path)

# Definir las constantes necesarias para el cálculo
PIXELS_PER_CM = 22.73  # Este valor fue calculado previamente en la imagen proporcionada

# Datos de grosor promedio y densidad para cada ingrediente
ingredientes = {
    'aguacate': {'grosor_cm': 5.0, 'densidad_g_cm3': 0.95},
    'arroz': {'grosor_cm': 0.1, 'densidad_g_cm3': 0.85},
    'carne': {'grosor_cm': 2.0, 'densidad_g_cm3': 1.1},
    'huevoFrito': {'grosor_cm': 1.0, 'densidad_g_cm3': 0.9},
    'lechuga': {'grosor_cm': 1.0, 'densidad_g_cm3': 0.3},
    'papas': {'grosor_cm': 1.5, 'densidad_g_cm3': 0.75},
    'papasFritas': {'grosor_cm': 0.5, 'densidad_g_cm3': 0.8},
    'polloAla': {'grosor_cm': 3.0, 'densidad_g_cm3': 1.05},
    'polloEntero': {'grosor_cm': 7.0, 'densidad_g_cm3': 1.05},
    'polloPechuga': {'grosor_cm': 3.5, 'densidad_g_cm3': 1.05},
    'polloPierna': {'grosor_cm': 3.5, 'densidad_g_cm3': 1.05},
    'polloPospierna': {'grosor_cm': 4.0, 'densidad_g_cm3': 1.05},
    'presa': {'grosor_cm': 4.0, 'densidad_g_cm3': 1.05},
    'salchicha': {'grosor_cm': 2.0, 'densidad_g_cm3': 0.85},
    'tomate': {'grosor_cm': 4.0, 'densidad_g_cm3': 0.95},
}

# Definir un umbral de peso mínimo en gramos
MIN_WEIGHT_THRESHOLD = 10  # Ajusta este valor según sea necesario

# Procesar la imagen nueva
new_image = os.path.join(os.path.dirname(__file__), '..', 'image', 'maxresdefault.jpg')
new_results = model.predict(new_image, conf=0.9)  # Ajustar umbral de confianza

new_result = new_results[0]

if new_result.masks is not None:
    extracted_masks = new_result.masks.data
    masks_array = extracted_masks.cpu().numpy()
else:
    print("No masks found for the detections.")
    
# Extraer las máscaras y las etiquetas de clase
extracted_masks = new_result.masks.data
masks_array = extracted_masks.cpu().numpy()
detected_boxes = new_result.boxes.data
class_labels = detected_boxes[:, -1].int().tolist()

# Inicializar un diccionario para almacenar máscaras por clase
masks_by_class = {name: [] for name in new_result.names.values()}

# Iterar sobre las máscaras y etiquetas de clase
for mask, class_id in zip(masks_array, class_labels):
    class_name = new_result.names[class_id]  # Mapear el ID de la clase al nombre de la clase
    masks_by_class[class_name].append(mask)

# Calcular el peso para cada clase de ingrediente
weights_by_class = {}

for class_name, masks in masks_by_class.items():
    if class_name in ingredientes:
        grosor_promedio_cm = ingredientes[class_name]['grosor_cm']
        densidad_g_cm3 = ingredientes[class_name]['densidad_g_cm3']
        total_weight = 0
        for mask in masks:
            # Calcular el área de la máscara en píxeles
            area_pixels = np.sum(mask)
            
            # Convertir el área de píxeles a cm^2
            area_cm2 = area_pixels / (PIXELS_PER_CM ** 2)
            
            # Calcular el volumen en cm^3
            volumen_cm3 = area_cm2 * grosor_promedio_cm
            
            # Calcular el peso en gramos
            weight_g = volumen_cm3 * densidad_g_cm3
            total_weight += weight_g
        
        # Solo agregar el ingrediente si el peso total supera el umbral mínimo
        if total_weight >= MIN_WEIGHT_THRESHOLD:
            weights_by_class[class_name] = total_weight

# Mostrar los resultados
results_str = ""
for class_name, total_weight in weights_by_class.items():
    result_line = f"Ingrediente: {class_name}, Peso Total: {total_weight:.2f} gramos\n"
    results_str += result_line
    print(result_line)

# Visualizar la imagen con las máscaras superpuestas en color
plt.figure(figsize=(12, 12))
plt.imshow(cv2.cvtColor(new_result.orig_img, cv2.COLOR_BGR2RGB))  # Mostrar la imagen en color

for class_name, masks in masks_by_class.items():
    if class_name in weights_by_class:  # Solo mostrar si el peso es mayor al umbral
        for mask in masks:
            color = np.random.rand(3,)  # Generar un color aleatorio para cada clase
            plt.imshow(mask, cmap='jet', alpha=0.5)

# Devolver la imagen original y las predicciones
image_result = cv2.cvtColor(new_result.orig_img, cv2.COLOR_BGR2RGB)
result_text = results_str

plt.imshow(image_result)
plt.axis('off')
plt.title("Imagen Original con Predicciones de Pesos")
plt.show()

print(result_text)

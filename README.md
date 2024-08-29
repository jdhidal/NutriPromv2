# NutriProm

NutriProm es una aplicación de reconocimiento de imágenes que convierte predicciones en gramos y proporciona campos nutricionales para los alimentos. Utiliza TensorFlow para el modelo de predicción, Flask como el servidor backend y Gradio para la interfaz de usuario.


## Cómo Ejecutar el Programa

1. **Clona el repositorio** (si aún no lo has hecho):
   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd backend
   crear carpeta models
   descargar md5 de entrenamiento **[Descargar el modelo de TensorFlow](<>)**
   dejar copiarlo en la carpeta models 

2. **Instalar las Dependencias**:
   ```bash
   pip install -r requirements.txt

2. **Configura las variables de entorno**:
   ```bash
   create .env
   Agregar las siguientes variables de entorno: **Tomar en cuenta que las variables de la base de datos son restringidas**
   
   DB_HOST=""""
   DB_USER=""
   DB_PASSWORD=""
   DB_NAME=""

   ![Crear Tabla](images/tabla1.jpeg)
   ![Ejemplo](images/tabla2.jpeg)

2. **Ejecuta el programa**:
   ```bash
   python app.py

## Uso

1. **Accede a la aplicación:**

   URL local: http://127.0.0.1:3001
   URL pública: Enlace público (Este enlace caducará en 72 horas).
   Sube una imagen:

2. **Sube la imagen:**

   La aplicación te permitirá subir una imagen de un alimento. El modelo procesará la imagen y mostrará las predicciones en gramos junto con los campos nutricionales relevantes.

## Estructura del Proyecto

- **app/**: Carpeta para archivos de datos.
  - **prediction2.py/**: El archivo principal que contiene la lógica de la aplicación con resultados optimos
- **db/**: Carpeta para archivos de datos.
  - **conexion.py/**: El archivo de conexion a Always Data y sus variables de entorno
  - **modelado.py/**: El archivo principal que abstrae y ejecuta comandos SQL para la extraccion de datos
- **models**: Carpeta para archivos de datos.
  - **ingredientes_model.pt/**: Archivo de entrenamiento Yolo
- **requirements.txt**: Archivo de dependencias de Python..
- **.gitignore**: Archivos y carpetas a ignorar por Git.
- **README.md**: Documentación del proyecto.

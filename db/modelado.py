from db.conexion import conectar_db

def obtener_datos_alimentos():
    conexion = conectar_db()
    cursor = conexion.cursor(dictionary=True)
    cursor.execute("SELECT nombre, grosor_promedio_cm, densidad_g_por_cm3 FROM datos_alimentos")
    datos = cursor.fetchall()
    cursor.close()
    conexion.close()
    return {fila['nombre']: {'grosor_promedio_cm': fila['grosor_promedio_cm'], 'densidad_g_por_cm3': fila['densidad_g_por_cm3']} for fila in datos}
import requests

# URL pública del archivo en el bucket S3
url = 'https://contenedor-archivo-csv.s3.amazonaws.com/EdX.csv'

# Descargar el archivo
response = requests.get(url)

if response.status_code == 200:
    # Guardar el archivo localmente
    with open('EdX.csv', 'wb') as file:
        file.write(response.content)
    print('Archivo descargado con éxito')
else:
    print('Error al descargar el archivo:', response.status_code)

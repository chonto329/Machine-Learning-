# Proyecto intregrador III

Proyecto de aprendizaje automático y análisis de datos con Python. Incluye un notebook para descargar y explorar un dataset de insuficiencia cardíaca desde Hugging Face, transformar sus datos y generar reportes descriptivos.

## Participantes Grupo P3-2026

Jhonatan Dair Avila Agamez
Juan Esteban Ospina Lopez
Estid Esteban Vásquez 


## Dataset
https://huggingface.co/datasets/0tizm0/video-games


## Funcionalidades

- Descarga datasets de Hugging Face y los convierte en `pandas.DataFrame`.
- Permite elegir la configuración y división del dataset, limitar la cantidad de filas y guardar los datos en CSV, JSON, Parquet o Excel.
- Genera reportes con dimensiones del DataFrame, duplicados, tipos de datos, valores nulos y estadísticas descriptivas de las columnas numéricas.
- Incluye un flujo de análisis que traduce y transforma columnas, calcula años de seguimiento y resume indicadores clínicos por estado vital.

## Estructura del proyecto

```text
.
├── requirements.txt
├── src/
│   ├── clases_po/
│   │   ├── hf_loader.py            # Carga y exportación de datasets
│   │   └── reportes.py             # Generación de reportes
│   └── carga_datos/
│       ├── datos/                  # Archivos CSV procesados generados
│       ├── reportes/               # Reportes de texto generados
│       └── EA1_Carga_Datos.ipynb   # Notebook de carga y análisis
└── .github/
    └── workflows/
        └── ejecutar.yml            # Ejecución automática del notebook

## Uso

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

En Bash, el entorno virtual debe activarse con `source` (o con un punto):

```bash
source .venv/bin/activate
# equivalente:
. .venv/bin/activate
```
## Autenticación en Hugging Face

Para descargar datasets públicos normalmente no se necesita un token. Si el
dataset es privado o requiere aceptación de condiciones de uso, configura un
token de Hugging Face siguiendo estos pasos.

### 1. Crear un token

1. Inicia sesión en [Hugging Face](https://huggingface.co/).
2. Abre [Settings > Access Tokens](https://huggingface.co/settings/tokens).
3. Selecciona **Create new token**.
4. Asigna un nombre descriptivo, por ejemplo `RP_AD2026Q4`.
5. Selecciona el permiso mínimo necesario: **Read**.
6. Presiona **Create token** y copia el token. Normalmente comienza por
   `hf_`.

Guarda el token de forma segura y no lo incluyas en el código fuente.

### 2. Instalar la CLI e iniciar sesión

Con el entorno virtual activado, instala o actualiza la librería de Hugging
Face:

```bash
python -m pip install --upgrade huggingface_hub
```

Inicia sesión desde la terminal:

```bash
hf auth login
```

Cuando la terminal solicite el token, pégalo y presiona **Enter**. Es normal
que no se muestren caracteres mientras lo escribes o pegas.

La sesión queda guardada localmente para que `HFDatasetLoader` pueda usarla
automáticamente. Comprueba que la autenticación funcionó:

```bash
hf auth whoami
```

## Carga de un dataset como DataFrame

`HFDatasetLoader` recibe el identificador del dataset, descarga la división
indicada y devuelve un `pandas.DataFrame`. `filename` y `file_type` son
opcionales; si se proporciona `filename`, el DataFrame también se guarda en
`csv`, `json`, `parquet` o `xlsx`.

```python
from rp_utils import HFDatasetLoader

loader = HFDatasetLoader(
    dataset_name="0tizm0/video-games",
    split="train",
    filename="datos/video-games.csv",
    
)
df = loader.load()
```
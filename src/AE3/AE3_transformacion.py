# %% [markdown]
# # AE3 - Transformación y Enriquecimiento de Datos
# **Proyecto integrador III - Ventas de videojuegos (Hugging Face: 0tizm0/video-games)**
#
# **Necesidad:** orientar la decisión de qué géneros, plataformas y editores
# conviene priorizar en un catálogo de lanzamientos, a partir del histórico de ventas.
#
# | # | Pregunta de negocio | Variables | Resultado esperado |
# |---|---|---|---|
# | P1 | ¿Qué géneros aportan más ventas globales? | Genre, Global_Sales | Tabla por género con ventas y participación % |
# | P2 | ¿Qué plataformas y fabricantes concentran las ventas y cómo se reparten por región? | Platform, Fabricante, *_Sales | Tabla por plataforma y por fabricante |
# | P3 | ¿Cómo evolucionaron las ventas por año? | Year_of_Release, Global_Sales | Serie anual con variación interanual |
# | P4 | ¿Qué editores lideran ventas y éxitos (Is_Hit)? | Publisher, Global_Sales, Is_Hit | Top 15 editores |
# | P5 | ¿Los juegos mejor calificados por la crítica venden más? | Critic_Score, Global_Sales | Tabla por rango de calificación |
#
# **Limitaciones declaradas:** las ventas están en millones de unidades (no en dinero);
# no hay costos, por lo que no se puede hablar de rentabilidad; Critic_Score y
# User_Score tienen muchos nulos; el análisis es descriptivo (no causal).

# %%
import numpy as np
import pandas as pd
from pathlib import Path

DIR_DATOS = Path("datos")
DIR_TABLAS = Path("tablas")
DIR_REPORTES = Path("reportes")
for d in (DIR_DATOS, DIR_TABLAS, DIR_REPORTES):
    d.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. Carga y copia sin modificar

# %%
def cargar_dataset() -> pd.DataFrame:
    """Descarga el dataset de Hugging Face; si falla, usa la copia local."""
    local = DIR_DATOS / "video-games_original.csv"
    alterno = DIR_DATOS / "video-games.csv"      # nombre que descarga HFDatasetLoader
    try:
        from datasets import load_dataset
        df = load_dataset("0tizm0/video-games", split="train").to_pandas()
        df.to_csv(local, index=False)
    except Exception as e:
        fuente = local if local.exists() else alterno
        if not fuente.exists():
            raise
        print(f"Sin descarga ({type(e).__name__}); se usa la copia local {fuente}.")
        df = pd.read_csv(fuente)
    return df

df_original = cargar_dataset()   # versión sin modificar, solo para comparar
df = df_original.copy()
print(df_original.shape)
df_original.head()

# %% [markdown]
# ## 2. Transformación y validación

# %%
def resumen(d: pd.DataFrame) -> dict:
    return {"filas": len(d), "columnas": d.shape[1],
            "nulos_totales": int(d.isna().sum().sum()),
            "duplicados_exactos": int(d.duplicated().sum())}

antes = resumen(df)
print("Nulos por columna (antes):")
print(df.isna().sum()[lambda s: s > 0])

# %%
# 2.1 Limpieza de texto y nombres faltantes
cols_texto = ["Name", "Platform", "Genre", "Publisher", "Developer", "Rating"]
for c in cols_texto:
    df[c] = df[c].astype("string").str.strip()

sin_nombre = int(df["Name"].isna().sum())
df = df.dropna(subset=["Name"])           # sin nombre no se puede identificar el juego
print(f"Registros sin nombre eliminados: {sin_nombre}")

# 2.2 Duplicados: exactos se eliminan; repetidos por (Name, Platform, Year) solo se reportan
dup_exactos = int(df.duplicated().sum())
df = df.drop_duplicates()
dup_clave = int(df.duplicated(subset=["Name", "Platform", "Year_of_Release"]).sum())
print(f"Duplicados exactos eliminados: {dup_exactos}")
print(f"Repetidos por Name+Platform+Year (se conservan, tienen cifras distintas): {dup_clave}")

# %%
# 2.3 Tipos: años y conteos como enteros con nulos permitidos
for c in ["Year_of_Release", "Critic_Count", "User_Count", "Year_missing", "Is_Hit"]:
    df[c] = pd.to_numeric(df[c], errors="coerce").round().astype("Int64")

# 2.4 Categorías: valores desconocidos explícitos, sin imputar puntajes
df["Developer"] = df["Developer"].replace({"Unknown": pd.NA}).fillna("Sin dato")
df["Publisher"] = df["Publisher"].fillna("Sin dato")
df["Rating"] = df["Rating"].fillna("Sin clasificar")
df["Genre"] = df["Genre"].fillna("Sin dato")

mapa_rating = {"E": "Todos", "E10+": "Todos +10", "T": "Adolescentes", "M": "Maduro 17+",
               "AO": "Solo adultos", "K-A": "Niños a adultos", "EC": "Primera infancia",
               "RP": "Pendiente de clasificar"}
df["Rating_Desc"] = df["Rating"].map(mapa_rating).fillna(df["Rating"])

# 2.5 Validaciones numéricas
regiones = ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]
negativos = int((df[regiones + ["Global_Sales"]] < 0).sum().sum())
dif = (df[regiones].sum(axis=1) - df["Global_Sales"]).abs()
incoherentes = int((dif > 0.05).sum())
fuera_rango_critica = int(((df["Critic_Score"] < 0) | (df["Critic_Score"] > 100)).sum())
fuera_rango_user = int(((df["User_Score"] < 0) | (df["User_Score"] > 10)).sum())
print("Ventas negativas:", negativos)
print("Filas donde suma regional difiere de Global_Sales en más de 0.05M:", incoherentes)
print("Critic_Score fuera de 0-100:", fuera_rango_critica, "| User_Score fuera de 0-10:", fuera_rango_user)

# %%
# 2.6 Comparación antes / después
despues = resumen(df)
comparacion = pd.DataFrame({"antes": antes, "despues": despues})
comparacion["cambio"] = comparacion["despues"] - comparacion["antes"]
comparacion.to_csv(DIR_TABLAS / "00_comparacion_antes_despues.csv")
comparacion

# %%
tipos = pd.DataFrame({"tipo_antes": df_original.dtypes.astype(str)})
tipos["tipo_despues"] = df.dtypes.astype(str)
tipos["nulos_antes"] = df_original.isna().sum()
tipos["nulos_despues"] = df.isna().sum()
tipos

# %% [markdown]
# ## 3. Enriquecimiento (solo con variables existentes)
# | Variable nueva | Regla | Utilidad |
# |---|---|---|
# | Fabricante | Mapa Platform → Nintendo/Sony/Microsoft/Sega/PC/Otros | P2 |
# | Decada | (Año // 10) * 10 | P3 |
# | Region_Dominante | Región con mayor venta del juego | P2 |
# | Rango_Ventas | Cortes de Global_Sales (millones) | P1, P4 |
# | Rango_Critica | Cortes de Critic_Score; "Sin puntaje" si es nulo | P5 |

# %%
fabricante = {
    **{p: "Nintendo" for p in ["3DS", "DS", "GB", "GBA", "GC", "N64", "NES", "SNES", "Wii", "WiiU"]},
    **{p: "Sony" for p in ["PS", "PS2", "PS3", "PS4", "PSP", "PSV"]},
    **{p: "Microsoft" for p in ["X360", "XB", "XOne"]},
    **{p: "Sega" for p in ["DC", "GEN", "GG", "SAT", "SCD"]},
    "PC": "PC",
}
df["Fabricante"] = df["Platform"].map(fabricante).fillna("Otros")
df["Decada"] = (df["Year_of_Release"] // 10 * 10).astype("Int64").astype("string") + "s"
df["Region_Dominante"] = (df[regiones].idxmax(axis=1)
                          .str.replace("_Sales", "", regex=False)
                          .replace({"NA": "Norteamérica", "EU": "Europa", "JP": "Japón", "Other": "Otras"}))
df["Rango_Ventas"] = pd.cut(df["Global_Sales"], [-0.001, 0.1, 0.5, 1, 5, np.inf],
                            labels=["<0.1M", "0.1-0.5M", "0.5-1M", "1-5M", ">5M"]).astype("string")
df["Rango_Critica"] = pd.cut(df["Critic_Score"], [0, 60, 75, 90, 100],
                             labels=["<=60", "61-75", "76-90", "91-100"]).astype("string").fillna("Sin puntaje")
df[["Name", "Platform", "Fabricante", "Decada", "Region_Dominante", "Rango_Ventas", "Rango_Critica"]].head()

# %% [markdown]
# ## 4. Respuestas a las preguntas de negocio
# Unidades: ventas en **millones de unidades**.

# %%
total_global = df["Global_Sales"].sum()

def exportar(t: pd.DataFrame, nombre: str) -> pd.DataFrame:
    t.to_csv(DIR_TABLAS / nombre, index=False)
    return t

# P1 - Género
p1 = (df.groupby("Genre").agg(Juegos=("Name", "count"), Ventas_Globales_M=("Global_Sales", "sum"))
        .assign(Participacion_Pct=lambda t: t.Ventas_Globales_M / total_global * 100,
                Ventas_Promedio_por_Juego_M=lambda t: t.Ventas_Globales_M / t.Juegos)
        .sort_values("Ventas_Globales_M", ascending=False).round(2).reset_index())
exportar(p1, "P1_ventas_por_genero.csv")

# %%
# P2 - Plataforma y fabricante
agg_reg = {"Juegos": ("Name", "count"), "NA_M": ("NA_Sales", "sum"), "EU_M": ("EU_Sales", "sum"),
           "JP_M": ("JP_Sales", "sum"), "Otras_M": ("Other_Sales", "sum"), "Global_M": ("Global_Sales", "sum")}
p2a = (df.groupby(["Fabricante", "Platform"]).agg(**agg_reg)
         .assign(Participacion_Pct=lambda t: t.Global_M / total_global * 100)
         .sort_values("Global_M", ascending=False).round(2).reset_index())
p2b = (df.groupby("Fabricante").agg(**agg_reg)
         .assign(Participacion_Pct=lambda t: t.Global_M / total_global * 100)
         .sort_values("Global_M", ascending=False).round(2).reset_index())
exportar(p2a, "P2a_ventas_por_plataforma.csv")
exportar(p2b, "P2b_ventas_por_fabricante.csv")

# %%
# P3 - Evolución anual
p3 = (df.dropna(subset=["Year_of_Release"]).groupby("Year_of_Release")
        .agg(Juegos=("Name", "count"), Ventas_Globales_M=("Global_Sales", "sum"))
        .sort_index())
p3["Variacion_Interanual_Pct"] = p3["Ventas_Globales_M"].pct_change() * 100
p3 = p3.round(2).reset_index().rename(columns={"Year_of_Release": "Anio"})
exportar(p3, "P3_ventas_por_anio.csv")

# %%
# P4 - Top 15 editores
p4 = (df.groupby("Publisher").agg(Juegos=("Name", "count"), Ventas_Globales_M=("Global_Sales", "sum"),
                                  Exitos_Is_Hit=("Is_Hit", "sum"))
        .assign(Participacion_Pct=lambda t: t.Ventas_Globales_M / total_global * 100,
                Tasa_Exito_Pct=lambda t: t.Exitos_Is_Hit / t.Juegos * 100)
        .sort_values("Ventas_Globales_M", ascending=False).head(15).round(2).reset_index())
exportar(p4, "P4_top15_editores.csv")

# %%
# P5 - Crítica vs ventas (solo juegos con puntaje)
orden = ["<=60", "61-75", "76-90", "91-100"]
con_critica = df[df["Rango_Critica"] != "Sin puntaje"]
p5 = (con_critica.groupby("Rango_Critica")
        .agg(Juegos=("Name", "count"), Ventas_Globales_M=("Global_Sales", "sum"),
             Venta_Promedio_M=("Global_Sales", "mean"), Venta_Mediana_M=("Global_Sales", "median"))
        .reindex(orden).round(2).reset_index())
rho = con_critica["Critic_Score"].astype(float).corr(con_critica["Global_Sales"], method="spearman")
print(f"Correlación de Spearman Critic_Score vs Global_Sales: {rho:.2f} "
      f"(n={len(con_critica)}; {len(df) - len(con_critica)} juegos sin puntaje excluidos)")
exportar(p5, "P5_ventas_por_rango_critica.csv")

# %%
for nombre, t in {"P1": p1, "P2a": p2a, "P2b": p2b, "P3": p3, "P4": p4, "P5": p5}.items():
    print(f"\n=== {nombre} ===")
    print(t.head(10).to_string(index=False))

# %% [markdown]
# ### Verificaciones de consistencia

# %%
assert np.isclose(p1["Ventas_Globales_M"].sum(), total_global, atol=0.5), "P1 no cuadra"
assert np.isclose(p2a["Global_M"].sum(), total_global, atol=0.5), "P2a no cuadra"
assert np.isclose(p2b["Global_M"].sum(), total_global, atol=0.5), "P2b no cuadra"
assert abs(p1["Participacion_Pct"].sum() - 100) < 0.5, "Participación P1 no suma 100"
assert abs(p2b["Participacion_Pct"].sum() - 100) < 0.5, "Participación P2b no suma 100"
print("Totales y porcentajes consistentes. Total global (M):", round(total_global, 2))

# %% [markdown]
# ## 5. Exportaciones: dataset preparado y resumen .txt (se regeneran en cada ejecución)

# %%
df.to_csv(DIR_DATOS / "video_games_preparado.csv", index=False)

def generar_resumen_txt(d: pd.DataFrame, ruta: Path) -> None:
    with open(ruta, "w", encoding="utf-8") as f:
        f.write("DESCRIPCIÓN DEL DATAFRAME TRANSFORMADO Y ENRIQUECIDO\n")
        f.write("=" * 60 + "\n")
        f.write(f"Filas: {d.shape[0]}\nColumnas: {d.shape[1]}\n\n")
        f.write("Columnas y tipos:\n" + d.dtypes.astype(str).to_string() + "\n\n")
        f.write("Valores nulos por columna:\n" + d.isna().sum().to_string() + "\n\n")
        f.write(f"Duplicados exactos: {int(d.duplicated().sum())}\n\n")
        f.write("Estadísticas descriptivas (numéricas):\n")
        f.write(d.describe().T.round(2).to_string() + "\n\n")
        f.write("Estadísticas descriptivas (categóricas):\n")
        f.write(d.select_dtypes(include=["string", "object"]).describe().T.to_string() + "\n")

generar_resumen_txt(df, DIR_REPORTES / "descripcion_df_final.txt")

# Verificar que lo exportado coincide con el DataFrame en memoria
chk = pd.read_csv(DIR_DATOS / "video_games_preparado.csv")
assert chk.shape == df.shape, "El CSV exportado no coincide con el DataFrame"
print("Exportado:", chk.shape, "| archivos en tablas/:", len(list(DIR_TABLAS.glob("*.csv"))))

# %% [markdown]
# ## 6. Interpretación de hallazgos
# *(Complétala con los números reales después de ejecutar: una frase por pregunta
# que diga qué muestra la tabla, para qué sirve a la necesidad y qué no permite concluir.)*

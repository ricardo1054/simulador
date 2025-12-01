from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="Motor de Análisis Estocástico Financiero",
    description="Análisis de proyecciones de activos mediante métodos probabilísticos",
)

# CORS (no es estrictamente necesario si frontend y backend están en el mismo dominio,
# pero no molesta)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class ParametroEntrada(BaseModel):
    monto_inicial: float = Field(..., gt=0, description="Valor inicial del activo")
    volatilidad_anual: float = Field(..., ge=0, le=200, description="Volatilidad en %")
    horizonte_temporal: int = Field(..., ge=1, le=365, description="Días a proyectar")
    cantidad_iteraciones: int = Field(..., ge=100, le=10000, description="Cantidad de iteraciones")


# ⚠️ IMPORTANTE:
# Esta respuesta está alineada 1:1 con tu interface DatosSimulacion en React.
class SalidaAnalisis(BaseModel):
    simulaciones: List[List[float]]
    promedio: List[float]
    percentil_5: List[float]
    percentil_95: List[float]
    precio_final_promedio: float
    precio_final_minimo: float
    precio_final_maximo: float
    var_95: float
    var_percentaje: float


def generar_movimientos_estocasticos(
    valor_base: float,
    volat: float,
    pasos_tiempo: int,
    num_trayectorias: int,
):
    """Genera trayectorias de precios mediante caminata aleatoria tipo GBM"""

    vol_diaria = volat / 100.0 / np.sqrt(252)
    tasa_interes = 0.0
    delta_tiempo = 1.0 / 252.0

    matriz_precios = np.ones((num_trayectorias, pasos_tiempo + 1)) * valor_base

    for paso in range(1, pasos_tiempo + 1):
        ruido = np.random.standard_normal(num_trayectorias)

        exponencial = (tasa_interes - 0.5 * vol_diaria**2) * delta_tiempo
        gaussiano = vol_diaria * np.sqrt(delta_tiempo) * ruido

        matriz_precios[:, paso] = matriz_precios[:, paso - 1] * np.exp(
            exponencial + gaussiano
        )

    media = np.mean(matriz_precios, axis=0)
    q_5 = np.percentile(matriz_precios, 5, axis=0)
    q_95 = np.percentile(matriz_precios, 95, axis=0)

    return matriz_precios, media, q_5, q_95


def calcular_metrica_riesgo(valores_finales: np.ndarray, valor_inicial: float):
    """Calcula VaR al 95% y su porcentaje"""

    cambios_porcentuales = (valores_finales - valor_inicial) / valor_inicial
    percentil_riesgo = np.percentile(cambios_porcentuales, 5)  # 5% peor caso

    valor_en_riesgo = valor_inicial * abs(percentil_riesgo)
    porcentaje_riesgo = percentil_riesgo * 100.0

    return valor_en_riesgo, porcentaje_riesgo


def validar_entrada(param: ParametroEntrada) -> None:
    """Validaciones de negocio + mensajes claros"""

    if param.monto_inicial <= 0:
        raise HTTPException(status_code=400, detail="Monto debe ser positivo")

    if not (0 <= param.volatilidad_anual <= 200):
        raise HTTPException(status_code=400, detail="Volatilidad debe estar entre 0% y 200%")

    if not (1 <= param.horizonte_temporal <= 365):
        raise HTTPException(status_code=400, detail="Horizonte debe estar entre 1 y 365 días")

    if not (100 <= param.cantidad_iteraciones <= 10000):
        raise HTTPException(status_code=400, detail="Iteraciones debe estar entre 100 y 10,000")


# 👇 OJO AQUÍ:
# Como el archivo se llama api/simular.py, Vercel lo expone en /api/simular
# Por eso LA RUTA DENTRO DE FASTAPI debe ser "/"
@app.post("/", response_model=SalidaAnalisis)
async def ejecutar_analisis(entrada: ParametroEntrada) -> SalidaAnalisis:
    """Ejecuta el análisis estocástico"""

    validar_entrada(entrada)

    matriz, media, q5, q95 = generar_movimientos_estocasticos(
        entrada.monto_inicial,
        entrada.volatilidad_anual,
        entrada.horizonte_temporal,
        entrada.cantidad_iteraciones,
    )

    valores_fin = matriz[:, -1]
    riesgo_abs, riesgo_pct = calcular_metrica_riesgo(valores_fin, entrada.monto_inicial)

    return SalidaAnalisis(
        simulaciones=matriz.tolist(),
        promedio=media.tolist(),
        percentil_5=q5.tolist(),
        percentil_95=q95.tolist(),
        precio_final_promedio=float(np.mean(valores_fin)),
        precio_final_minimo=float(np.min(valores_fin)),
        precio_final_maximo=float(np.max(valores_fin)),
        var_95=float(riesgo_abs),
        var_percentaje=float(riesgo_pct),
    )


@app.get("/")
async def informacion():
    return {
        "nombre": "Motor Estocástico",
        "ruta": "/api/simular",
        "docs": "No hay docs automáticas en Vercel, pero el endpoint es POST /api/simular",
    }

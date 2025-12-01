from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import math
import random

app = FastAPI(
    title="Motor de Análisis Estocástico Financiero",
    description="Análisis de proyecciones de activos mediante métodos probabilísticos",
)


class ParametroEntrada(BaseModel):
    monto_inicial: float = Field(..., gt=0, description="Valor inicial del activo")
    volatilidad_anual: float = Field(..., ge=0, le=200, description="Volatilidad en %")
    horizonte_temporal: int = Field(..., ge=1, le=365, description="Días a proyectar")
    cantidad_iteraciones: int = Field(..., ge=100, le=10000, description="Cantidad de iteraciones")


# ⚠ Esta respuesta está alineada con tu interface DatosSimulacion en React
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


def validar_entrada(param: ParametroEntrada) -> None:
    if param.monto_inicial <= 0:
        raise HTTPException(status_code=400, detail="Monto debe ser positivo")

    if not (0 <= param.volatilidad_anual <= 200):
        raise HTTPException(status_code=400, detail="Volatilidad debe estar entre 0% y 200%")

    if not (1 <= param.horizonte_temporal <= 365):
        raise HTTPException(status_code=400, detail="Horizonte debe estar entre 1 y 365 días")

    if not (100 <= param.cantidad_iteraciones <= 10000):
        raise HTTPException(status_code=400, detail="Iteraciones debe estar entre 100 y 10,000")


def simular_trayectorias(
    monto_inicial: float,
    volatilidad_anual: float,
    horizonte_temporal: int,
    cantidad_iteraciones: int,
):
    """
    Simulación tipo GBM sin NumPy: todo con listas y math/random.
    """

    # conversión a volatilidad diaria
    sigma_diaria = volatilidad_anual / 100.0 / math.sqrt(252.0)
    dt = 1.0 / 252.0
    mu = 0.0

    pasos = horizonte_temporal  # número de días
    simulaciones: List[List[float]] = []

    for _ in range(cantidad_iteraciones):
        precio = monto_inicial
        camino = [monto_inicial]
        for _ in range(pasos):
            z = random.gauss(0.0, 1.0)
            incremento = (mu - 0.5 * sigma_diaria**2) * dt + sigma_diaria * math.sqrt(dt) * z
            precio = precio * math.exp(incremento)
            camino.append(precio)
        simulaciones.append(camino)

    # Estadísticos por tiempo (columna)
    num_pasos = len(simulaciones[0])
    promedio: List[float] = []
    p5: List[float] = []
    p95: List[float] = []

    for j in range(num_pasos):
        columna = [trayectoria[j] for trayectoria in simulaciones]
        columna_ordenada = sorted(columna)
        n = len(columna_ordenada)

        # índices de percentiles (método simple)
        idx5 = max(0, min(n - 1, int(0.05 * (n - 1))))
        idx95 = max(0, min(n - 1, int(0.95 * (n - 1))))

        promedio.append(sum(columna) / n)
        p5.append(columna_ordenada[idx5])
        p95.append(columna_ordenada[idx95])

    # Métricas finales
    finales = [tray[-1] for tray in simulaciones]
    promedio_final = sum(finales) / len(finales)
    minimo_final = min(finales)
    maximo_final = max(finales)

    # VaR 95% (5% peor caso)
    cambios_pct = [(v - monto_inicial) / monto_inicial for v in finales]
    cambios_ordenados = sorted(cambios_pct)
    n = len(cambios_ordenados)
    idx_riesgo = max(0, min(n - 1, int(0.05 * (n - 1))))
    percentil_riesgo = cambios_ordenados[idx_riesgo]

    var_95 = monto_inicial * abs(percentil_riesgo)
    var_percentaje = percentil_riesgo * 100.0  # negativo = pérdida

    return (
        simulaciones,
        promedio,
        p5,
        p95,
        promedio_final,
        minimo_final,
        maximo_final,
        var_95,
        var_percentaje,
    )


# OJO: el archivo se llama api/simular.py
# → Vercel lo expone como /api/simular
# Por eso la ruta interna debe ser "/"
@app.post("/", response_model=SalidaAnalisis)
async def ejecutar_analisis(entrada: ParametroEntrada) -> SalidaAnalisis:
    """Ejecuta el análisis estocástico de Monte Carlo"""
    validar_entrada(entrada)

    (
        simulaciones,
        promedio,
        p5,
        p95,
        promedio_final,
        minimo_final,
        maximo_final,
        var_95,
        var_percentaje,
    ) = simular_trayectorias(
        entrada.monto_inicial,
        entrada.volatilidad_anual,
        entrada.horizonte_temporal,
        entrada.cantidad_iteraciones,
    )

    return SalidaAnalisis(
        simulaciones=simulaciones,
        promedio=promedio,
        percentil_5=p5,
        percentil_95=p95,
        precio_final_promedio=promedio_final,
        precio_final_minimo=minimo_final,
        precio_final_maximo=maximo_final,
        var_95=var_95,
        var_percentaje=var_percentaje,
    )


@app.get("/")
async def informacion():
    return {
        "nombre": "Motor Estocástico",
        "ruta": "/api/simular",
        "docs": "POST /api/simular",
    }

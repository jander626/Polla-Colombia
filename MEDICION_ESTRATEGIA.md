# Revisión de la estrategia — 24 de agosto de 2026

Encargo: una estrategia diaria, con **pocos indicadores**, precisa, y que el
backtest lo demuestre. Datos: IBKR, 30 instrumentos, 1.252 sesiones
(2021-08-26 → 2026-08-21).

## El protocolo, que es lo que hace que esto valga algo

Se reservaron **instrumentos Y fechas**, no solo fechas como hasta ahora:

|                        | fechas de descubrimiento | fechas reservadas |
|---|---|---|
| instrumentos descubrimiento | DESCUBRIMIENTO | fuera de tiempo |
| instrumentos reservados     | fuera de muestra | **NADA COMPARTIDO** |

El reparto de instrumentos va por hash del símbolo: elegirlos a mano habría
sido el primer sitio por donde colar el sobreajuste. Solo cuenta el cuadrante
inferior derecho — instrumentos que la regla no vio, en fechas que no vio.

Y **seis hipótesis declaradas de antemano** en vez de una rejilla de 162.
Cuantos más intentos, más fácil que uno parezca bueno por azar; con seis, el
listón de credibilidad es mucho más alto.

## Dos hallazgos antes de los resultados

**1. El backtest medía operaciones que nadie podía tomar.** En vivo, el bot no
repite señal sobre un instrumento con posición abierta
(`trading_bot._prepare_delivery`), pero `run_backtest` las simulaba todas. Una
sobreventa que dura tres sesiones contaba como tres operaciones, y las dos
últimas entran más caro. Estaba midiendo peor que la realidad, y en contra
justo de las reglas que disparan en racha. Corregido con
`BacktestParams.one_position_per_symbol`, y un test lo fija.

**2. La salida tiene que compartir tesis con la entrada.** Una entrada por
sobreventa dice "el precio se pasó y volverá": eso tiene horizonte de días y
objetivo cercano. Emparejarla con "aguanta a un movimiento de 3 ATR durante 30
días" son dos tesis distintas pegadas. Medido, cuesta **19 puntos de acierto**:

| Entrada `rsi2` con salida… | acierto | R media |
|---|---|---|
| tendencia (3 ATR, 30 días) | 24.4% | +0.185 |
| **media (1 ATR, 5 días)** | **43.0%** | **+0.188** |

## Resultado de las seis hipótesis

Cuadrante sin nada compartido, todas con la misma salida de tendencia:

| Regla | Indicadores | ops | acierto | exceso | límite inf. |
|---|---|---|---|---|---|
| L1 tendencia+RSI(2) | EMA200, RSI(2) | 144 | 36.1% | +0.492% | −0.585% |
| L2 ruptura 20d | máximo 20 sesiones | 138 | 60.1% | +0.079% | −1.511% |
| L3 ruptura en tendencia | SMA200 + máx 20 | 122 | 58.2% | −0.326% | −2.020% |
| L4 momentum 12-1 | SMA200 + mom 12-1 | 193 | 46.1% | −0.064% | −1.177% |
| L5 retroceso a la SMA50 | SMA200 + SMA50 | 75 | 34.7% | +0.077% | −1.081% |
| L6 sobreventa sin filtro | RSI(14) | 41 | 24.4% | +1.002% | −1.123% |
| S1 corto (espejo de L1) | EMA200, RSI(2) | 79 | 26.6% | −0.256% | −1.807% |
| S3 corto (ruptura) | SMA200 + mín 20 | 55 | 41.8% | −0.473% | −3.572% |

**Ninguna demuestra ventaja.** Todos los límites inferiores son negativos.

**L4 es la trampa que el protocolo cazó**: fue la única que pasó en
descubrimiento (límite inferior +0.022%) y se cayó en el cuadrante limpio
(−0.064%). Eso es exactamente lo que produce probar muchas cosas.

## Tres cosas que sí quedaron establecidas

**El acierto es un dial de la salida, no una medida de habilidad.** Con
objetivo cercano el acierto sube y la R baja; con objetivo lejano al revés. La
misma entrada da 24% o 43% según dónde se ponga el objetivo. Pedir "precisa"
sin mirar la R es pedir que se mueva ese dial.

**Los cortos vuelven a salir negativos**, ahora en un tercer diseño
independiente y con instrumentos reservados: −0.256% y −0.473% de exceso.

**La mejor candidata es L1 y hace falta más muestra, no más búsqueda.** Con la
muestra completa: 802 operaciones, exceso +0.239%, límite inferior −0.135%.
Para que ese límite toque cero harían falta **~1.959 operaciones, unos 12
años** al ritmo actual.

## Robustez de L1

No es un filo de cuchillo. El exceso es positivo en **todo** el rango probado:

| umbral RSI(2) | 5 | 10 | 15 | 20 | 25 |
|---|---|---|---|---|---|
| exceso | +0.273% | +0.239% | +0.342% | +0.361% | +0.245% |

| media larga | SMA100 | SMA150 | SMA200 | SMA250 | EMA200 |
|---|---|---|---|---|---|
| exceso | +0.174% | +0.249% | +0.239% | +0.249% | +0.288% |

Se dejan los valores centrales (RSI(2)<10, EMA200), no los mejores: elegir el
máximo de la meseta sería el sobreajuste otra vez.

Año a año: R media +0.11 (2022), +0.25 (2023), +0.21 (2024), +0.26 (2025),
−0.01 (2026 parcial). No vive de un tramo.

## Qué se implementó

`--regla reversion` activa la regla de dos indicadores con su salida corta.
Sigue **sin ser la de por defecto**: es mejor candidata que la actual en todo
lo medido, pero "mejor candidata" no es "demostrada".

| | retroceso (actual) | reversion (nueva) |
|---|---|---|
| Indicadores | 6 filtros + 7 componentes | 2 |
| Señales / día en 141 instrumentos | ~0.6 | ~3.1 |
| R media | +0.045 | +0.164 |
| Límite inferior del exceso | −0.607% | −0.207% |

Reproducible: `python -m tools.estudio`, `python -m tools.potencia`,
`python -m tools.robustez`, `python -m tools.salidas`.

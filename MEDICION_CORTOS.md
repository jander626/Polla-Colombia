# Medición del lado corto — 24 de agosto de 2026

Primera medición del lado corto sobre datos reales. Fuente: conector de IBKR
(velas diarias, 5 años). **Veredicto: no se habilita.**

## Qué se midió

| | |
|---|---|
| Universo | 18 instrumentos (17 acciones grandes + SPY) |
| Periodo | 2021-08-26 → 2026-08-21 (1.252 sesiones) |
| Fuente | IBKR `get_price_history`, `ONE_DAY`, 5 años |
| Sesión en curso | descartada (viene a medias) |

Es un **subconjunto**, no el universo de 141. La comparación con las cifras
históricas del proyecto (549 operaciones) no es directa.

## Resultado

| | Largos | Cortos |
|---|---|---|
| Señales generadas | 136 | **25** |
| Órdenes ejecutadas | 129 | **23** |
| Acierto | 26.4% | **17.4%** |
| R media | −0.071 | **−0.254** |
| Retorno medio por operación | −0.249% | **−0.933%** |
| Exceso sobre el nulo del mismo sentido | −0.157% | muestra insuficiente |
| Límite inferior del exceso | −1.003% | — |

Barrido de 162 mediciones (54 combinaciones × 3 salidas), ordenado por el 40%
final del periodo que ninguna vio al definirse:

- **Largos: 7 de 81 pasan validación.** Con 162 intentos al 5%, el azar
  produce ~8. Siete no es una señal, es lo que da el ruido.
- **Cortos: 0 de 81 pasan validación.** Y en todos, `val.ops = 0`.

## Por qué los cortos no son medibles en esta ventana

El régimen que necesitan —S&P bajo su media de 200— ocupó 219 de 1.252
sesiones (17.5%), y casi todo en un solo tramo: **junio de 2022 a enero de
2023**. Los demás episodios duran de 1 a 32 sesiones, demasiado poco para que
la estructura de un valor concreto gire.

Resultado: **19 de las 23 operaciones cortas son de 2022.** No son 23
observaciones independientes, son un episodio de mercado visto 23 veces. El
error estándar de `benchmark_comparison` (`excess.std()/sqrt(n)`) las trata
como independientes, así que cualquier intervalo que salga de ahí es
demasiado estrecho.

Y la ventana de validación (desde 2024-08-22) **no contiene ni un solo corto**.
No es un problema de tamaño de universo: meter los 141 instrumentos
multiplicaría las operaciones de 2022, pero seguiría sin haber ninguna en
2024-2026. Haría falta un histórico más largo (2018, 2020) para tener más de
un episodio bajista.

## Lo que juega a favor de los cortos y aun así no los salva

- **El sesgo de supervivencia va en su contra.** El universo son los líquidos
  de hoy: en corto es vender una cesta de supervivientes. El −0.93% real
  probablemente sea algo mejor. Pero no convierte 4 aciertos de 23 en ventaja.
- **El coste de mantener el corto no está modelado** (solo
  `round_trip_cost = 0.0010`), así que el número real sería PEOR, no mejor.

## Decisión

`--direction` existe y el motor está probado, pero el escaneo en vivo sigue en
`long`. Activar cortos con −0.254R medidos y sin una sola operación validable
sería repetir el error del porcentaje de confianza, esta vez con la deriva del
mercado en contra y con evidencia en contra en lugar de ausencia de evidencia.

Reproducible con:

```
python -m tools.medir_cortos     # largos vs cortos, con el nulo por sentido
python -m tools.barrido          # las 162 mediciones
python -m tools.resumen_cortos   # desglose por sentido
```

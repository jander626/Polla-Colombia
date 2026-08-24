# Contexto para Claude

Léeme antes de tocar nada. Este proyecto cambió de propósito por evidencia, y
sin saber por qué es muy fácil "arreglarlo" deshaciendo lo que costó descubrir.

## Qué es esto

Un **cribador**, no un generador de señales. Cada mañana, antes de la apertura
de EE.UU., revisa ~141 instrumentos operables en Quantfury y envía por Telegram
los que cumplen la regla activa, con la zona de entrada, el stop estructural y
el ratio riesgo/beneficio calculados. Desde el 24 de agosto de 2026 la regla
activa es `reversion` (EMA200 + RSI de 2 sesiones), no la de seis filtros —
ver [Revisión de la estrategia](#revisión-de-la-estrategia-24-de-agosto-de-2026).

El usuario ejecuta a mano en Quantfury (no hay API pública de trading). Habla
español, está en Colombia (UTC-5) y opera con dinero real.

## Lo más importante: no reintroduzcas el porcentaje de confianza

El bot publicaba "acierto histórico 32%, beneficio esperado +0.053R". Eso se
retiró el 16 de agosto de 2026 tras cuatro mediciones independientes sobre
cinco años:

| Medición | Resultado |
|---|---|
| Partición temporal | Ventaja en 2021-24 (`+0.088R`), **ninguna** en 2024-26 (`-0.013R`) |
| Barrido de 81 combinaciones | **0 de 81** demuestran ventaja en validación |
| Contra comprar y mantener el índice | Exceso `+0.145%`, límite inferior `-0.267%` |
| Diagnóstico por componentes | 30% del peso puntuaba **al revés**; solo el 18% predecía |

La tercera es la decisiva: de ese `+0.625%` que rendía una operación media,
`+0.480%` se conseguía comprando el índice esos mismos días. **El 77% del
resultado es exposición al mercado, no selección.** Demostrar el resto exigiría
~4.900 operaciones, unos 45 años al ritmo actual.

Publicar un porcentaje que se lee como probabilidad de acierto, sobre una
ventaja que no existe, era la única parte del bot capaz de costar dinero de
verdad. `_confidence_block` ya no existe; en su lugar está
`notify._evidence_line`, que dice lo que se sabe con el número delante.

Y los datos son buenos: se cruzaron con IBKR y los niveles calculados desde
ambas fuentes coinciden al 0.01%, con la puntuación (que mira 273 barras atrás)
al 0.55%. **La falta de ventaja no es un artefacto de datos sucios.**

## Reglas que no se relajan

- **Toda decisión ambigua del backtest cae en contra de la estrategia.** Vela
  que toca stop y objetivo → pérdida. Huecos → se ejecutan en la apertura, no
  en el nivel teórico. Si alguien relaja esto, el backtest empieza a mentir.
- **El límite inferior decide, no la media.** "Mayor que cero" no es
  "demostrado". Este error se cometió dos veces (en las alertas y en el
  veredicto de la partición temporal) y las dos veces costó caro.
- **El backtest y el escaneo en vivo llaman a las mismas funciones**
  (`compute_features` + `signals_from_features`). Si se separan, el backtest
  deja de decir nada sobre el comportamiento real. Hay un test que lo fija:
  `test_live_and_backtest_agree`.
- **Nada mira al futuro.** El trailing del backtest usa solo máximos de barras
  anteriores; hay un test con una vela que sube a 120 y baja a 99 el mismo día.
- **Un NaN no aprueba un filtro.** `bool(np.nan)` es `True` en Python; por eso
  existe `strategy._stage_ok`.

## Lo que hay que saber para trabajar aquí

**No hay datos de mercado en el sandbox.** La red bloquea Twelve Data, Yahoo y
Stooq (`000` en curl). Los backtests solo corren en GitHub Actions, y el
workflow lo lanza el usuario a mano — Claude recibe 403 al intentar dispararlo.

**Salvo que haya conector de IBKR.** Con él sí se pueden bajar velas diarias
desde la sesión (`get_price_history`, 5 años, OHLCV limpio) y medir en minutos
en vez de por ronda de clic. Ojo: las respuestas de `search_contracts` son
enormes y queman contexto rápido; usa identificadores conocidos cuando puedas
(AAPL 265598 · ABBV 118089500 · SPY 756733 · EUR/USD 12087792).

**El bot en producción no puede usar IBKR**: corre en GitHub Actions, donde no
hay conector. `data.py` se queda con Twelve Data.

**El cron de GitHub Actions no se respeta.** Está pedido cada 15 minutos y
dispara dos o tres veces al día a horas impredecibles. Por eso la ventana de
escaneo es ancha (06:00–09:25 de Nueva York) y la decisión de escanear se toma
en Python, no en el cron.

## Estado del seguimiento en vivo

Es ahora **la única medición que no viene de un backtest**, y por tanto la
única que puede llegar a demostrar algo. Protégela.

El usuario opera a mano, así que cada candidato tiene tres desenlaces y el
simulador solo modela uno:

1. Lo toma y sale por objetivo o stop → el seguimiento lo resuelve bien solo.
2. Lo toma y **sale cuando quiere** → el seguimiento registraría una salida
   ficticia. Pasó con ABBV (cerrada a mano en 266, objetivo en 268.27).
3. **No lo toma** → el seguimiento anotaría una operación que nadie tuvo. Pasó
   con F, marcada `status: "not_taken"`.

Los casos 2 y 3 se corregían editando `trading_state.json` a mano. Desde el 24
de agosto de 2026 se cuentan por Telegram: `/tomada SÍMBOLO PRECIO`,
`/cerrar SÍMBOLO PRECIO` y `/paso SÍMBOLO`, con `TradingState.mark_taken`,
`close_manually` y `mark_skipped` detrás.

Cuatro reglas de ese código que no son adorno:

- **Sin precio de entrada no se cierra.** `/cerrar` sobre algo que nunca pasó
  por `/tomada` se niega y pide el dato (`/cerrar TXN 291,40 282,10`).
  Rellenarlo con el techo de la zona daría un número creíble y falso, que es
  peor que no dar ninguno.
- **`/tomada` no reemplaza al simulador, solo el precio.** La salida la sigue
  decidiendo `simulate_signal`; lo único que cambia es contra qué entrada se
  miden R y retorno, y el precio simulado se conserva en
  `simulated_entry_price` para poder auditar la diferencia.
- **Una entrada tomada fuera de la zona no caduca.** Si el usuario dice que
  entró donde el simulador nunca habría ejecutado, la operación se queda
  abierta esperando `/cerrar` en vez de irse al histórico como `expired` con
  la posición todavía viva. Manda el usuario, no el simulador.
- **Una entrada por debajo del stop se rechaza.** Habría nacido cerrada: es un
  dedazo, y aceptarlo en silencio envenena la única medición real que hay.

`/paso` no cuenta como fallo —no se arriesgó dinero— y sale en `/rendimiento`
como "candidatos que dejaste pasar", que es el número que impide leer el
rendimiento como si describiera lo que hizo el usuario.

## Cortos: motor listo, sin medir

El 24 de agosto de 2026 se añadió el lado corto (`StrategyParams.direction`).
**No es una estrategia nueva: es la misma reflejada** — tendencia bajista,
rebote, y vuelta a caer. Está escrito así a propósito; dos cuerpos de reglas
independientes se habrían desincronizado a la primera.

**Nada de esto se ha medido todavía.** No hay datos de mercado en el sandbox y
el conector de IBKR pide autenticación que una sesión no interactiva no puede
dar. Por eso `--direction` existe pero el valor por defecto es `long` en vivo:
activar cortos sin medición sería repetir el error del porcentaje de
confianza, y esta vez con la deriva del mercado en contra.

Lo que protege el reflejo:

- **La propiedad del espejo.** `tests/test_short.py` refleja series de precios
  enteras (`p' = K - p`) y exige que largo y corto den el MISMO desenlace y la
  MISMA R al decimal, en siete escenarios (huecos, vela ambigua, caducidad,
  salida por tiempo) y con trailing. Un solo `<` sin girar rompe esa igualdad.
- **El nulo lleva el sentido de la operación.** En `benchmark_comparison`, el
  nulo de un corto es VENDER el índice, no comprarlo. Comparar un corto contra
  comprar el índice mediría el signo del mercado, no la selección — y habría
  invalidado justo la medición que convirtió esto en un cribador.
- **La calibración no se mezcla.** `direction` entra en `signature`, y el
  workflow solo escribe `calibration.json` cuando `direction=long`.
- **Los registros antiguos siguen siendo largos.** `direction` se lee con
  default `"long"`, así que nada de lo ya guardado cambia de sentido.

Lo que hay que saber antes de creerse un backtest de cortos:

- **El régimen invertido deja pocos días operables.** El S&P pasa la mayor
  parte del tiempo sobre su media de 200, así que los cortos salen agrupados
  en dos o tres episodios. Operaciones agrupadas no son independientes, y el
  error estándar de `benchmark_comparison` (`excess.std()/sqrt(n)`) las trata
  como si lo fueran: el límite inferior saldrá **demasiado optimista**. Hay
  que descontarlo a ojo o corregirlo antes de decidir nada.
- **El coste de mantener un corto no está modelado.** El único coste es
  `round_trip_cost = 0.0010`. Lo que Quantfury cobre por noche —y el modelo
  aguanta 30 días— no está ahí. Averiguarlo antes de leer resultados.
- **El sesgo de supervivencia cambia de lado.** El universo son los líquidos
  de hoy: en corto eso es vender una cesta de supervivientes, y los desastres
  no están. Juega a FAVOR de los cortos, al revés que en largo.
- **Barrer los dos sentidos son 162 mediciones.** Cuantos más intentos, más
  fácil que uno parezca bueno por azar. El veredicto lo da el límite inferior
  en validación, nunca la media en entrenamiento.

Cómo medirlo cuando haya datos:

```
python trading_bot.py backtest --direction short --years 5
python trading_bot.py backtest --direction both --search   # 162 mediciones
```

O el workflow `Backtest y calibración` con el input `direction`.

## Revisión de la estrategia (24 de agosto de 2026)

El usuario pidió una estrategia diaria, con pocos indicadores, precisa, y que
el backtest lo demuestre. Se midieron seis hipótesis declaradas de antemano
—no una rejilla— con datos reales de IBKR, reservando **instrumentos Y
fechas**, no solo fechas: el reparto de instrumentos va por hash del símbolo
para que no lo elija quien mide. El detalle completo, con las ocho tablas de
resultados, está en `MEDICION_ESTRATEGIA.md`.

**Ninguna hipótesis demuestra ventaja.** La mejor —`reversion`, EMA200 + RSI(2)
de dos sesiones— tiene el límite inferior del exceso negativo con la muestra
disponible: para que toque cero harían falta ~1.959 operaciones, unos 12
años. Es mejor candidata que la regla vieja en todo lo medido (R media, límite
inferior, robustez a los umbrales), pero "mejor candidata" no es "demostrada".

Se activó igualmente como regla por defecto en vivo (`StrategyParams.entry_rule
= "reversion"` vía la CLI, ver `trading_bot._rule_params`) por una razón
distinta a la ventaja: dispara ~3 veces al día en vez de ~0.6. El seguimiento
en vivo es la única medición que no viene de un backtest; a 0.6 señales
diarias, demostrar algo con él llevaría décadas. A 3, meses.

Dos fallos de medición se corrigieron de camino, y son los que importan más
que el resultado en sí:

- **El backtest simulaba operaciones que en vivo nunca se habrían enviado.**
  El escaneo no repite señal sobre un símbolo con posición abierta, pero
  `run_backtest` las simulaba todas. `BacktestParams.one_position_per_symbol`
  (default `True`) y `backtest.simulate_sequence` lo arreglan; hay un test que
  fija que `run_backtest` y `run_search` usan la misma regla.
- **La salida tiene que compartir tesis con la entrada.** Una entrada por
  reversión dice "esto se pasó y va a volver": eso es un horizonte de días,
  no de un movimiento de tendencia a 30 días. Por eso `reversion_params()` y
  `reversion_backtest()` (en `config.py`) van SIEMPRE juntas — hay un test
  que lo fija (`test_la_salida_va_atada_a_la_entrada`).

**La ficha de Telegram describe la regla que disparó de verdad.** Antes de
que `Signal` llevara `entry_rule`, una señal de `reversion` se anunciaba con
"tocó la EMA20" y "MACD girando" —cosas que esa regla no mira—, porque el
texto asumía siempre el retroceso de seis filtros. `notify._filters_passed`
ahora rama por `signal.entry_rule`, y el objetivo del mensaje (`_target_label`)
se calcula desde los niveles en vez de estar fijo en "3 ATR".

### La regla de decisión, fijada antes de ver el resultado

`config.LIVE_DECISION_SAMPLE = 200`. Con menos operaciones cerradas,
`/rendimiento` solo muestra progreso. Con 200, el límite inferior de la R
media (`risk.mean_lower_bound`) decide: positivo, sigue; si no, se apaga y el
dinero va al índice. Se fijó el umbral y el criterio ANTES de tener el
resultado a propósito — es la única forma de que dentro de unos meses la
decisión no se reescriba alrededor del número que haya salido.

**No cambies este umbral, ni el criterio, para que un resultado concreto
quede del lado que se prefiera.** Si hay que revisarlo, que sea con una razón
metodológica escrita aquí antes de mirar `/rendimiento`, no después.

### Sobre seguir buscando

Van 162 (barrido direccional) + 81 (barrido de seis filtros) + 8 (hipótesis
declaradas) mediciones. Con suficientes intentos, algo acaba pareciendo bueno
por azar — el momentum 12-1 lo demostró dentro del propio experimento: pasó en
descubrimiento (+0.022%) y se cayó en el cuadrante sin nada compartido
(−0.064%). **Antes de proponer una hipótesis nueva, hay que tener una tesis
económica para ella, no solo curiosidad por ver qué rinde mejor.** La
respuesta que falta no está en más búsqueda sobre el mismo histórico: está en
los meses que vienen, y para eso existe la regla de decisión de arriba.

### Dónde está la ventaja de verdad

Con una ventaja de ~0.2% por operación —si resulta real—, el resultado de la
cuenta lo decide el tamaño de posición y respetar el stop, no qué regla
dispara. Esto no es un llamado a escribir código: es la razón por la que
seguir afinando la señal, más allá de un punto, deja de mover la aguja.

## Cómo se trabaja

- Rama de desarrollo: `claude/trading-bot-telegram-alerts-0l55mu`. Se abre PR y
  se fusiona (el usuario pidió que Claude lo haga: "Fusiona el PR tu. Yo no sé
  cómo hacerlo").
- `pytest tests/` antes de cada push. Están en español y documentan **por qué**
  existe cada regla, casi siempre porque falló en producción.
- Los comentarios del código explican decisiones, no mecánica. Si quitas una
  regla, quita también el comentario que la justifica — o mejor, no la quites.

## Advertencia sobre expectativas

El usuario preguntó por un 5-10% mensual con apalancamiento. No es alcanzable:
con sus propios números eso implica una caída máxima que arruina la cuenta, y
supera con holgura al mejor fondo de la historia. Ya se le dijo con los números
delante. No lo prometas de otra forma.

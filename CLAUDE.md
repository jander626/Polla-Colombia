# Contexto para Claude

Léeme antes de tocar nada. Este proyecto cambió de propósito por evidencia, y
sin saber por qué es muy fácil "arreglarlo" deshaciendo lo que costó descubrir.

## Qué es esto

Un **cribador**, no un generador de señales. Cada mañana, antes de la apertura
de EE.UU., revisa ~141 instrumentos operables en Quantfury y envía por Telegram
los que cumplen seis filtros de retroceso en tendencia alcista, con la zona de
entrada, el stop estructural y el ratio riesgo/beneficio calculados.

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

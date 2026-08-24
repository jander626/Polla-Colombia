# Cribador de Quantfury — acciones y forex

Revisa el mercado una vez al día, antes de la apertura de Estados Unidos, y
manda por Telegram los instrumentos operables en Quantfury que están en
tendencia y muy estirados en sentido contrario (EMA200 + RSI de 2 sesiones),
con la zona de entrada, el stop estructural y el ratio riesgo/beneficio ya
calculados.

> **No emite señales con ventaja demostrada, y no publica ningún porcentaje de
> confianza.** Nació haciendo ambas cosas; dejó de hacerlas cuando cuatro
> mediciones independientes coincidieron en que la ventaja no existe (ver
> [Por qué es un cribador](#por-qué-es-un-cribador-y-no-un-generador-de-señales)).
> Lo que aporta es reducir 141 gráficos a unos pocos candidatos con sus
> niveles hechos. La decisión es tuya.

> **La ganancia grande, si existe, no está en la señal.** Con una ventaja de
> ~0.2% por operación —si es real—, lo que decide si la cuenta sobrevive es el
> tamaño de posición y respetar el stop, no qué indicador dispara. Un solo
> stop saltado a mano borra meses de ventaja teórica.

> **Esto no es asesoría financiera.**

---

## Cómo funciona

Entre las **06:00 y las 09:25 hora de Nueva York** el bot descarga velas
diarias de ~141 instrumentos (acciones líquidas de NYSE/NASDAQ, cuatro ETFs y
los 14 pares de divisas de Quantfury), calcula indicadores y aplica la regla
activa. De los que sobreviven se queda con los mejores, los pasa por un
filtro de noticias y manda la ficha.

**Hay días que no llega nada, y es correcto.** Un cribador que encuentra algo
todos los días no está cribando. Cuando no envía nada, el mensaje incluye el
embudo —cuántos instrumentos sobrevivieron a cada etapa— para que un silencio
no sea indistinguible de una avería.

### La estrategia

Desde el **24 de agosto de 2026** el cribador en vivo usa `--regla reversion`
por defecto: dos indicadores, no seis filtros. Se cambió tras medir seis
hipótesis declaradas de antemano con instrumentos Y fechas reservados —no
solo fechas— y salir mejor en todo lo comparable (detalle completo en
[`MEDICION_ESTRATEGIA.md`](MEDICION_ESTRATEGIA.md)):

| Regla | Condición |
|---|---|
| Régimen de fondo | `Close > EMA200` (en corto, `Close < EMA200`) |
| Reversión extrema | `RSI(2) < 10` (en corto, `RSI(2) > 90`) |
| Liquidez | Volumen medio en dólares por encima del umbral |
| Volatilidad sana | `ATR/precio` ni muerto ni caótico |

La salida va atada a la entrada y no es la misma que antes: **objetivo a 1
ATR**, plazo máximo de **5 días**, sin suelo de ratio riesgo/beneficio.
Separar la salida de la entrada —objetivo a 3 ATR, 30 días, como pedía la
regla vieja— cuesta 19 puntos de acierto medidos: una entrada por reversión
dice "el precio se pasó y volverá", y eso tiene horizonte de días, no de un
movimiento de tendencia de un mes.

La regla anterior —seis filtros: régimen, fuerza de tendencia (ADX), retroceso
a la EMA20, reanudación, liquidez y volatilidad, con objetivo a 3 ATR y R:B
mínimo de 1.5— sigue disponible con `--regla retroceso`, para comparar o
volver atrás. Ninguna de las dos regla tiene ventaja demostrada todavía: con
`reversion`, el límite inferior del exceso sobre el índice es negativo. Lo que
cambió es que da ~3 señales al día en vez de ~0.6, así que demostrarla con el
seguimiento en vivo es cuestión de meses y no de décadas.

En los dos casos, los niveles salen del ATR: la **entrada es un límite**
condicional (un techo en largo, un suelo en corto; si el precio abre con un
hueco fuerte a favor, la operación no se ejecuta) y el **stop** se apoya en la
estructura del precio.

Esta parte sigue siendo útil aunque la regla activa no tenga ventaja
demostrada: el stop estructural y el ratio están bien calculados, y son la
mitad de la información que hace falta para decidir.

## Por qué es un cribador y no un generador de señales

El proyecto empezó publicando un porcentaje de confianza calibrado contra el
backtest. Esa cifra se retiró el **16 de agosto de 2026**, cuando cuatro
mediciones independientes sobre cinco años de datos coincidieron:

| Medición | Resultado |
|---|---|
| Partición temporal | Ventaja en 2021-24 (`+0.088R`), **ninguna** en 2024-26 (`-0.013R`) |
| Barrido de 81 combinaciones de parámetros | **0 de 81** demuestran ventaja en validación |
| Contra comprar y mantener el índice | Exceso `+0.145%` por operación, límite inferior `-0.267%` |
| Diagnóstico de la puntuación | 30% del peso puntuaba **al revés**; solo el 18% predecía |

La tercera es la decisiva. De ese `+0.625%` que rendía una operación media,
`+0.480%` lo habrías tenido comprando el índice esos mismos días: el **77% del
resultado es exposición al mercado, no selección**. Y con la dispersión medida,
demostrar el exceso restante exigiría unas 4.900 operaciones — **unos 45 años**
al ritmo actual. Un edge que necesita 45 años para probarse es indistinguible
de cero en cualquier horizonte útil.

Publicar un "32% de confianza" sobre eso era la única parte del bot capaz de
costar dinero de verdad: se lee como una probabilidad de acierto, y detrás no
había nada que la sostuviera.

**Lo que queda en su lugar** es una frase que dice lo que se sabe, con el
número delante:

```
En el histórico reciente (2024-07→2026-08) estos filtros NO muestran ventaja
demostrada (-0.007R sobre 218 operaciones). Criba, no recomendación: los
niveles están calculados, la decisión es tuya.
```

Si algún día el histórico reciente vuelve a mostrar ventaja, la misma línea lo
dirá — y seguirá llamándose criba.

### La medición sigue viva

`calibration.json` se sigue generando y el seguimiento de resultados también:
son lo que permitirá detectar un cambio. El `/rendimiento` en vivo es además
la única medición que no viene de un backtest, y por eso la única que puede
llegar a demostrar algo. Hacen falta cientos de operaciones antes de que
signifique nada.

---

## Puesta en marcha

### 1. Secretos en GitHub

`Settings → Secrets and variables → Actions`:

| Secreto | Para qué | Obligatorio |
|---|---|---|
| `TWELVEDATA_API_KEY` | Datos de mercado ([gratis](https://twelvedata.com/pricing)) | Sí |
| `TELEGRAM_BOT_TOKEN` | Envío de las fichas (vía [@BotFather](https://t.me/BotFather)) | Sí |
| `GEMINI_API_KEY` | Filtro de noticias ([AI Studio](https://aistudio.google.com/apikey)) | No |
| `TELEGRAM_CHAT_ID` | Chat por defecto | No |

Sin `GEMINI_API_KEY` el bot funciona igual, pero las fichas salen marcadas
como *sin verificación de noticias*.

### 2. Calibrar

El backtest ya no habilita ninguna promesa, pero sigue midiendo qué hacen
estos filtros y genera la línea de evidencia que aparece en cada ficha:

`Actions → Backtest y calibración → Run workflow`

Tarda unos 20 minutos (el plan gratuito limita a 8 llamadas por minuto). Al
terminar publica el informe en el resumen del job y commitea
`calibration.json`. El informe incluye la comparación contra comprar y mantener
el índice, la partición temporal y el diagnóstico por componentes.

Con `search = true` barre 81 combinaciones de parámetros de una sola pasada y
las ordena por el tramo de validación, no por el resultado total.

### 3. Activar

El workflow `Cribador diario` corre solo de lunes a viernes. Escribe
`/start` al bot en Telegram para registrar tu chat.

---

## Comandos de Telegram

| Comando | Qué hace |
|---|---|
| `/senales` | Últimos candidatos resueltos |
| `/abiertas` | Operaciones en curso |
| `/rendimiento` | Acierto real frente a la confianza prometida |
| `/instrumentos` | Universo vigilado |
| `/ayuda` | Ayuda |

`/rendimiento` es el control externo de la calibración. Si el acierto real
queda sistemáticamente por debajo de lo prometido, la tabla está mal y hay que
rehacer el backtest.

### Contarle lo que hiciste de verdad

El bot no ve tu cuenta de Quantfury. Sin estos tres comandos da por hecho que
tomaste todo lo que te mandó y que saliste donde decía el plan, y eso mete
operaciones inventadas en la única medición que no viene de un backtest.

| Comando | Cuándo |
|---|---|
| `/tomada SÍMBOLO PRECIO` | Entraste. Registra tu precio real de entrada |
| `/cerrar SÍMBOLO PRECIO` | Saliste antes de tocar objetivo o stop |
| `/paso SÍMBOLO` | No llegaste a tomarla |

```
/tomada TXN 282,10
/cerrar TXN 291,40
/paso NVDA
```

Detalles que importan:

- El precio se lee con coma o con punto: `282,10` y `282.10` valen igual.
- Si nunca registraste la entrada, dala en el mismo mensaje —primero la
  salida, después la entrada—: `/cerrar TXN 291,40 282,10`. Sin entrada el bot
  se niega a cerrar en vez de inventar el precio con el techo de la zona.
- `/paso` saca el candidato del seguimiento **sin** contarlo como fallo: no se
  arriesgó dinero en él. Aparece en `/rendimiento` como "candidatos que
  dejaste pasar".
- La R se mide siempre contra el riesgo planificado (techo de la zona menos
  stop), igual que en el backtest, porque es el que conocías al dimensionar.
- Si dices que entraste por encima del techo de la zona, queda registrado
  —pasó de verdad—, pero el simulador no puede seguir una orden que nunca
  habría ejecutado: esa operación solo se cierra con `/cerrar`.

Antes de esto, los tres casos se corregían editando `trading_state.json` a
mano. Así se arreglaron ABBV (cerrada a mano en 266 con el objetivo en 268.27)
y F (enviada pero no tomada).

---

## Uso local

```bash
pip install -r requirements-dev.txt
export TWELVEDATA_API_KEY=...

python trading_bot.py backtest --years 5 --write   # calibrar
python trading_bot.py scan --force --dry-run       # escanear sin enviar nada
python trading_bot.py track                        # revisar operaciones abiertas
python -m pytest tests/                            # 175 tests
```

---

## Limitaciones

Conviene tenerlas presentes antes de poner dinero:

- **Objetivo realista**: 45-58% de aciertos con riesgo/beneficio cercano a 2.
  Rentable en el agregado, pero con rachas de 6-8 pérdidas seguidas que son
  estadísticamente normales y se sienten fatal.
- **Todo backtest sobrestima.** No captura el slippage real ni el riesgo de no
  colocar la orden a tiempo.
- **Sesgo de supervivencia.** El universo son los instrumentos líquidos de hoy,
  así que el histórico sale mejor de lo que habría sido en vivo. El informe lo
  advierte explícitamente.
- **Quantfury no tiene API pública de trading**, así que la ejecución es manual.
  El bot no toca tu cuenta.

---

## Arquitectura

```
trading/
  config.py       Umbrales, horarios y festivos de NYSE
  universe.py     Instrumentos de Quantfury + filtro por volumen real
  data.py         Proveedor intercambiable (Twelve Data → yfinance) con caché
  indicators.py   EMA, RSI, MACD, ATR, ADX en pandas puro
  strategy.py     Filtros y puntuación 0-100
  risk.py         Niveles y calibración de Wilson
  backtest.py     Simulación y generación de calibration.json
  llm_filter.py   Riesgo de evento con Gemini (solo resta)
  notify.py       Telegram y formato de mensajes
  state.py        Estado y seguimiento de señales
  schedule.py     Ventana horaria con zonas horarias reales
```

Dos invariantes sostienen la credibilidad del sistema, y hay tests que fallan
si se rompen:

- **El backtest y el escaneo en vivo comparten las mismas funciones.** Si
  divergieran, la calibración dejaría de decir nada sobre el comportamiento real.
- **La estrategia no puede ganar dinero sobre ruido puro.** `test_null_hypothesis`
  corre la estrategia sobre paseos aleatorios sin deriva y exige que la
  esperanza no sea significativamente positiva. Un backtest con look-ahead
  aprobaría cualquier estrategia; este test lo detecta.

---

Este repositorio alojaba antes un bot de pronósticos del Mundial 2026. Se
retiró al dejar de necesitarse; su capa de Telegram —troceado de mensajes y
reintento en texto plano cuando el Markdown falla— sobrevive en `notify.py`.
Sigue disponible en el historial de git.

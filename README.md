# Bot de señales de trading — acciones y forex

Analiza el mercado una vez al día, antes de la apertura de Estados Unidos, y
avisa por Telegram de oportunidades **solo de compra** en instrumentos
operables en Quantfury, con precio de entrada, objetivo, stop, ratio
riesgo/beneficio y un porcentaje de confianza calibrado.

> **Esto no es asesoría financiera.** Es una herramienta de cribado que evita
> mirar 140 gráficos cada mañana. Las órdenes las colocas tú.

---

## Cómo funciona

A las **08:30 hora de Nueva York** el bot descarga velas diarias de ~141
instrumentos (acciones líquidas de NYSE/NASDAQ, cuatro ETFs y los 14 pares de
divisas de Quantfury), calcula indicadores y aplica seis filtros. De los que
sobreviven se queda con los mejores, los pasa por un filtro de noticias y
manda la alerta.

**Hay días que no llega nada, y es correcto.** Un bot que encuentra
oportunidades todos los días se las está inventando.

### La estrategia

Retroceso en tendencia alcista: no comprar rupturas ni adivinar suelos, sino
esperar a que algo que ya sube se tome un descanso y entrar cuando reanuda.

| Filtro | Condición |
|---|---|
| Régimen alcista | `Close > EMA200` y `EMA50 > EMA200` |
| Fuerza de tendencia | `ADX(14) > 20` |
| Retroceso real | El RSI cayó de 45 y el precio visitó la zona de la EMA20 |
| Reanudación | RSI cruzando al alza, MACD girando o cierre sobre el máximo previo |
| Liquidez | Volumen medio en dólares por encima del umbral |
| Volatilidad sana | `ATR/precio` ni muerto ni caótico |

Los niveles salen del ATR: la **entrada es un techo** (si abre con hueco
alcista, la operación no se ejecuta), el **stop** se apoya en el mínimo del
retroceso y el **objetivo** está a 3 ATR. Cualquier señal por debajo de 1.5 de
ratio riesgo/beneficio se descarta.

### El porcentaje de confianza

Es la decisión de diseño central: **no lo inventa un modelo de lenguaje.**

Pedirle a un LLM "dame la probabilidad de que esto funcione" devuelve un
número plausible y ficticio, que es peor que no dar ninguno porque invita a
confiar en él. En su lugar:

1. El backtest simula la estrategia sobre años de historia.
2. Agrupa las señales por tramo de puntuación y mide el acierto real de cada uno.
3. Publica el **límite inferior del intervalo de Wilson**, que castiga la falta
   de muestra: 2 aciertos de 3 no se publican como "67%".
4. Gemini solo puede **restar** confianza por riesgo de evento. Nunca sumar.

Cuando la alerta dice 64%, significa algo comprobable: *de las señales
históricas con este perfil, el 64% alcanzó el objetivo antes que el stop.*

Sin `calibration.json` las alertas salen marcadas como **sin calibrar**, y eso
es deliberado: un bot recién instalado no ha demostrado nada.

---

## Puesta en marcha

### 1. Secretos en GitHub

`Settings → Secrets and variables → Actions`:

| Secreto | Para qué | Obligatorio |
|---|---|---|
| `TWELVEDATA_API_KEY` | Datos de mercado ([gratis](https://twelvedata.com/pricing)) | Sí |
| `TELEGRAM_BOT_TOKEN` | Envío de alertas (vía [@BotFather](https://t.me/BotFather)) | Sí |
| `GEMINI_API_KEY` | Filtro de noticias ([AI Studio](https://aistudio.google.com/apikey)) | No |
| `TELEGRAM_CHAT_ID` | Chat por defecto | No |

Sin `GEMINI_API_KEY` el bot funciona igual, pero las alertas salen marcadas
como *sin verificación de noticias*.

### 2. Calibrar

**Antes de fiarte de ninguna alerta**, ejecuta el backtest:

`Actions → Backtest y calibración → Run workflow`

Tarda unos 20 minutos (el plan gratuito limita a 8 llamadas por minuto). Al
terminar publica el informe en el resumen del job y commitea
`calibration.json`. Si el R medio sale negativo, el propio informe lo dice: hay
que ajustar parámetros antes de seguir.

### 3. Activar

El workflow `Bot de señales de trading` corre solo de lunes a viernes. Escribe
`/start` al bot en Telegram para registrar tu chat.

---

## Comandos de Telegram

| Comando | Qué hace |
|---|---|
| `/senales` | Últimas señales cerradas |
| `/abiertas` | Operaciones en curso |
| `/rendimiento` | Acierto real frente a la confianza prometida |
| `/instrumentos` | Universo vigilado |
| `/ayuda` | Ayuda |

`/rendimiento` es el control externo de la calibración. Si el acierto real
queda sistemáticamente por debajo de lo prometido, la tabla está mal y hay que
rehacer el backtest.

---

## Uso local

```bash
pip install -r requirements-dev.txt
export TWELVEDATA_API_KEY=...

python trading_bot.py backtest --years 5 --write   # calibrar
python trading_bot.py scan --force --dry-run       # escanear sin enviar nada
python trading_bot.py track                        # revisar operaciones abiertas
python -m pytest tests/                            # 160 tests
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

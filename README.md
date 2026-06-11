# 🌍 Bot de Pronósticos — Mundial 2026

Bot que analiza y pronostica partidos del Mundial 2026 usando inteligencia artificial. Corre automáticamente en GitHub Actions (sin servidor) y te notifica por Telegram **24 horas antes de cada partido**.

---

## ¿Qué hace?

- ✅ Envía pronósticos automáticos 24h antes de cada partido
- ✅ Analiza forma reciente, noticias, lesiones y estadísticas con Claude AI
- ✅ Indica resultado más probable (local/empate/visitante), marcador y confianza
- ✅ Maneja la fase de grupos Y las eliminatorias (los equipos que clasifiquen se toman automáticamente del API)
- ✅ Responde comandos manuales por Telegram:
  - `/proximos` — lista los próximos 10 partidos
  - `/pronostico <número o equipo>` — pide el análisis de un partido específico
  - `/ayuda` — muestra todos los comandos

---

## Configuración (una sola vez)

Necesitas configurar **3–4 secretos** en GitHub. Sigue estos pasos:

### Paso 1 — Crear el bot de Telegram

1. Abre Telegram y busca **@BotFather**
2. Envía `/newbot`
3. Pon un nombre (ej. `Mi Bot del Mundial`) y un username (ej. `mibot_mundial_bot`)
4. BotFather te dará un **token** con formato `123456789:ABCdefGHI...` → este es tu `TELEGRAM_BOT_TOKEN`

5. Abre una conversación con tu nuevo bot y envía cualquier mensaje (ej. `/start`).  
   Después ve a: `https://api.telegram.org/bot<TU_TOKEN>/getUpdates`  
   En el JSON, busca `"chat":{"id":...}` → ese número es tu `TELEGRAM_CHAT_ID`

### Paso 2 — Obtener la API de football-data.org

1. Ve a [football-data.org](https://www.football-data.org/client/register) y crea una cuenta gratuita
2. Confirma tu email y obtén tu **API key** en el dashboard
3. El plan gratuito incluye acceso al Mundial → `FOOTBALL_DATA_API_KEY`

### Paso 3 — Obtener la API de Anthropic (Claude)

1. Ve a [console.anthropic.com](https://console.anthropic.com) y crea una cuenta
2. En *API Keys*, genera una nueva clave → `ANTHROPIC_API_KEY`
3. Requiere agregar créditos (el Mundial completo (~104 partidos) cuesta aprox. $3–5 USD)

### Paso 4 — Agregar los secretos en GitHub

1. Ve a tu repositorio en GitHub
2. Clic en **Settings** → **Secrets and variables** → **Actions**
3. Agrega estos 4 secretos uno por uno con **New repository secret**:

| Nombre del secreto | Valor |
|---|---|
| `TELEGRAM_BOT_TOKEN` | El token de @BotFather |
| `TELEGRAM_CHAT_ID` | Tu chat ID de Telegram |
| `FOOTBALL_DATA_API_KEY` | Tu API key de football-data.org |
| `ANTHROPIC_API_KEY` | Tu API key de Anthropic |

### Paso 5 — Activar GitHub Actions

1. Ve a la pestaña **Actions** de tu repositorio
2. Si aparece un aviso "Workflows disabled", clic en **Enable** para activarlos
3. Para probar que todo funciona: clic en **World Cup Forecast Bot** → **Run workflow** → selecciona `test` como modo → **Run workflow**

Si el bot está configurado correctamente, recibirás un mensaje en Telegram: ✅ *Bot de pronósticos activo.*

---

## Cómo funciona

```
Cada 30 minutos (GitHub Actions cron)
    │
    ├── Revisa Telegram para comandos manuales (/proximos, /pronostico, etc.)
    │
    └── Revisa si hay partidos en las próximas 24h
            │
            └── Para cada partido nuevo → Claude AI analiza con web search
                        │
                        └── Envía pronóstico por Telegram
                                    │
                                    └── Guarda en state.json para no repetir
```

### Formato del pronóstico

```
🌍 PRONÓSTICO MUNDIAL 2026
━━━━━━━━━━━━━━━━━━━━
⚽ Colombia vs Argentina
📅 15/06/2026 18:00 UTC
🏆 GROUP_STAGE Grupo B
━━━━━━━━━━━━━━━━━━━━

🎯 RESULTADO MÁS PROBABLE: 2 (visitante)
⚽ MARCADOR PROBABLE: 1-2
📊 CONFIANZA: 62%
🔑 FACTORES CLAVE
  • Argentina llega con 3 victorias consecutivas
  • Colombia sin su mediocampista titular por lesión
  • Historial favorece a Argentina en torneos mayores
📝 RESUMEN: Argentina muestra mejor forma...
```

---

## Ejecución manual

Si quieres lanzar el bot manualmente sin esperar el cron:

1. Ve a **Actions** → **World Cup Forecast Bot**
2. Clic en **Run workflow**
3. Selecciona el modo:
   - `auto` — modo normal (detecta partidos + procesa comandos)
   - `poll` — solo procesa comandos de Telegram
   - `test` — envía mensaje de prueba

---

## Estructura del proyecto

```
├── bot.py                          # Script principal
├── requirements.txt                # Dependencias Python
├── state.json                      # Estado persistente (pronósticos enviados)
└── .github/
    └── workflows/
        └── forecast.yml            # Workflow de GitHub Actions
```

---

## Preguntas frecuentes

**¿Necesito un servidor?**  
No. Todo corre en GitHub Actions de forma gratuita (hasta 2000 minutos/mes en planes free).

**¿Cuánto cuesta?**  
Solo la API de Claude: aprox. $3–5 USD para el Mundial completo. El resto es gratuito.

**¿Me llega si el partido es a las 3am?**  
Sí. El pronóstico se envía 24h antes, así que si el partido es a las 3am, lo recibirías a las 3am del día anterior (hora UTC). Puedes ajustar `FORECAST_WINDOW_HOURS` en `bot.py`.

**¿Qué pasa con los partidos de eliminatoria que aún no tienen equipos definidos?**  
La API de football-data.org actualiza automáticamente los equipos cuando se conocen, y el bot los recogerá en el siguiente ciclo.

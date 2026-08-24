"""Tests de las correcciones manuales del seguimiento: /tomada, /cerrar, /paso.

El usuario coloca las órdenes a mano en Quantfury, así que el bot solo conoce
de su cuenta lo que él le cuente. Antes de estos comandos había tres
desenlaces y el seguimiento modelaba uno:

1. La toma y sale por objetivo o stop → el simulador acierta solo.
2. La toma y **sale cuando quiere** → el simulador anotaba una salida
   ficticia. Pasó con ABBV, cerrada a mano en 266 con el objetivo en 268.27.
3. **No la toma** → el simulador anotaba una operación que nadie tuvo. Pasó
   con F.

Los casos 2 y 3 se arreglaban editando `trading_state.json` a mano. Estos
tests protegen que dejen de necesitarlo, y sobre todo que las correcciones no
puedan meter basura en la única medición del proyecto que no sale de un
backtest.
"""

from __future__ import annotations

from datetime import date

import pytest

from trading import notify
from trading.config import DEFAULT_BACKTEST, replace
from trading.state import TradingState

from .test_state import FILL_BAR, SIGNAL_BAR, make_df, make_record

BT = replace(DEFAULT_BACKTEST, round_trip_cost=0.0)
DAY = date(2026, 8, 21)


def state_with(symbol: str = "AAPL") -> TradingState:
    state = TradingState()
    state.open_signals.append(make_record(symbol))
    return state


# ── Lectura del precio que se teclea en el móvil ─────────────────────────────

@pytest.mark.parametrize(
    "text, expected",
    [
        ("280.5", 280.5),
        ("280,5", 280.5),          # el usuario escribe desde Colombia
        ("$280.50", 280.5),        # el teclado del móvil cuela el símbolo
        (" 280.5 ", 280.5),
        ("1.234,56", 1234.56),     # convención española
        ("1,234.56", 1234.56),     # convención inglesa
        ("1,0850", 1.085),         # un par de forex, no mil ochocientos
        ("14", 14.0),
    ],
)
def test_a_price_is_read_however_it_is_typed(text, expected):
    """Rechazar '266,5' devolvería al usuario a editar el JSON a mano."""
    assert notify.parse_price(text) == pytest.approx(expected)


@pytest.mark.parametrize("text", ["", "  ", "abc", "-5", "0", "nan", "inf", "12.3.4"])
def test_a_price_that_is_not_a_price_is_rejected(text):
    """Un precio imposible no se adivina: se pregunta otra vez."""
    assert notify.parse_price(text) is None


# ── /tomada ───────────────────────────────────────────────────────────────────

def test_taken_records_the_real_entry_price():
    """El simulador entra al techo de la zona; Quantfury da otro céntimo."""
    state = state_with()
    update = state.mark_taken("AAPL", 98.4, day=DAY)

    assert update.ok and update.reason == "taken"
    record = state.open_signals[0]
    assert record["taken"] is True
    assert record["real_entry_price"] == 98.4
    assert record["real_entry_date"] == "2026-08-21"


def test_taken_accepts_the_symbol_in_lowercase():
    """El usuario escribe desde el móvil; exigir mayúsculas es un obstáculo."""
    state = state_with()
    assert state.mark_taken("aapl", 98.4, day=DAY).ok


def test_taken_on_a_symbol_that_is_not_open_changes_nothing():
    state = state_with("AAPL")
    update = state.mark_taken("MSFT", 98.4, day=DAY)

    assert not update.ok and update.reason == "not_open"
    assert "taken" not in state.open_signals[0]


def test_taken_below_the_stop_is_refused():
    """Una entrada bajo el stop habría nacido cerrada: es un dedazo.

    Guardarlo en silencio envenenaría la única medición real que hay, y el
    número resultante sería creíble, que es lo peligroso.
    """
    state = state_with()
    update = state.mark_taken("AAPL", 90.0, day=DAY)   # stop en 95

    assert not update.ok and update.reason == "below_stop"
    assert "real_entry_price" not in state.open_signals[0]


def test_taken_above_the_zone_is_recorded_but_warns():
    """Comprar por encima del techo es un hecho, no una propuesta.

    Se registra —pasó de verdad— pero el simulador nunca habría ejecutado esa
    orden, así que no podrá cerrarla solo y hay que avisarlo.
    """
    state = state_with()
    update = state.mark_taken("AAPL", 104.0, day=DAY)  # techo de la zona en 100

    assert update.ok and update.warning == "outside_zone"
    assert state.open_signals[0]["real_entry_price"] == 104.0
    assert "/cerrar" in notify.format_taken(update.record, update.warning)


# ── /paso ─────────────────────────────────────────────────────────────────────

def test_skipped_leaves_the_tracking_without_inventing_a_trade():
    """El caso de F: cribada, enviada, y el usuario no la tomó."""
    state = state_with("F")
    update = state.mark_skipped("f")

    assert update.ok
    assert state.open_signals == []
    assert state.history[0]["status"] == "not_taken"
    assert state.history[0]["symbol"] == "F"


def test_a_skipped_candidate_counts_in_neither_direction():
    """No se arriesgó dinero: sumarla al acierto o al fallo sería inventar."""
    state = state_with("F")
    state.history.append({"status": "closed", "is_win": True, "r_multiple": 2.0})
    state.mark_skipped("F")

    stats = state.performance()
    assert stats["closed"] == 1
    assert stats["win_rate"] == pytest.approx(1.0)
    assert stats["not_taken"] == 1


def test_a_skipped_candidate_can_be_screened_again():
    """No tomarla no bloquea el instrumento: no hay exposición que duplicar."""
    state = state_with("F")
    state.mark_skipped("F")
    assert state.open_symbols() == set()


def test_skipping_something_that_is_not_open_changes_nothing():
    state = state_with("AAPL")
    update = state.mark_skipped("MSFT")

    assert not update.ok and update.reason == "not_open"
    assert len(state.open_signals) == 1 and state.history == []


# ── /cerrar ───────────────────────────────────────────────────────────────────

def test_manual_close_reproduces_the_abbv_correction():
    """El caso que se arregló editando el JSON: salida a mano antes del objetivo.

    Los números salen de `trading_state.json`: entrada 245.65, salida 266.00,
    techo de la zona 248.06 y stop 240.53.
    """
    state = TradingState()
    state.open_signals.append(
        {
            **make_record("ABBV"),
            "entry_max": 248.06132508686156,
            "stop": 240.5262226377128,
            "target": 268.2746459554772,
        }
    )

    update = state.close_manually("ABBV", 266.0, entry_price=245.65, day=DAY)

    assert update.ok
    closed = state.history[0]
    assert closed["status"] == "closed" and closed["outcome"] == "manual"
    assert closed["r_multiple"] == pytest.approx(2.7007, abs=1e-3)
    assert closed["is_win"] is True
    assert state.open_signals == []


def test_manual_close_uses_the_entry_that_taken_registered():
    """Dos comandos, una sola verdad: /cerrar no vuelve a preguntar."""
    state = state_with()
    state.mark_taken("AAPL", 98.0, day=DAY)
    update = state.close_manually("AAPL", 108.0, day=DAY)

    assert update.ok
    assert state.history[0]["entry_price"] == 98.0
    assert state.history[0]["entry_date"] == "2026-08-21"


def test_manual_close_measures_r_against_the_planned_risk():
    """La R se mide contra el riesgo que el usuario conocía al dimensionar.

    Es la misma regla que el backtest: si entró más barato por un hueco, su
    pérdida en el stop es menor de 1R, y usar el riesgo realizado la contaría
    como 1R completa exagerando tanto pérdidas como ganancias.
    """
    state = state_with()                       # techo 100, stop 95 -> riesgo 5
    state.close_manually("AAPL", 110.0, entry_price=98.0, day=DAY, bt=BT)

    # Contra el riesgo planificado: (110 - 98) / 5 = 2.4
    # Contra el realizado (98 - 95 = 3) saldría 4.0, que sería inflarlo.
    assert state.history[0]["r_multiple"] == pytest.approx(2.4)


def test_manual_close_discounts_the_round_trip_cost():
    """Quantfury no cobra comisión pero opera sobre el spread."""
    state = state_with()
    state.close_manually("AAPL", 110.0, entry_price=100.0, day=DAY)

    esperado = 0.10 - DEFAULT_BACKTEST.round_trip_cost
    assert state.history[0]["return_pct"] == pytest.approx(esperado, abs=1e-4)


def test_a_manual_close_in_loss_is_not_counted_as_a_win():
    state = state_with()
    state.close_manually("AAPL", 96.0, entry_price=100.0, day=DAY)

    assert state.history[0]["is_win"] is False
    assert state.performance()["win_rate"] == pytest.approx(0.0)


def test_closing_without_knowing_the_entry_refuses_instead_of_guessing():
    """Inventar la entrada con el techo de la zona daría un número falso y creíble."""
    state = state_with()
    update = state.close_manually("AAPL", 110.0, day=DAY)

    assert not update.ok and update.reason == "unknown_entry"
    assert len(state.open_signals) == 1        # sigue abierta, no se pierde
    assert state.history == []


def test_a_bad_price_does_not_close_anything():
    state = state_with()
    update = state.close_manually("AAPL", float("nan"), entry_price=98.0, day=DAY)

    assert not update.ok and update.reason == "bad_price"
    assert len(state.open_signals) == 1


def test_a_manual_close_survives_the_json_round_trip(tmp_path):
    """NaN o infinito dejarían el estado ilegible en la siguiente corrida."""
    import json

    state = state_with()
    state.close_manually("AAPL", 110.0, entry_price=98.0, day=DAY)

    path = str(tmp_path / "estado.json")
    state.save(path)
    raw = open(path, encoding="utf-8").read()
    assert "NaN" not in raw and "Infinity" not in raw
    json.loads(raw)


# ── Cómo conviven las correcciones con el seguimiento automático ─────────────

def test_a_taken_trade_resolves_against_the_real_entry():
    """El simulador decide la salida; el precio de entrada lo pone el usuario.

    No se toca cómo se detecta la salida —eso sigue siendo `simulate_signal`,
    igual que en el backtest—; solo cambia el precio contra el que se miden el
    retorno y la R.
    """
    state = state_with()
    state.mark_taken("AAPL", 98.0, day=date(2024, 1, 2))
    bars = {"AAPL": make_df([SIGNAL_BAR, FILL_BAR, (100.0, 116.0, 99.0, 115.5)])}

    state.update_open_signals(bars, BT)

    closed = state.history[0]
    assert closed["outcome"] == "win"
    assert closed["entry_price"] == 98.0
    # (115 - 98) / 5 = 3.4, no (115 - 99) / 5 = 3.2 que daría la simulación.
    assert closed["r_multiple"] == pytest.approx(3.4)
    # Y se conserva el precio del simulador para poder auditar la diferencia.
    assert closed["simulated_entry_price"] == pytest.approx(99.0)


def test_a_trade_taken_outside_the_zone_is_not_expired_away():
    """El usuario dice que está dentro; el simulador dice que nunca se ejecutó.

    Manda el usuario: la operación existe y tiene dinero real. Caducarla la
    borraría del seguimiento con la posición todavía abierta.
    """
    state = state_with()
    state.mark_taken("AAPL", 104.0, day=date(2024, 1, 2))   # por encima del techo
    arriba = (108.0, 112.0, 107.0, 110.0)
    bars = {"AAPL": make_df([SIGNAL_BAR, arriba, arriba, arriba])}

    closed = state.update_open_signals(bars, replace(BT, entry_valid_days=2))

    assert closed == []
    assert len(state.open_signals) == 1        # espera a /cerrar
    assert state.history == []


def test_an_untaken_trade_still_expires_normally():
    """La corrección no puede convertir el caducar en algo que no ocurre."""
    state = state_with()
    arriba = (108.0, 112.0, 107.0, 110.0)
    bars = {"AAPL": make_df([SIGNAL_BAR, arriba, arriba, arriba])}

    state.update_open_signals(bars, replace(BT, entry_valid_days=2))

    assert state.open_signals == []
    assert state.history[0]["status"] == "expired"


def test_a_skipped_candidate_is_never_resolved_by_the_market():
    """Es el fallo que /paso viene a impedir: una operación que nadie tuvo."""
    state = state_with()
    state.mark_skipped("AAPL")
    bars = {"AAPL": make_df([SIGNAL_BAR, FILL_BAR, (100.0, 116.0, 99.0, 115.5)])}

    assert state.update_open_signals(bars, BT) == []
    assert len(state.history) == 1
    assert state.history[0]["status"] == "not_taken"
    assert "outcome" not in state.history[0]


# ── Lo que ve el usuario ──────────────────────────────────────────────────────

def test_the_history_never_shows_an_outcome_of_none():
    """Con F en el histórico, /senales imprimía '❌ F — None'."""
    history = [
        {"symbol": "F", "status": "not_taken"},
        {"symbol": "ABBV", "status": "closed", "outcome": "manual",
         "is_win": True, "return_pct": 0.0818},
        {"symbol": "TXN", "status": "expired"},
    ]
    texto = notify.format_recent(history)

    assert "None" not in texto
    assert "🚫" in texto and "No tomada" in texto
    assert "Cerrada a mano" in texto and "+8.2%" in texto
    assert "Caducada" in texto


def test_the_performance_says_how_many_you_let_pass():
    """Sin ese número, el rendimiento parece describir lo que hizo el usuario."""
    stats = {
        "closed": 2, "wins": 1, "win_rate": 0.5, "avg_r": 0.5,
        "total_r": 1.0, "open": 1, "not_taken": 3,
    }
    assert "dejaste pasar: 3" in notify.format_performance(stats)


def test_the_help_explains_the_three_commands():
    ayuda = notify.format_help()
    for comando in ("/tomada", "/cerrar", "/paso"):
        assert comando in ayuda


def test_an_error_names_what_is_actually_open():
    """Un 'no se pudo' sin motivo deja al usuario sin saber qué cree el bot."""
    texto = notify.format_manual_error("not_open", "msft", ["AAPL", "TXN"])
    assert "MSFT" in texto and "AAPL" in texto and "TXN" in texto


# ── El comando completo, tal y como llega de Telegram ────────────────────────

class FakeClient:
    """Cliente de Telegram que no habla con nadie: guarda lo que se enviaría."""

    def __init__(self) -> None:
        self.sent: list[str] = []

    def send(self, chat_id: str, text: str) -> bool:
        self.sent.append(text)
        return True

    def broadcast(self, text: str, chat_ids) -> bool:
        self.sent.append(text)
        return True


def run(text: str, state: TradingState) -> tuple[str, bool]:
    """Ejecuta un mensaje de Telegram entero y devuelve (respuesta, cambió)."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import trading_bot

    client = FakeClient()
    parts = text.split()
    changed = trading_bot._handle_command(
        parts[0].lower().split("@")[0], parts[1:], "1", state, client
    )
    return client.sent[-1], changed


def test_the_close_command_works_end_to_end():
    state = state_with()
    state.mark_taken("AAPL", 98.0, day=DAY)

    respuesta, changed = run("/cerrar AAPL 108,5", state)

    assert changed is True          # cmd_poll tiene que guardar el estado
    assert state.open_signals == []
    assert state.history[0]["exit_price"] == 108.5
    assert "AAPL" in respuesta


def test_the_taken_command_works_end_to_end():
    state = state_with()
    respuesta, changed = run("/tomada aapl 98.4", state)

    assert changed is True
    assert state.open_signals[0]["real_entry_price"] == 98.4
    assert "98.40" in respuesta


def test_the_skip_command_works_end_to_end():
    state = state_with()
    respuesta, changed = run("/paso AAPL", state)

    assert changed is True
    assert state.history[0]["status"] == "not_taken"
    assert "no tomada" in respuesta


def test_the_close_command_accepts_the_entry_in_the_same_message():
    """Quien no registró la entrada en su momento no debería tener que fingirla."""
    state = state_with()
    _, changed = run("/cerrar AAPL 110 98", state)

    assert changed is True
    assert state.history[0]["entry_price"] == 98.0
    assert state.history[0]["exit_price"] == 110.0


@pytest.mark.parametrize("texto", ["/cerrar", "/cerrar AAPL", "/tomada AAPL", "/paso"])
def test_an_incomplete_command_explains_itself_instead_of_failing(texto):
    """Callar ante un comando a medias es indistinguible de un bot averiado."""
    state = state_with()
    respuesta, changed = run(texto, state)

    assert changed is False
    assert "Escríbelo así" in respuesta
    assert len(state.open_signals) == 1


def test_a_command_with_a_bad_price_does_not_touch_the_state():
    state = state_with()
    respuesta, changed = run("/cerrar AAPL cientodiez", state)

    assert changed is False and len(state.open_signals) == 1
    assert "No entendí el precio" in respuesta


def test_a_command_on_an_unknown_symbol_lists_what_is_open():
    state = state_with("AAPL")
    respuesta, changed = run("/paso NVDA", state)

    assert changed is False
    assert "NVDA" in respuesta and "AAPL" in respuesta
    assert len(state.open_signals) == 1


def test_the_open_list_shows_the_real_entry_once_you_report_it():
    state = state_with()
    state.mark_taken("AAPL", 98.4, day=DAY)

    respuesta, _ = run("/abiertas", state)

    assert "entrada 98.40" in respuesta
    assert "/cerrar" in respuesta      # recuerda cómo contarle la salida

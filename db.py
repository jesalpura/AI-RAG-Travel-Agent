"""
db.py — SQLite persistence layer for the Travel AI Agent
─────────────────────────────────────────────────────────
Drop this file next to agent.py and add one line at the top of agent.py:

    from db import db   # noqa: F401  (auto-initialises the database)

Then wrap each tool function as shown in agent_patched.py.
"""

import sqlite3
import time
import traceback as tb
from contextlib import contextmanager
from pathlib import Path

# ── Database path (same directory as this file) ──────────────────────────────
DB_PATH = Path(__file__).parent / "travel_agent.db"


# ── Connection helper ─────────────────────────────────────────────────────────
@contextmanager
def _conn():
    """Yield a thread-safe connection; always commit or rollback."""
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")   # safe for concurrent Streamlit threads
    con.execute("PRAGMA foreign_keys=ON")
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


# ── Schema bootstrap (runs once on import) ────────────────────────────────────
def _init_schema():
    with _conn() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at DATETIME DEFAULT (datetime('now')),
            updated_at DATETIME DEFAULT (datetime('now')),
            label      TEXT
        );

        CREATE TABLE IF NOT EXISTS messages (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            role       TEXT    NOT NULL CHECK(role IN ('user','assistant')),
            content    TEXT    NOT NULL,
            created_at DATETIME DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);

        CREATE TABLE IF NOT EXISTS tool_calls (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  INTEGER REFERENCES sessions(id) ON DELETE SET NULL,
            tool_name   TEXT    NOT NULL,
            raw_input   TEXT    NOT NULL,
            city        TEXT,
            result      TEXT,
            cache_hit   INTEGER DEFAULT 0 CHECK(cache_hit IN (0,1)),
            duration_ms INTEGER,
            called_at   DATETIME DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_toolcalls_tool ON tool_calls(tool_name);
        CREATE INDEX IF NOT EXISTS idx_toolcalls_city ON tool_calls(city);

        CREATE TABLE IF NOT EXISTS weather_cache (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            city        TEXT    NOT NULL,
            temp_c      REAL,
            description TEXT,
            raw_json    TEXT,
            fetched_at  DATETIME DEFAULT (datetime('now'))
        );
        CREATE UNIQUE INDEX IF NOT EXISTS uidx_weather_city ON weather_cache(city);

        CREATE TABLE IF NOT EXISTS hotels_cache (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            city           TEXT    NOT NULL,
            name           TEXT    NOT NULL,
            overall_rating REAL,
            price_lowest   TEXT,
            fetched_at     DATETIME DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_hotels_city ON hotels_cache(city);

        CREATE TABLE IF NOT EXISTS restaurants_cache (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            city       TEXT    NOT NULL,
            name       TEXT    NOT NULL,
            rating     REAL,
            address    TEXT,
            fetched_at DATETIME DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_restaurants_city ON restaurants_cache(city);

        CREATE TABLE IF NOT EXISTS places_cache (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            city       TEXT    NOT NULL,
            name       TEXT    NOT NULL,
            rating     REAL,
            latitude   REAL,
            longitude  REAL,
            fetched_at DATETIME DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_places_city ON places_cache(city);

        CREATE TABLE IF NOT EXISTS flights_cache (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            origin_city      TEXT    NOT NULL,
            origin_code      TEXT    NOT NULL,
            destination_city TEXT    NOT NULL,
            destination_code TEXT    NOT NULL,
            travel_date      TEXT    NOT NULL,
            airline          TEXT,
            stops            INTEGER,
            price_inr        TEXT,
            duration_min     INTEGER,
            departure_time   TEXT,
            arrival_time     TEXT,
            fetched_at       DATETIME DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_flights_route
            ON flights_cache(origin_code, destination_code, travel_date);

        CREATE TABLE IF NOT EXISTS airport_codes (
            city_lower TEXT PRIMARY KEY,
            iata_code  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS rag_chunks (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT    NOT NULL,
            chunk_index INTEGER NOT NULL,
            content     TEXT    NOT NULL,
            embedding   BLOB,
            indexed_at  DATETIME DEFAULT (datetime('now')),
            UNIQUE(source_file, chunk_index)
        );

        CREATE TABLE IF NOT EXISTS error_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  INTEGER REFERENCES sessions(id) ON DELETE SET NULL,
            tool_name   TEXT,
            error_type  TEXT,
            message     TEXT,
            traceback   TEXT,
            occurred_at DATETIME DEFAULT (datetime('now'))
        );
        """)

        # Seed airport codes (safe — uses INSERT OR IGNORE)
        _AIRPORT_CODES = {
            "ahmedabad": "AMD", "mumbai": "BOM", "delhi": "DEL",
            "new delhi": "DEL", "jaipur": "JAI", "goa": "GOI",
            "bangalore": "BLR", "bengaluru": "BLR", "hyderabad": "HYD",
            "chennai": "MAA", "kolkata": "CCU", "manali": "KUU",
            "kullu": "KUU", "shimla": "SLV", "leh": "IXL",
            "srinagar": "SXR", "pune": "PNQ", "kochi": "COK",
            "cochin": "COK", "nagpur": "NAG", "varanasi": "VNS",
            "amritsar": "ATQ", "udaipur": "UDR", "jodhpur": "JDH",
        }
        con.executemany(
            "INSERT OR IGNORE INTO airport_codes(city_lower, iata_code) VALUES (?,?)",
            _AIRPORT_CODES.items(),
        )


_init_schema()


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═════════════════════════════════════════════════════════════════════════════

# ── Sessions ──────────────────────────────────────────────────────────────────

def new_session(label: str = "") -> int:
    """Create a new session row and return its id."""
    with _conn() as con:
        cur = con.execute(
            "INSERT INTO sessions(label) VALUES (?)", (label,)
        )
        return cur.lastrowid


def touch_session(session_id: int):
    """Update updated_at timestamp for an existing session."""
    with _conn() as con:
        con.execute(
            "UPDATE sessions SET updated_at=datetime('now') WHERE id=?",
            (session_id,),
        )


# ── Messages ──────────────────────────────────────────────────────────────────

def save_message(session_id: int, role: str, content: str):
    """Persist a single chat message and touch the session timestamp."""
    with _conn() as con:
        con.execute(
            "INSERT INTO messages(session_id, role, content) VALUES (?,?,?)",
            (session_id, role, content),
        )
        con.execute(
            "UPDATE sessions SET updated_at=datetime('now') WHERE id=?",
            (session_id,),
        )


def load_messages(session_id: int) -> list[dict]:
    """Return all messages for a session as a list of dicts."""
    with _conn() as con:
        rows = con.execute(
            "SELECT role, content, created_at FROM messages "
            "WHERE session_id=? ORDER BY id",
            (session_id,),
        ).fetchall()
    return [dict(r) for r in rows]

#__________________TOOL-USAGE_________________________
def get_tool_usage():
    with _conn() as con:
        rows = con.execute("""
            SELECT tool_name,
                   COUNT(*) as count
            FROM tool_calls
            GROUP BY tool_name
            ORDER BY count DESC
        """).fetchall()

    return [dict(r) for r in rows]

# ── Tool-call logging ─────────────────────────────────────────────────────────

def log_tool_call(
    tool_name: str,
    raw_input: str,
    result: str,
    city: str = "",
    cache_hit: bool = False,
    duration_ms: int = 0,
    session_id: int | None = None,
):
    """Insert one row into tool_calls."""
    with _conn() as con:
        con.execute(
            """INSERT INTO tool_calls
               (session_id, tool_name, raw_input, city, result, cache_hit, duration_ms)
               VALUES (?,?,?,?,?,?,?)""",
            (session_id, tool_name, raw_input, city or "",
             result, int(cache_hit), duration_ms),
        )


# ── Weather ───────────────────────────────────────────────────────────────────

def upsert_weather(city: str, temp_c: float, description: str, raw_json: str = ""):
    """Insert or replace weather data for a city."""
    with _conn() as con:
        con.execute(
            """INSERT INTO weather_cache(city, temp_c, description, raw_json, fetched_at)
               VALUES (?,?,?,?,datetime('now'))
               ON CONFLICT(city) DO UPDATE SET
                   temp_c=excluded.temp_c,
                   description=excluded.description,
                   raw_json=excluded.raw_json,
                   fetched_at=excluded.fetched_at""",
            (city.lower(), temp_c, description, raw_json),
        )

#________Dashboard__________________
def get_stats():
    with _conn() as con:
        return {
            "sessions": con.execute(
                "SELECT COUNT(*) FROM sessions"
            ).fetchone()[0],

            "messages": con.execute(
                "SELECT COUNT(*) FROM messages"
            ).fetchone()[0],

            "tool_calls": con.execute(
                "SELECT COUNT(*) FROM tool_calls"
            ).fetchone()[0],

            "errors": con.execute(
                "SELECT COUNT(*) FROM error_log"
            ).fetchone()[0],
        }
    
# ── Hotels ────────────────────────────────────────────────────────────────────

def save_hotels(city: str, properties: list[dict]):
    """
    Persist hotel results.  Clears old rows for the city first so results
    stay fresh (SerpApi is the source of truth).

    Each dict in `properties` should have keys: name, overall_rating, price_lowest.
    """
    with _conn() as con:
        con.execute("DELETE FROM hotels_cache WHERE city=?", (city.lower(),))
        con.executemany(
            """INSERT INTO hotels_cache(city, name, overall_rating, price_lowest)
               VALUES (?,?,?,?)""",
            [
                (
                    city.lower(),
                    h.get("name", ""),
                    h.get("overall_rating"),
                    h.get("price_lowest", ""),
                )
                for h in properties
            ],
        )


# ── Restaurants ───────────────────────────────────────────────────────────────

def save_restaurants(city: str, places: list[dict]):
    """
    Persist restaurant results.  Each dict should have: name, rating, address.
    """
    with _conn() as con:
        con.execute("DELETE FROM restaurants_cache WHERE city=?", (city.lower(),))
        con.executemany(
            """INSERT INTO restaurants_cache(city, name, rating, address)
               VALUES (?,?,?,?)""",
            [
                (
                    city.lower(),
                    p.get("title", p.get("name", "")),
                    p.get("rating"),
                    p.get("address", ""),
                )
                for p in places
            ],
        )


# ── Places / map locations ────────────────────────────────────────────────────

def save_places(city: str, places: list[dict]):
    """
    Persist tourist-place results.
    Each dict should have: name (or title), rating, lat, lon.
    """
    with _conn() as con:
        con.execute("DELETE FROM places_cache WHERE city=?", (city.lower(),))
        con.executemany(
            """INSERT INTO places_cache(city, name, rating, latitude, longitude)
               VALUES (?,?,?,?,?)""",
            [
                (
                    city.lower(),
                    p.get("name", p.get("title", "")),
                    p.get("rating"),
                    p.get("lat") or p.get("gps_coordinates", {}).get("latitude"),
                    p.get("lon") or p.get("gps_coordinates", {}).get("longitude"),
                )
                for p in places
            ],
        )


# ── Flights ───────────────────────────────────────────────────────────────────

def save_flights(
    origin_city: str, origin_code: str,
    destination_city: str, destination_code: str,
    travel_date: str,
    flights: list[dict],
):
    """
    Persist flight results for a route+date.
    Each dict in `flights` is one SerpApi best_flights entry.
    """
    with _conn() as con:
        con.execute(
            "DELETE FROM flights_cache WHERE origin_code=? AND destination_code=? AND travel_date=?",
            (origin_code, destination_code, travel_date),
        )
        rows = []
        for f in flights:
            legs = f.get("flights", [{}])
            rows.append((
                origin_city.lower(), origin_code,
                destination_city.lower(), destination_code,
                travel_date,
                legs[0].get("airline", ""),
                len(legs) - 1,
                str(f.get("price", "")),
                f.get("total_duration"),
                legs[0].get("departure_airport", {}).get("time", ""),
                legs[-1].get("arrival_airport", {}).get("time", ""),
            ))
        con.executemany(
            """INSERT INTO flights_cache
               (origin_city, origin_code, destination_city, destination_code,
                travel_date, airline, stops, price_inr, duration_min,
                departure_time, arrival_time)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )


# ── Error logging ─────────────────────────────────────────────────────────────

def log_error(
    message: str,
    tool_name: str = "",
    session_id: int | None = None,
    exc: Exception | None = None,
):
    """Write an exception or error message to error_log."""
    with _conn() as con:
        con.execute(
            """INSERT INTO error_log
               (session_id, tool_name, error_type, message, traceback)
               VALUES (?,?,?,?,?)""",
            (
                session_id,
                tool_name,
                type(exc).__name__ if exc else "",
                message,
                tb.format_exc() if exc else "",
            ),
        )

def get_tool_performance():
    with _conn() as con:
        rows = con.execute("""
            SELECT tool_name,
                   AVG(duration_ms) as avg_time
            FROM tool_calls
            GROUP BY tool_name
        """).fetchall()

    return [dict(r) for r in rows]

# ── Utility: timing wrapper ───────────────────────────────────────────────────

class _Timer:
    """Context manager that records elapsed milliseconds."""
    def __enter__(self):
        self._start = time.perf_counter()
        return self
    def __exit__(self, *_):
        self.ms = int((time.perf_counter() - self._start) * 1000)

timer = _Timer  # re-export so callers can do `with db.timer() as t:`

def get_top_cities():
    with _conn() as con:
        rows = con.execute("""
            SELECT city,
                   COUNT(*) as count
            FROM tool_calls
            WHERE city != ''
            GROUP BY city
            ORDER BY count DESC
            LIMIT 10
        """).fetchall()

    return [dict(r) for r in rows]


def get_daily_activity():
    with _conn() as con:
        rows = con.execute("""
            SELECT DATE(created_at) as day,
                   COUNT(*) as messages
            FROM messages
            GROUP BY DATE(created_at)
            ORDER BY day
        """).fetchall()

    return [dict(r) for r in rows]

def get_messages(limit=100):
    with _conn() as con:
        rows = con.execute("""
            SELECT *
            FROM messages
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()

    return [dict(r) for r in rows]


# ── Convenience: expose the db module as a single importable object ───────────
class _DB:
    new_session = staticmethod(new_session)
    touch_session = staticmethod(touch_session)
    save_message = staticmethod(save_message)
    load_messages = staticmethod(load_messages)
    log_tool_call = staticmethod(log_tool_call)
    upsert_weather = staticmethod(upsert_weather)
    save_hotels = staticmethod(save_hotels)
    save_restaurants = staticmethod(save_restaurants)
    save_places = staticmethod(save_places)
    save_flights = staticmethod(save_flights)
    log_error = staticmethod(log_error)
    timer = staticmethod(_Timer)
    get_stats = staticmethod(get_stats)
    get_tool_usage = staticmethod(get_tool_usage)
    get_top_cities = staticmethod(get_top_cities)
    get_daily_activity = staticmethod(get_daily_activity)
    get_tool_performance = staticmethod(get_tool_performance)
    get_messages = staticmethod(get_messages)
    get_stats = staticmethod(get_stats)



db = _DB()

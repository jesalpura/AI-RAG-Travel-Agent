import streamlit as st
from agent import get_travel_agent, create_memory, clear_tool_cache
from db import _conn  # direct connection for ad-hoc search queries

from db import DB_PATH

print("STREAMLIT DATABASE:", DB_PATH.resolve())

# _____________________________________________
# Page config
# _____________________________________________
st.set_page_config(
    page_title="Global Wanderer AI",
    page_icon="✈️",
    layout="wide",
)

# _____________________________________________
# Custom CSS — travel-themed dark aesthetic
# _____________________________________________
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.main-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    background: linear-gradient(135deg, #f6a623, #e8735a);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}
.subtitle {
    color: #888;
    font-size: 0.95rem;
    margin-top: 4px;
    margin-bottom: 1.5rem;
}
.map-label {
    font-size: 0.85rem;
    color: #aaa;
    margin-bottom: 6px;
    margin-top: 16px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Sidebar search result cards ── */
.result-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 10px;
    padding: 10px 12px;
    margin-bottom: 8px;
    font-size: 0.82rem;
    line-height: 1.55;
}
.result-card .card-title {
    font-weight: 600;
    font-size: 0.88rem;
    color: #f6a623;
    margin-bottom: 2px;
}
.result-card .card-meta {
    color: #999;
    font-size: 0.76rem;
}
.result-card .card-body {
    color: #ccc;
    margin-top: 4px;
    white-space: pre-wrap;
    word-break: break-word;
}
.section-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #777;
    margin: 10px 0 4px;
}
.no-results {
    color: #777;
    font-size: 0.83rem;
    text-align: center;
    padding: 14px 0;
}
</style>
""", unsafe_allow_html=True)

# _____________________________________________
# Header
# _____________________________________________
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown('<div class="main-title">✈️ Global Wanderer AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Plan your next adventure with real-time weather, '
        'flights, hotels, attractions & live maps.</div>',
        unsafe_allow_html=True,
    )

# _____________________________________________
# Session state
# _____________________________________________
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent_memory" not in st.session_state:
    memory, session_id = create_memory()
    st.session_state.agent_memory = memory
    st.session_state.session_id = session_id


# __________________________________________________
# DB SEARCH HELPERS
# __________________________________________________

def _like(term: str) -> str:
    """Return a SQL LIKE pattern for a search term."""
    return f"%{term.strip().lower()}%"


def search_chat_history(term: str) -> list[dict]:
    """Full-text search across all stored messages."""
    with _conn() as con:
        rows = con.execute(
            """
            SELECT m.role, m.content, m.created_at, s.label as session_label
            FROM messages m
            JOIN sessions s ON s.id = m.session_id
            WHERE LOWER(m.content) LIKE ?
            ORDER BY m.id DESC
            LIMIT 30
            """,
            (_like(term),),
        ).fetchall()
    return [dict(r) for r in rows]


def search_hotels(term: str) -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            """
            SELECT city, name, overall_rating, price_lowest, fetched_at
            FROM hotels_cache
            WHERE LOWER(city) LIKE ? OR LOWER(name) LIKE ?
            ORDER BY overall_rating DESC NULLS LAST
            LIMIT 20
            """,
            (_like(term), _like(term)),
        ).fetchall()
    return [dict(r) for r in rows]


def search_restaurants(term: str) -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            """
            SELECT city, name, rating, address, fetched_at
            FROM restaurants_cache
            WHERE LOWER(city) LIKE ? OR LOWER(name) LIKE ?
            ORDER BY rating DESC NULLS LAST
            LIMIT 20
            """,
            (_like(term), _like(term)),
        ).fetchall()
    return [dict(r) for r in rows]


def search_places(term: str) -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            """
            SELECT city, name, rating, latitude, longitude, fetched_at
            FROM places_cache
            WHERE LOWER(city) LIKE ? OR LOWER(name) LIKE ?
            ORDER BY rating DESC NULLS LAST
            LIMIT 20
            """,
            (_like(term), _like(term)),
        ).fetchall()
    return [dict(r) for r in rows]


def search_flights(term: str) -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            """
            SELECT origin_city, origin_code, destination_city, destination_code,
                   travel_date, airline, stops, price_inr, duration_min,
                   departure_time, arrival_time, fetched_at
            FROM flights_cache
            WHERE LOWER(origin_city) LIKE ?
               OR LOWER(destination_city) LIKE ?
               OR LOWER(airline) LIKE ?
            ORDER BY fetched_at DESC
            LIMIT 20
            """,
            (_like(term), _like(term), _like(term)),
        ).fetchall()
    return [dict(r) for r in rows]


def search_weather(term: str) -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            """
            SELECT city, temp_c, description, fetched_at
            FROM weather_cache
            WHERE LOWER(city) LIKE ?
            ORDER BY fetched_at DESC
            LIMIT 10
            """,
            (_like(term),),
        ).fetchall()
    return [dict(r) for r in rows]


def get_all_known_cities() -> list[str]:
    """Return a sorted unique list of all cities stored across every cache table."""
    with _conn() as con:
        rows = con.execute(
            """
            SELECT DISTINCT LOWER(city) as city FROM hotels_cache
            UNION
            SELECT DISTINCT LOWER(city) FROM restaurants_cache
            UNION
            SELECT DISTINCT LOWER(city) FROM places_cache
            UNION
            SELECT DISTINCT LOWER(city) FROM weather_cache
            UNION
            SELECT DISTINCT LOWER(origin_city) FROM flights_cache
            UNION
            SELECT DISTINCT LOWER(destination_city) FROM flights_cache
            ORDER BY city
            """
        ).fetchall()
    return [r[0].title() for r in rows if r[0]]


def get_all_tools_used() -> list[str]:
    """Return a list of distinct tool names logged in tool_calls."""
    with _conn() as con:
        rows = con.execute(
            "SELECT DISTINCT tool_name FROM tool_calls ORDER BY tool_name"
        ).fetchall()
    return [r[0] for r in rows]


# _____________________________________________
# MAP HELPERS
# _____________________________________________
def build_map_embed(locations: list) -> str:
    if not locations:
        return ""
    lats = [loc["lat"] for loc in locations if "lat" in loc]
    lons = [loc["lon"] for loc in locations if "lon" in loc]
    if not lats:
        return ""
    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)
    markers_js = ""
    for loc in locations:
        name = loc.get("name", "Location").replace("'", "\\'")
        lat = loc.get("lat", 0)
        lon = loc.get("lon", 0)
        markers_js += (
            f'L.marker([{lat}, {lon}])'
            f'.addTo(map)'
            f'.bindPopup("<b>{name}</b>");\n'
        )
    map_html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  body {{ margin: 0; padding: 0; }}
  #map {{ width: 100%; height: 380px; border-radius: 12px; }}
</style>
</head>
<body>
<div id="map"></div>
<script>
  var map = L.map('map').setView([{center_lat}, {center_lon}], 13);
  L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
    attribution: '© OpenStreetMap contributors'
  }}).addTo(map);
  {markers_js}
</script>
</body>
</html>
"""
    return map_html


def render_map(locations: list):
    if not locations:
        return
    map_html = build_map_embed(locations)
    if map_html:
        st.markdown('<div class="map-label">📍 Locations on Map</div>', unsafe_allow_html=True)
        st.components.v1.html(map_html, height=400, scrolling=False)


# _____________________________________________
# Chat history display
# _____________________________________________
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("locations"):
            render_map(message["locations"])

# _____________________________________________
# Chat input
# _____________________________________________
if prompt := st.chat_input("Where do you want to go? (e.g. plan a trip to Goa, flights from Ahmedabad to Delhi, show map of Jaipur)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consulting travel guides and real-time data..."):
            try:
                result = get_travel_agent(
                    prompt,
                    memory=st.session_state.agent_memory,
                    session_id=st.session_state.session_id
                )
                if isinstance(result, dict):
                    response_text = result.get("text", "")
                    locations = result.get("locations", [])
                else:
                    response_text = str(result)
                    locations = []

                st.markdown(response_text)
                if locations:
                    render_map(locations)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "locations": locations,
                })

            except Exception as e:
                err = str(e)
                st.error(f"❌ Something went wrong: {err}")
                if "GOOGLE_API_KEY" in err or "API_KEY" in err or "api key" in err.lower():
                    st.warning("🔑 **API key issue** — check that `GOOGLE_API_KEY` is set in your Streamlit secrets or local `.env` file.")
                elif "SERP" in err or "serpapi" in err.lower():
                    st.warning("🔑 **SerpApi key issue** — check that `SERP_API_KEY` is set in your Streamlit secrets or local `.env` file.")
                elif "WEATHER" in err or "openweathermap" in err.lower():
                    st.warning("🌤 **Weather API issue** — check that `WEATHER_API_KEY` is set in your Streamlit secrets or local `.env` file.")
                elif "faiss" in err.lower() or "faiss_index" in err.lower():
                    st.warning(
                        "🗄 **FAISS index missing** — run your index-building script first to create `faiss_index/`. "
                        "The agent can still answer using its other tools."
                    )
                elif "travel_data" in err.lower():
                    st.warning("📄 **`travel_data.txt` not found** — create this file in your project root.")
                else:
                    st.info("💡 Check your terminal / console for the full traceback.")


# _____________________________________________________________________________
# SIDEBAR
# _____________________________________________________________________________
with st.sidebar:
    st.header("✈️ Global Wanderer AI")
    st.markdown("---")

    # _____ Tab switcher: Info | Search | Filter ____________________________

    tab_info, tab_search, tab_filter = st.tabs(["ℹ️ Info", "🔍 Search", "🎛 Filter"])

    # _________________________________________
    # TAB 1 — Info (original sidebar content)
    # _________________________________________
    with tab_info:
        st.write("**Try asking:**")
        st.markdown("""
- *Plan a 3-day trip to Goa*
- *Flights from Ahmedabad to Delhi*
- *Show me attractions in Jaipur on map*
- *Best hotels in Mumbai*
- *Weather in Manali*
        """)
        st.markdown("---")
        if st.button("🗑 Clear Chat History"):
            st.session_state.messages = []
            memory, session_id = create_memory()
            st.session_state.agent_memory = memory
            st.session_state.session_id = session_id
            clear_tool_cache()
            st.rerun()

    # _________________________________________
    # TAB 2 — Search stored data
    # _________________________________________

    with tab_search:
        st.caption("Search everything stored in the database — chats, hotels, flights, places and more.")

        search_query = st.text_input(
            "Search stored data",
            placeholder="e.g. Goa, IndiGo, Taj hotel…",
            key="sidebar_search_input",
            label_visibility="collapsed",
        )

        # Data-type selector
        search_scope = st.multiselect(
            "Search in",
            options=["💬 Chats", "✈️ Flights", "🏨 Hotels", "🍽 Restaurants", "📍 Places", "🌤 Weather"],
            default=["💬 Chats", "✈️ Flights", "🏨 Hotels", "🍽 Restaurants", "📍 Places", "🌤 Weather"],
            key="sidebar_search_scope",
        )

        if search_query.strip():
            total_found = 0

            # ______Chat history ________________________________________

            if "💬 Chats" in search_scope:
                chats = search_chat_history(search_query)
                if chats:
                    st.markdown(f'<div class="section-label">💬 Chat messages ({len(chats)})</div>', unsafe_allow_html=True)
                    for row in chats:
                        role_icon = "🧑" if row["role"] == "user" else "🤖"
                        preview = row["content"][:220].replace("\n", " ")
                        if len(row["content"]) > 220:
                            preview += "…"
                        st.markdown(
                            f'<div class="result-card">'
                            f'<div class="card-title">{role_icon} {row["role"].title()}</div>'
                            f'<div class="card-meta">{row["created_at"]}</div>'
                            f'<div class="card-body">{preview}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    total_found += len(chats)

            # _____Flights __________________________________________

            if "✈️ Flights" in search_scope:
                flights = search_flights(search_query)
                if flights:
                    st.markdown(f'<div class="section-label">✈️ Flights ({len(flights)})</div>', unsafe_allow_html=True)
                    for f in flights:
                        stops_label = "Non-stop" if f["stops"] == 0 else f'{f["stops"]} stop(s)'
                        duration = f"{f['duration_min'] // 60}h {f['duration_min'] % 60}m" if f["duration_min"] else "—"
                        st.markdown(
                            f'<div class="result-card">'
                            f'<div class="card-title">✈️ {f["origin_city"].title()} ({f["origin_code"]}) → {f["destination_city"].title()} ({f["destination_code"]})</div>'
                            f'<div class="card-meta">{f["travel_date"]} · {f["airline"] or "—"} · {stops_label}</div>'
                            f'<div class="card-body">🕐 {f["departure_time"] or "—"} → {f["arrival_time"] or "—"} &nbsp;|&nbsp; ⏱ {duration} &nbsp;|&nbsp; 💰 {f["price_inr"] or "—"}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    total_found += len(flights)

            # _____Hotels ________________________________________________

            if "🏨 Hotels" in search_scope:
                hotels = search_hotels(search_query)
                if hotels:
                    st.markdown(f'<div class="section-label">🏨 Hotels ({len(hotels)})</div>', unsafe_allow_html=True)
                    for h in hotels:
                        rating_str = f"⭐ {h['overall_rating']}" if h["overall_rating"] else "No rating"
                        price_str = h["price_lowest"] or "Price N/A"
                        st.markdown(
                            f'<div class="result-card">'
                            f'<div class="card-title">🏨 {h["name"]}</div>'
                            f'<div class="card-meta">{h["city"].title()} · {rating_str} · {price_str}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    total_found += len(hotels)

            # ______Restaurants _____________________________________________

            if "🍽 Restaurants" in search_scope:
                restaurants = search_restaurants(search_query)
                if restaurants:
                    st.markdown(f'<div class="section-label">🍽 Restaurants ({len(restaurants)})</div>', unsafe_allow_html=True)
                    for r in restaurants:
                        rating_str = f"⭐ {r['rating']}" if r["rating"] else "No rating"
                        st.markdown(
                            f'<div class="result-card">'
                            f'<div class="card-title">🍽 {r["name"]}</div>'
                            f'<div class="card-meta">{r["city"].title()} · {rating_str}</div>'
                            f'<div class="card-body">{r["address"] or ""}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    total_found += len(restaurants)

            # _________Places ____________________________________________

            if "📍 Places" in search_scope:
                places = search_places(search_query)
                if places:
                    st.markdown(f'<div class="section-label">📍 Attractions ({len(places)})</div>', unsafe_allow_html=True)
                    for p in places:
                        rating_str = f"⭐ {p['rating']}" if p["rating"] else "No rating"
                        coords = ""
                        if p["latitude"] and p["longitude"]:
                            coords = f' · 🗺 {round(p["latitude"],4)}, {round(p["longitude"],4)}'
                        st.markdown(
                            f'<div class="result-card">'
                            f'<div class="card-title">📍 {p["name"]}</div>'
                            f'<div class="card-meta">{p["city"].title()} · {rating_str}{coords}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    total_found += len(places)

            # _________Weather _____________________________________________
            if "🌤 Weather" in search_scope:
                weathers = search_weather(search_query)
                if weathers:
                    st.markdown(f'<div class="section-label">🌤 Weather ({len(weathers)})</div>', unsafe_allow_html=True)
                    for w in weathers:
                        st.markdown(
                            f'<div class="result-card">'
                            f'<div class="card-title">🌤 {w["city"].title()}</div>'
                            f'<div class="card-body">🌡 {w["temp_c"]}°C · {w["description"]}<br>'
                            f'<span class="card-meta">Fetched: {w["fetched_at"]}</span></div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    total_found += len(weathers)

            if total_found == 0:
                st.markdown(
                    f'<div class="no-results">No stored results found for <b>"{search_query}"</b>.<br>'
                    f'Try asking the agent about it first!</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("Start typing to search your trip history and cached data.")

    # ─────────────────────────────────────────
    # TAB 3 — Filter by city / data type
    # ─────────────────────────────────────────
    with tab_filter:
        st.caption("Browse stored data by city or category.")

        # ── City picker ───────────────────────────────────────────────────
        known_cities = get_all_known_cities()
        if not known_cities:
            st.info("No cached data yet. Start chatting to build your history!")
        else:
            selected_city = st.selectbox(
                "📍 Filter by city",
                options=["— All cities —"] + known_cities,
                key="filter_city",
            )

            city_term = "" if selected_city == "— All cities —" else selected_city

            # ── Category checkboxes ───────────────────────────────────────
            st.markdown("**Show categories:**")
            col_a, col_b = st.columns(2)
            with col_a:
                show_flights     = st.checkbox("✈️ Flights",     value=True,  key="f_flights")
                show_hotels      = st.checkbox("🏨 Hotels",      value=True,  key="f_hotels")
                show_restaurants = st.checkbox("🍽 Restaurants", value=False, key="f_restaurants")
            with col_b:
                show_places  = st.checkbox("📍 Places",  value=True,  key="f_places")
                show_weather = st.checkbox("🌤 Weather", value=True,  key="f_weather")
                show_chats   = st.checkbox("💬 Chats",  value=False, key="f_chats")

            # Min rating slider (applies to hotels, restaurants, places)
            min_rating = st.slider("⭐ Minimum rating", 0.0, 5.0, 0.0, 0.5, key="filter_min_rating")

            st.markdown("---")

            # ── Filtered results ──────────────────────────────────────────
            filter_term = city_term if city_term else "%"   # blank city = all

            # Flights
            if show_flights:
                with _conn() as con:
                    flight_rows = con.execute(
                        """
                        SELECT origin_city, origin_code, destination_city, destination_code,
                               travel_date, airline, stops, price_inr, duration_min,
                               departure_time, arrival_time
                        FROM flights_cache
                        WHERE (? = '%' OR LOWER(origin_city) LIKE ? OR LOWER(destination_city) LIKE ?)
                        ORDER BY fetched_at DESC LIMIT 15
                        """,
                        (filter_term,
                         f"%{city_term.lower()}%" if city_term else "%",
                         f"%{city_term.lower()}%" if city_term else "%"),
                    ).fetchall()
                flights_data = [dict(r) for r in flight_rows]
                if flights_data:
                    st.markdown(f'<div class="section-label">✈️ Flights ({len(flights_data)})</div>', unsafe_allow_html=True)
                    for f in flights_data:
                        stops_label = "Non-stop" if f["stops"] == 0 else f'{f["stops"]} stop(s)'
                        duration = f"{f['duration_min'] // 60}h {f['duration_min'] % 60}m" if f["duration_min"] else "—"
                        st.markdown(
                            f'<div class="result-card">'
                            f'<div class="card-title">✈️ {f["origin_city"].title()} ({f["origin_code"]}) → {f["destination_city"].title()} ({f["destination_code"]})</div>'
                            f'<div class="card-meta">{f["travel_date"]} · {f["airline"] or "—"} · {stops_label}</div>'
                            f'<div class="card-body">🕐 {f["departure_time"] or "—"} → {f["arrival_time"] or "—"} &nbsp;|&nbsp; ⏱ {duration} &nbsp;|&nbsp; 💰 {f["price_inr"] or "—"}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

            # Hotels
            if show_hotels:
                with _conn() as con:
                    hotel_rows = con.execute(
                        """
                        SELECT city, name, overall_rating, price_lowest
                        FROM hotels_cache
                        WHERE (? = '%' OR LOWER(city) LIKE ?)
                          AND (overall_rating IS NULL OR overall_rating >= ?)
                        ORDER BY overall_rating DESC NULLS LAST LIMIT 15
                        """,
                        (filter_term,
                         f"%{city_term.lower()}%" if city_term else "%",
                         min_rating),
                    ).fetchall()
                hotels_data = [dict(r) for r in hotel_rows]
                if hotels_data:
                    st.markdown(f'<div class="section-label">🏨 Hotels ({len(hotels_data)})</div>', unsafe_allow_html=True)
                    for h in hotels_data:
                        rating_str = f"⭐ {h['overall_rating']}" if h["overall_rating"] else "No rating"
                        st.markdown(
                            f'<div class="result-card">'
                            f'<div class="card-title">🏨 {h["name"]}</div>'
                            f'<div class="card-meta">{h["city"].title()} · {rating_str} · {h["price_lowest"] or "—"}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

            # Restaurants
            if show_restaurants:
                with _conn() as con:
                    rest_rows = con.execute(
                        """
                        SELECT city, name, rating, address
                        FROM restaurants_cache
                        WHERE (? = '%' OR LOWER(city) LIKE ?)
                          AND (rating IS NULL OR rating >= ?)
                        ORDER BY rating DESC NULLS LAST LIMIT 15
                        """,
                        (filter_term,
                         f"%{city_term.lower()}%" if city_term else "%",
                         min_rating),
                    ).fetchall()
                rest_data = [dict(r) for r in rest_rows]
                if rest_data:
                    st.markdown(f'<div class="section-label">🍽 Restaurants ({len(rest_data)})</div>', unsafe_allow_html=True)
                    for r in rest_data:
                        rating_str = f"⭐ {r['rating']}" if r["rating"] else "No rating"
                        st.markdown(
                            f'<div class="result-card">'
                            f'<div class="card-title">🍽 {r["name"]}</div>'
                            f'<div class="card-meta">{r["city"].title()} · {rating_str}</div>'
                            f'<div class="card-body">{r["address"] or ""}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

            # Places
            if show_places:
                with _conn() as con:
                    place_rows = con.execute(
                        """
                        SELECT city, name, rating, latitude, longitude
                        FROM places_cache
                        WHERE (? = '%' OR LOWER(city) LIKE ?)
                          AND (rating IS NULL OR rating >= ?)
                        ORDER BY rating DESC NULLS LAST LIMIT 15
                        """,
                        (filter_term,
                         f"%{city_term.lower()}%" if city_term else "%",
                         min_rating),
                    ).fetchall()
                places_data = [dict(r) for r in place_rows]
                if places_data:
                    st.markdown(f'<div class="section-label">📍 Attractions ({len(places_data)})</div>', unsafe_allow_html=True)
                    for p in places_data:
                        rating_str = f"⭐ {p['rating']}" if p["rating"] else "No rating"
                        st.markdown(
                            f'<div class="result-card">'
                            f'<div class="card-title">📍 {p["name"]}</div>'
                            f'<div class="card-meta">{p["city"].title()} · {rating_str}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

            # Weather
            if show_weather:
                with _conn() as con:
                    weather_rows = con.execute(
                        """
                        SELECT city, temp_c, description, fetched_at
                        FROM weather_cache
                        WHERE (? = '%' OR LOWER(city) LIKE ?)
                        ORDER BY fetched_at DESC LIMIT 10
                        """,
                        (filter_term,
                         f"%{city_term.lower()}%" if city_term else "%"),
                    ).fetchall()
                weather_data = [dict(r) for r in weather_rows]
                if weather_data:
                    st.markdown(f'<div class="section-label">🌤 Weather ({len(weather_data)})</div>', unsafe_allow_html=True)
                    for w in weather_data:
                        st.markdown(
                            f'<div class="result-card">'
                            f'<div class="card-title">🌤 {w["city"].title()}</div>'
                            f'<div class="card-body">🌡 {w["temp_c"]}°C · {w["description"]}<br>'
                            f'<span class="card-meta">Fetched: {w["fetched_at"]}</span></div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

            # Chat messages
            if show_chats:
                with _conn() as con:
                    chat_rows = con.execute(
                        """
                        SELECT m.role, m.content, m.created_at
                        FROM messages m
                        WHERE (? = '%' OR LOWER(m.content) LIKE ?)
                        ORDER BY m.id DESC LIMIT 20
                        """,
                        (filter_term,
                         f"%{city_term.lower()}%" if city_term else "%"),
                    ).fetchall()
                chat_data = [dict(r) for r in chat_rows]
                if chat_data:
                    st.markdown(f'<div class="section-label">💬 Chat messages ({len(chat_data)})</div>', unsafe_allow_html=True)
                    for c in chat_data:
                        role_icon = "🧑" if c["role"] == "user" else "🤖"
                        preview = c["content"][:200].replace("\n", " ")
                        if len(c["content"]) > 200:
                            preview += "…"
                        st.markdown(
                            f'<div class="result-card">'
                            f'<div class="card-title">{role_icon} {c["role"].title()}</div>'
                            f'<div class="card-meta">{c["created_at"]}</div>'
                            f'<div class="card-body">{preview}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

            # Empty state
            all_off = not any([show_flights, show_hotels, show_restaurants, show_places, show_weather, show_chats])
            if all_off:
                st.caption("Select at least one category above to see results.")        

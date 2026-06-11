import os
import re
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.tools import Tool
from langchain_classic.agents import initialize_agent
from langchain_classic.agents import AgentType
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from datetime import datetime, timedelta
import traceback
import requests

import time
import functools
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

# ─────────────────────────────────────────────
# DATABASE — persist every tool result & message
# ─────────────────────────────────────────────
from db import db  # noqa: E402  (import after load_dotenv intentional)

# ─────────────────────────────────────────────
# SESSION CACHE — avoids duplicate API calls
# Results are cached per city for the lifetime
# of the Python process (i.e. one Streamlit run).
# Cache up to 64 unique cities per tool.
# ─────────────────────────────────────────────
def _cache(fn):
    """LRU cache wrapper — caches tool results by their string input."""
    cached = functools.lru_cache(maxsize=64)(fn)
    # Expose a way to clear cache from the UI (clear chat button)
    cached.clear = cached.cache_clear
    return cached

serp_api_key = os.getenv("SERP_API_KEY")


# ─────────────────────────────────────────────
# WEATHER TOOL
# ─────────────────────────────────────────────
@_cache
def get_weather(raw_input):
    city = _extract_city(raw_input) if "in " in raw_input.lower() else raw_input.strip()
    api_key = os.getenv("WEATHER_API_KEY")

    with db.timer() as t:
        geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={api_key}"
        geo_data = requests.get(geo_url).json()

        if not geo_data:
            result = f"Could not find location for {city}"
            db.log_tool_call("Weather Tool", raw_input, result, city=city, duration_ms=t.ms)
            return result

        lat = geo_data[0]["lat"]
        lon = geo_data[0]["lon"]

        weather_url = (
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        )
        weather_response = requests.get(weather_url)
        weather_data = weather_response.json()

    if weather_response.status_code != 200:
        result = "Could not fetch weather data"
        db.log_tool_call("Weather Tool", raw_input, result, city=city, duration_ms=t.ms)
        return result

    temp = weather_data["main"]["temp"]
    desc = weather_data["weather"][0]["description"]
    result = f"Weather in {city}: {temp}°C, {desc}"

    # ── Persist ──
    try:
        db.upsert_weather(city, temp, desc, raw_json=json.dumps(weather_data))
        db.log_tool_call("Weather Tool", raw_input, result, city=city, duration_ms=t.ms)
    except Exception as e:
        db.log_error(str(e), tool_name="Weather Tool", exc=e)

    return result


weather_tool = Tool(
    name="Weather Tool",
    func=get_weather,
    description="Use this when user asks about weather of any city",
)

# ─────────────────────────────────────────────
# HOTEL TOOL
# ─────────────────────────────────────────────
from serpapi import GoogleSearch


@_cache
def get_hotels(raw_input):
    city = _extract_city(raw_input)
    print("RAW INPUT:", raw_input)
    print("EXTRACTED CITY:", city)
    today = datetime.today().date()

    params = {
        "engine": "google_hotels",
        "q": f"{city} hotels",
        "check_in_date": today.strftime("%Y-%m-%d"),
        "check_out_date": (today + timedelta(days=1)).strftime("%Y-%m-%d"),
        "api_key": serp_api_key,
    }

    with db.timer() as t:
        results = GoogleSearch(params).get_dict()

    properties = results.get("properties", [])

    if not properties:
        result = "No hotels found"
        db.log_tool_call("Hotel Finder", raw_input, result, city=city, duration_ms=t.ms)
        return result

    output = f"Top hotels in {city}:\n"
    hotel_rows = []
    for hotel in properties[:3]:
        name = hotel.get("name", "Unknown")
        rating = hotel.get("overall_rating", "N/A")
        price = "N/A"
        if "rate_per_night" in hotel:
            price = hotel["rate_per_night"].get("lowest", "N/A")
        output += f"- {name} | Rating: {rating}⭐ | Price: {price}\n"
        hotel_rows.append({"name": name, "overall_rating": rating, "price_lowest": str(price)})

    # ── Persist ──
    try:
        db.save_hotels(city, hotel_rows)
        db.log_tool_call("Hotel Finder", raw_input, output, city=city, duration_ms=t.ms)
    except Exception as e:
        db.log_error(str(e), tool_name="Hotel Finder", exc=e)

    return output


hotel_tool = Tool(
    name="Hotel Finder",
    func=get_hotels,
    description="""Use this when user asks for hotels, accommodation, resorts, or places to stay.

IMPORTANT: Pass ONLY the bare city name.
Correct:  Mumbai
Wrong:    hotels in Mumbai""",
)

# ─────────────────────────────────────────────
# TOURIST PLACES TOOL
# Returns a JSON string with text + locations so the UI can render a map.
# ─────────────────────────────────────────────
@_cache
def get_places(raw_input):
    city = _extract_city(raw_input)
    params = {
        "engine": "google_maps",
        "q": f"top tourist attractions in {city} famous places",
        "api_key": serp_api_key,
    }

    with db.timer() as t:
        results = GoogleSearch(params).get_dict()

    places = results.get("local_results", [])

    if not places:
        result = json.dumps({"text": "No places found", "locations": []})
        db.log_tool_call("Tourist Places Finder", raw_input, result, city=city, duration_ms=t.ms)
        return result

    output_lines = [f"Top attractions in {city}:"]
    locations = []
    place_rows = []

    for place in places[:5]:
        name = place.get("title", "Unknown")
        rating = place.get("rating", "N/A")
        lat = place.get("gps_coordinates", {}).get("latitude")
        lon = place.get("gps_coordinates", {}).get("longitude")

        if lat and lon:
            locations.append({"name": name, "lat": lat, "lon": lon})
            place_rows.append({"name": name, "rating": rating, "lat": lat, "lon": lon})

        output_lines.append(f"- {name} | ⭐{rating} | ({lat},{lon})")

    result = json.dumps({"text": "\n".join(output_lines), "locations": locations})

    # ── Persist ──
    try:
        db.save_places(city, place_rows)
        db.log_tool_call("Tourist Places Finder", raw_input, result, city=city, duration_ms=t.ms)
    except Exception as e:
        db.log_error(str(e), tool_name="Tourist Places Finder", exc=e)

    return result


places_tool = Tool(
    name="Tourist Places Finder",
    func=get_places,
    description="""Use this when user asks for:
- places to visit
- tourist attractions
- sightseeing
- things to do
- show me on map
- map of attractions

Returns places with GPS coordinates for map display.""",
)

# ─────────────────────────────────────────────
# MAP TOOL
# Lets the agent explicitly trigger a map for any city.
# ─────────────────────────────────────────────
@_cache
def get_map_locations(raw_input):
    city = _extract_city(raw_input)
    """Fetch top attractions for a city and return structured location data for map display."""
    params = {
        "engine": "google_maps",
        "q": f"top tourist attractions in {city}",
        "api_key": serp_api_key,
    }

    with db.timer() as t:
        results = GoogleSearch(params).get_dict()

    places = results.get("local_results", [])

    if not places:
        result = json.dumps({"text": f"No map data found for {city}", "locations": []})
        db.log_tool_call("Map Locations Finder", raw_input, result, city=city, duration_ms=t.ms)
        return result

    locations = []
    place_rows = []
    lines = [f"📍 Map locations for {city}:"]

    for place in places[:8]:
        name = place.get("title", "Unknown")
        lat = place.get("gps_coordinates", {}).get("latitude")
        lon = place.get("gps_coordinates", {}).get("longitude")
        rating = place.get("rating", "N/A")

        if lat and lon:
            locations.append({"name": name, "lat": lat, "lon": lon})
            place_rows.append({"name": name, "rating": rating, "lat": lat, "lon": lon})
            lines.append(f"- {name} | ⭐{rating}")

    result = json.dumps({"text": "\n".join(lines), "locations": locations})

    # ── Persist ──
    try:
        db.save_places(city, place_rows)
        db.log_tool_call("Map Locations Finder", raw_input, result, city=city, duration_ms=t.ms)
    except Exception as e:
        db.log_error(str(e), tool_name="Map Locations Finder", exc=e)

    return result


map_tool = Tool(
    name="Map Locations Finder",
    func=get_map_locations,
    description="""Use this when user asks to:
- show on map
- show locations on map
- display map
- where is [place] on map
- map of [city]

Returns GPS coordinates so a map can be displayed in the UI.""",
)

# ─────────────────────────────────────────────
# HELPER: strip filler phrases → clean city name
# ─────────────────────────────────────────────
_FILLER_RE = re.compile(
    r"^(?:best\s+|top\s+|find\s+|show\s+|get\s+)?"
    r"(?:restaurants?|food\s+places?|cafes?|where\s+to\s+eat"
    r"|hotels?|accommodation|places?\s+to\s+stay"
    r"|tourist\s+attractions?|things\s+to\s+do|sightseeing"
    r"|attractions?|places?\s+to\s+visit)"
    r"(?:\s+in)?\s*",
    re.IGNORECASE,
)

def _extract_city(raw: str) -> str:
    """Strip LLM filler so we always get a bare city name."""
    return _FILLER_RE.sub("", raw.strip()).strip()




# ─────────────────────────────────────────────
# RESTAURANT TOOL
# ─────────────────────────────────────────────
@_cache
def get_restaurants(raw_input):
    city = _extract_city(raw_input)
    params = {
        "engine": "google_maps",
        "q": f"best restaurants in {city}",
        "api_key": serp_api_key,
    }

    with db.timer() as t:
        results = GoogleSearch(params).get_dict()

    places = results.get("local_results", [])

    if not places:
        result = f"No restaurants found in {city}"
        db.log_tool_call("Restaurant Finder", raw_input, result, city=city, duration_ms=t.ms)
        return result

    output = f"Top restaurants in {city}:\n"
    for place in places[:3]:
        name = place.get("title", "Unknown")
        rating = place.get("rating", "N/A")
        address = place.get("address", "N/A")
        output += f"- {name} | Rating: {rating}⭐ | {address}\n"

    # ── Persist ──
    try:
        db.save_restaurants(city, places[:3])
        db.log_tool_call("Restaurant Finder", raw_input, output, city=city, duration_ms=t.ms)
    except Exception as e:
        db.log_error(str(e), tool_name="Restaurant Finder", exc=e)

    return output


restaurant_tool = Tool(
    name="Restaurant Finder",
    func=get_restaurants,
    description="""Use this when user asks for restaurants, food places, cafes, or where to eat.

IMPORTANT: Pass ONLY the bare city name. Nothing else.
Correct:  Mumbai
Correct:  Goa
Wrong:    restaurants in Mumbai
Wrong:    best food in Goa""",
)

# ─────────────────────────────────────────────
# FLIGHT TOOL
# ─────────────────────────────────────────────
AIRPORT_CODES = {
    "ahmedabad": "AMD",
    "mumbai": "BOM",
    "delhi": "DEL",
    "new delhi": "DEL",
    "jaipur": "JAI",
    "goa": "GOI",
    "bangalore": "BLR",
    "bengaluru": "BLR",
    "hyderabad": "HYD",
    "chennai": "MAA",
    "kolkata": "CCU",
    "manali": "KUU",
    "kullu": "KUU",
    "shimla": "SLV",
    "leh": "IXL",
    "srinagar": "SXR",
    "pune": "PNQ",
    "kochi": "COK",
    "cochin": "COK",
    "nagpur": "NAG",
    "varanasi": "VNS",
    "amritsar": "ATQ",
    "udaipur": "UDR",
    "jodhpur": "JDH",
}


def _next_weekday_date() -> str:
    """Return tomorrow's date as YYYY-MM-DD (always a valid future date)."""
    from datetime import datetime, timedelta
    return (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")


def _parse_flight_cities(query: str):
    """
    Extract (origin, destination) city strings from free-form input.
    Handles: 'from X to Y', 'X to Y', 'flights X to Y', etc.
    Returns (origin_str, dest_str) or (None, None) if parsing fails.
    """
    q = query.lower().strip()

    # Strip common lead words
    q = re.sub(r"^(flights?|fly|ticket[s]?)\s*", "", q).strip()

    # Try 'from X to Y'
    m = re.search(r"from\s+(.+?)\s+to\s+(.+)", q)
    if m:
        return m.group(1).strip(), m.group(2).strip()

    # Try plain 'X to Y'
    m = re.search(r"^(.+?)\s+to\s+(.+)$", q)
    if m:
        return m.group(1).strip(), m.group(2).strip()

    return None, None


def get_flights(query):
    try:
        origin_city, destination_city = _parse_flight_cities(query)

        if not origin_city or not destination_city:
            return (
                "❌ Couldn't parse cities. Use format: 'from Mumbai to Delhi'\n"
                f"Supported cities: {', '.join(sorted(set(AIRPORT_CODES.values())))}"
            )

        origin_code      = AIRPORT_CODES.get(origin_city.lower())
        destination_code = AIRPORT_CODES.get(destination_city.lower())

        if not origin_code:
            # Fuzzy match — check if input is contained in any key
            origin_code = next(
                (code for city, code in AIRPORT_CODES.items() if origin_city.lower() in city),
                None
            )
        if not destination_code:
            destination_code = next(
                (code for city, code in AIRPORT_CODES.items() if destination_city.lower() in city),
                None
            )

        if not origin_code:
            return (
                f"❌ Airport not found for '{origin_city}'.\n"
                f"Supported cities: {', '.join(c.title() for c in sorted(AIRPORT_CODES))}"
            )
        if not destination_code:
            return (
                f"❌ Airport not found for '{destination_city}'.\n"
                f"Supported cities: {', '.join(c.title() for c in sorted(AIRPORT_CODES))}"
            )

        travel_date = _next_weekday_date()

        params = {
            "engine": "google_flights",
            "departure_id": origin_code,
            "arrival_id": destination_code,
            "outbound_date": travel_date,
            "type": "2",          # 1=Round trip (needs return_date), 2=One-way
            "currency": "INR",
            "hl": "en",
            "api_key": serp_api_key,
        }


        with db.timer() as t:
            results = GoogleSearch(params).get_dict()

        # SerpApi returns results under 'best_flights' OR 'other_flights'
        flights = results.get("best_flights") or results.get("other_flights") or []

        if not flights:
            serpapi_msg = results.get("error") or results.get("search_information", {}).get("query_displayed", "")
            result = (
                f"No flights found from {origin_city.title()} ({origin_code}) "
                f"to {destination_city.title()} ({destination_code}) on {travel_date}."
                + (f"\nSerpApi said: {serpapi_msg}" if serpapi_msg else "")
            )
            db.log_tool_call("Flight Finder", query, result,
                             city=f"{origin_city}-{destination_city}", duration_ms=t.ms)
            return result

        output = (
            f"✈️ Flights from {origin_city.title()} to {destination_city.title()}"
            f" on {travel_date}:\n\n"
        )
        for flight in flights[:5]:
            legs        = flight.get("flights", [{}])
            airline     = legs[0].get("airline", "Unknown")
            price       = flight.get("price", "N/A")
            duration    = flight.get("total_duration", "N/A")
            departure   = legs[0].get("departure_airport", {}).get("time", "")
            arrival     = legs[-1].get("arrival_airport", {}).get("time", "")
            stops       = len(legs) - 1
            stop_label  = "Non-stop" if stops == 0 else f"{stops} stop(s)"
            output += (
                f"✈ {airline}  |  {stop_label}\n"
                f"💰 ₹{price}  |  ⏱ {duration} min\n"
                f"🕐 {departure} → {arrival}\n\n"
            )

        # ── Persist ──
        try:
            db.save_flights(origin_city, origin_code,
                            destination_city, destination_code,
                            travel_date, flights[:5])
            db.log_tool_call("Flight Finder", query, output,
                             city=f"{origin_city}-{destination_city}", duration_ms=t.ms)
        except Exception as e:
            db.log_error(str(e), tool_name="Flight Finder", exc=e)

        return output

    except Exception as e:
        db.log_error(str(e), tool_name="Flight Finder", exc=e)
        return f"Flight search error: {str(e)}"


flight_tool = Tool(
    name="Flight Finder",
    func=get_flights,
    description="""Use this when user asks about:
- flights
- flight tickets
- air travel
- how to fly from X to Y

Input format: "from [city] to [city]"
Example: "from Ahmedabad to Jaipur"

Do NOT use JSON or dicts — only plain text city names.""",
)


# ─────────────────────────────────────────────
# RAG — built once, reused across calls
# ─────────────────────────────────────────────
_rag_tool_cache = None   # cached Tool object
_rag_error_msg  = None   # set if FAISS index is missing or unavailable


def _build_rag_tool():
    """
    Try to load the FAISS index built from PDFs in knowledge_base/.
    Returns (Tool, None) on success, (None, error_str) on failure.
    Builds its own LLM internally using a fresh pool key.
    """
    global _rag_tool_cache, _rag_error_msg

    # Return cached result immediately
    if _rag_tool_cache is not None:
        return _rag_tool_cache, None
    if _rag_error_msg is not None:
        return None, _rag_error_msg

    try:
        from langchain_classic.chains import RetrievalQA
        from langchain_community.vectorstores import FAISS
        from pathlib import Path

        base_dir = Path(__file__).resolve().parent
        index_dir = base_dir / "faiss_index"
        index_file = index_dir / "index.faiss"
        metadata_file = index_dir / "index.pkl"

        if not index_dir.exists() or not index_file.exists() or not metadata_file.exists():
            raise FileNotFoundError(
                "FAISS index not found. Upload PDFs on the Knowledge Base page "
                "and click 'Rebuild Vector Database'."
            )

        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        vector_db = FAISS.load_local(
            str(index_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )

        retriever = vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5},
        )

        rag_llm = _make_llm()
        qa_chain = RetrievalQA.from_chain_type(
            llm=rag_llm,
            retriever=retriever,
            return_source_documents=True,
        )

        def travel_rag(q):
            """Answer from the travel PDF knowledge base and include source files."""
            resp = qa_chain.invoke({"query": q})
            source_documents = resp.get("source_documents", [])
            sources = sorted({
                Path(doc.metadata.get("source", "unknown")).name
                for doc in source_documents
                if doc.metadata.get("source")
            })

            answer = resp.get("result", "")
            if sources:
                answer = answer.rstrip() + "\n\nSources:\n" + "\n".join(
                    f"- {source}" for source in sources
                )

            return json.dumps(
                {
                    "answer": answer,
                    "sources": sources,
                },
                ensure_ascii=False,
            )

        _rag_tool_cache = Tool(
            name="Travel Knowledge Base",
            func=travel_rag,
            description=(
                "Use this for travel guides, itineraries, destinations, "
                "and general travel information."
            ),
        )
        return _rag_tool_cache, None

    except Exception as e:
        _rag_error_msg = str(e)
        try:
            db.log_error(_rag_error_msg, tool_name="Travel Knowledge Base", exc=e)
        except Exception:
            traceback.print_exc()
        return None, _rag_error_msg


def reset_rag_cache():
    """Clear the cached RAG tool/error so a rebuilt FAISS index is picked up."""
    global _rag_tool_cache, _rag_error_msg
    _rag_tool_cache = None
    _rag_error_msg = None
# ─────────────────────────────────────────────
# MAIN AGENT FUNCTION
# Returns a dict: {"text": str, "locations": list}
# ─────────────────────────────────────────────

# Rate limiter: free-tier gemini-2.0-flash-lite = 30 RPM (1 call per 2s)
# Increase _MIN_GAP to 4 if you still hit limits
_last_llm_call = 0.0
_MIN_GAP = 2.0

def _rate_limit():
    """Sleep just enough to stay under the free-tier RPM cap."""
    global _last_llm_call
    gap = time.time() - _last_llm_call
    if gap < _MIN_GAP:
        wait = _MIN_GAP - gap
        pass  # rate limiting silently
        time.sleep(wait)
    _last_llm_call = time.time()


# 
# pip install langchain-groq
from langchain_groq import ChatGroq

def _make_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",  # or "mixtral-8x7b-32768"
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3,
        max_tokens=800,
    )


def _invoke_with_retry(callable_obj, input_data, max_retries=4):
    """
    Call any LLM/agent invoke with rate limiting + exponential backoff.
    On quota exhaustion: waits 4s -> 8s -> 16s before each retry.
    Raises a clean human-readable error if all retries fail.
    """
    QUOTA_SIGNALS = ["resource_exhausted", "429", "503", "quota", "rate limit", "high demand"]

    for attempt in range(max_retries):
        _rate_limit()
        try:
            return callable_obj(input_data)
        except Exception as e:
            err = str(e).lower()
            is_quota = any(s in err for s in QUOTA_SIGNALS)
            if is_quota and attempt < max_retries - 1:
                wait = 2 ** (attempt + 2)   # 4s, 8s, 16s, 32s
                pass  # retrying silently
                time.sleep(wait)
            elif is_quota:
                raise Exception(
                    f"⚠️ Gemini free-tier quota exhausted after {max_retries} retries. "
                    f"Wait ~1 minute and try again, or upgrade to a Gemini paid plan at "
                    f"https://ai.google.dev/pricing"
                )
            else:
                raise


def _build_agent(tools, llm, memory, prefix):
    return initialize_agent(
        tools, llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=6,
        early_stopping_method="generate",
        agent_kwargs={"prefix": prefix},
        return_intermediate_steps=True
    )


def clear_tool_cache():
    """Call this when user clears chat — forces fresh API data on next query."""
    for fn in [get_restaurants, get_hotels, get_places, get_map_locations, get_weather]:
        try:
            fn.cache_clear()
        except AttributeError:
            pass
    reset_rag_cache()



# ─────────────────────────────────────────────
# PARALLEL TRIP PLANNER
# Detects "plan a trip" intent, runs all tools
# simultaneously, then makes ONE LLM call to
# format the final answer.
# This cuts Gemini API calls from ~7 → 2.
# ─────────────────────────────────────────────
_TRIP_KEYWORDS = [
    "plan", "trip", "travel", "visit", "itinerary",
    "tour", "holiday", "vacation", "weekend", "getaway"
]

def _is_trip_plan_query(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in _TRIP_KEYWORDS)


def _extract_city_from_query(query: str) -> str:
    """Best-effort city extraction from a trip plan query."""
    q = query.lower()
    # "trip to X", "visit X", "plan X trip", "travel to X"
    for pattern in [
        r"(?:trip|travel|visit|going|plan|itinerary)\s+(?:to|for|in)\s+([a-zA-Z\s]+?)(?:\s+for|\s+in|\s*$)",
        r"(?:to|in|for)\s+([a-zA-Z\s]+?)(?:\s+for|\s+in|\s*$)",
    ]:
        m = re.search(pattern, q)
        if m:
            city = m.group(1).strip().rstrip(".,?!")
            if len(city) > 2:
                return city.title()
    # Fallback: last capitalised word sequence in original query
    words = query.split()
    for word in reversed(words):
        if word[0].isupper() and len(word) > 2:
            return word
    return ""


def _parallel_trip_plan(city: str, memory, include_flights_from: str = "") -> dict:
    """
    Fetch weather, places, hotels, restaurants in parallel.
    Returns {"text": formatted_answer, "locations": [...]}
    Uses exactly ONE Gemini call to format everything.
    """

    tasks = {
        "weather":     (get_weather,     city),
        "places":      (get_places,      city),
        "hotels":      (get_hotels,      city),
        "restaurants": (get_restaurants, city),
    }

    results = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fn, arg): name for name, (fn, arg) in tasks.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as e:
                results[name] = f"Could not fetch {name}: {e}"


    # Parse locations out of places JSON
    locations = []
    places_raw = results.get("places", "{}")
    if isinstance(places_raw, str):
        try:
            parsed = json.loads(places_raw)
            locations = parsed.get("locations", [])
            results["places"] = parsed.get("text", places_raw)
        except (json.JSONDecodeError, TypeError):
            pass

    # Build a single compact prompt for the LLM
    history_text = ""
    if memory:
        msgs = memory.chat_memory.messages if hasattr(memory, "chat_memory") else []
        if msgs:
            history_text = "\n".join(
                f"{'User' if m.type == 'human' else 'AI'}: {m.content}"
                for m in msgs[-6:]  # last 3 turns
            )
    print("\nTOOL RESULTS FETCHED:")
    print("-" * 40)

    for key, value in results.items():
        print(f"\n[{key.upper()}]")
        print(value)
    prompt = f"""You are a travel planner. Using ONLY the data below, write a helpful trip plan for {city}.

## Real-time data fetched:
WEATHER: {results.get("weather", "N/A")}
PLACES:  {results.get("places",  "N/A")}
HOTELS:  {results.get("hotels",  "N/A")}
RESTAURANTS: {results.get("restaurants", "N/A")}

## Format your response as:
### 🌤 Weather
### 🗺 Top Attractions
### 🏨 Hotels
### 🍽 Restaurants
### 📅 Suggested Itinerary (2-3 days)

Keep it concise and practical. Do not make up data not provided above.
"""

    # One LLM call to format the final answer
    llm = _make_llm()
    try:
        resp = _invoke_with_retry(llm.invoke, prompt)
        answer = resp.content if hasattr(resp, "content") else str(resp)
        if memory:
            memory.chat_memory.add_user_message(f"Plan a trip to {city}")
            memory.chat_memory.add_ai_message(answer)
    except Exception as e:
        answer = f"❌ Error generating plan: {e}"

    return {"text": answer, "locations": locations}


def create_memory(label: str = ""):
    """
    Create a fresh ConversationBufferMemory and a matching DB session.
    Returns (memory, session_id) — store session_id and pass it to
    get_travel_agent() so messages are saved against the right session.
    """
    session_id = db.new_session(label=label)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )
    return memory, session_id


def get_travel_agent(query, memory=None, session_id: int | None = None):
    """
    memory     : pass the SAME ConversationBufferMemory object on every call
                 so the agent remembers previous turns.  If None a fresh
                 (amnesiac) memory is used — fine for one-shot calls.
    session_id : DB session id returned by create_memory().  Used to persist
                 every user message and assistant reply.  Safe to omit.
    """
    if memory is None:
        memory, session_id = create_memory()

    # ── Persist the user's message ──
    if session_id:
        try:
            db.save_message(session_id, "user", query)
        except Exception as e:
            db.log_error(str(e), session_id=session_id, exc=e)

    # ── Fast path: trip plan queries use parallel tool execution ──
    # Runs all 4 tools simultaneously + 1 LLM call instead of ~7 sequential LLM calls
    if _is_trip_plan_query(query):
        city = _extract_city_from_query(query)
        if city:
            result = _parallel_trip_plan(city, memory)
            if session_id:
                try:
                    db.save_message(session_id, "assistant", result.get("text", ""))
                    db.touch_session(session_id)
                except Exception as e:
                    db.log_error(str(e), session_id=session_id, exc=e)
            return result
        # If city extraction failed, fall through to agent

    # ── Standard path: single-tool queries go through the agent ──
    llm = _make_llm()

    # Try to load RAG — gracefully skip if files are missing
    rag_tool, rag_err = _build_rag_tool()
    rag_warning = ""
    if rag_err:
        rag_warning = (
            f"\n\n> ⚠️ **Travel Knowledge Base unavailable** "
            f"(`{rag_err}`). Itinerary suggestions will use the LLM's "
            f"built-in knowledge instead."
        )

    tools = [
        *([ rag_tool ] if rag_tool else []),
        weather_tool,
        places_tool,
        map_tool,
        hotel_tool,
        restaurant_tool,
        flight_tool,
    ]

    prefix = """
You are a smart AI travel planner. Follow these rules STRICTLY to avoid unnecessary API calls:

## TOOL SELECTION RULES — only call what is needed:
| User asks about...         | Tools to use                                      |
|----------------------------|---------------------------------------------------|
| Full trip plan             | Weather Tool + Tourist Places Finder + Hotel Finder + Restaurant Finder |
| Flights only               | Flight Finder only                                |
| Hotels only                | Hotel Finder only                                 |
| Weather only               | Weather Tool only                                 |
| Restaurants / food only    | Restaurant Finder only                            |
| Tourist places / map       | Tourist Places Finder only                        |
| General travel info        | Travel Knowledge Base only                        |
| Simple factual question    | No tool — answer directly from knowledge          |

## CRITICAL RULES:
- DO NOT call a tool if the answer is already in chat history or the Observation.
- DO NOT call Hotel Finder + Restaurant Finder unless user explicitly asked for both.
- For FOLLOW-UP questions (e.g. "what about hotels?", "and the weather?"), use chat history to get the city — do NOT ask again.
- Non-travel queries → reply: "Sorry, I can only help with travel-related queries."
- Use tools for REAL-TIME data only (weather, flights, hotels, places). Use your knowledge for itineraries.
-When a tool returns hotels, flights, weather, attractions, or itineraries:

- Return the tool output exactly as received.
- Do not summarize.
- Do not say "Based on the previous observation".
- Do not restate reasoning.
- Present the tool result directly to the user.

## FORMAT:
Thought: (decide which tools are needed — commit to all of them NOW, do not add more later)
Action: tool name
Action Input: city name or flight query
Observation: result
... (repeat only if genuinely needed)
Final Answer: well-structured response combining all observations

Be concise, organised, and helpful.
"""

    agent = _build_agent(tools, llm, memory, prefix)
    response = _invoke_with_retry(agent.invoke, {"input": query})
    raw_output = response.get("output", str(response)) if isinstance(response, dict) else str(response)
   


    # ── Try to extract any JSON location blocks embedded in tool observations ──
    # The places/map tools embed JSON; we fish it out of the agent's scratchpad
    # by re-running the relevant tool if the response mentions locations.
    locations = []

    # Check if the raw output contains location data (lat/lon pattern)
    lat_lon_pattern = re.findall(r"\((-?\d+\.\d+),\s*(-?\d+\.\d+)\)", raw_output)
    if lat_lon_pattern:
        for lat_str, lon_str in lat_lon_pattern:
            locations.append({"lat": float(lat_str), "lon": float(lon_str)})

    # Also try to get named locations by checking intermediate steps
    if hasattr(response, "get"):
        intermediate = response.get("intermediate_steps", [])
        for step in intermediate:
            if len(step) >= 2:
                obs = step[1]
                if isinstance(obs, str):
                    try:
                        parsed = json.loads(obs)
                        if "locations" in parsed and parsed["locations"]:
                            locations = parsed["locations"]  # prefer named ones
                            break
                    except (json.JSONDecodeError, TypeError):
                        pass

    flight_output = None

    if hasattr(response, "get"):
        intermediate = response.get("intermediate_steps", [])

        for step in intermediate:
            if len(step) >= 2:
                action = step[0]
                obs = step[1]

                tool_name = getattr(action, "tool", "")

                if tool_name == "Flight Finder":
                    flight_output = obs
                    break

    print("\nLOCATIONS RETURNED:")            
    if flight_output:
        final = {"text": flight_output + rag_warning, "locations": []}
    else:
        final = {"text": raw_output + rag_warning, "locations": locations}

    # ── Persist the assistant's reply ──
    if session_id:
        try:
            db.save_message(session_id, "assistant", final["text"])
            db.touch_session(session_id)
        except Exception as e:
            db.log_error(str(e), session_id=session_id, exc=e)

    return final

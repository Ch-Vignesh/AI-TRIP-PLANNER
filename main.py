# app.py
import os
import json
import logging
from uuid import uuid4
from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Optional
import asyncio
from functools import partial

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

# Optional imports (may fail if credentials / libs not present)
try:
    import googlemaps
except Exception:
    googlemaps = None

try:
    from google.cloud import aiplatform
    from google.cloud.aiplatform import gapic as aiplatform_gapic  # may or may not be present
except Exception:
    aiplatform = None

import requests

# set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ease_my_trip_demo")

# Environment
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
GOOGLE_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "alloy")

# If Vertex AI available, init (best-effort)
if aiplatform and GOOGLE_PROJECT:
    try:
        aiplatform.init(project=GOOGLE_PROJECT, location=os.getenv("GOOGLE_CLOUD_REGION", "us-central1"))
        logger.info("Initialized Vertex AI SDK.")
    except Exception as e:
        logger.warning(f"Could not initialize Vertex AI SDK: {e}")

# In-memory stores for demo
ITINERARIES: Dict[str, Dict[str, Any]] = {}
BOOKINGS: Dict[str, Dict[str, Any]] = {}

# Models
class Preferences(BaseModel):
    user_name: str
    destination: str
    start_date: date
    end_date: date
    budget: float = Field(..., description="Total budget in INR")
    themes: List[str] = Field(default_factory=list, description="e.g. heritage, nightlife, adventure")
    language: str = "en"

class Place(BaseModel):
    name: str
    place_id: Optional[str] = None
    description: Optional[str] = None
    opening_hours: Optional[str] = None
    avg_visit_hours: float = 1.0
    estimated_cost: float = 0.0
    rating: Optional[float] = None
    booking_link: Optional[str] = None
    note: Optional[str] = None

class DayPlan(BaseModel):
    date: date
    slots: Dict[str, List[Place]]  # keys: morning/afternoon/evening
    note: Optional[str] = None

class Itinerary(BaseModel):
    id: str
    user_name: str
    destination: str
    start_date: date
    end_date: date
    budget: float
    themes: List[str]
    language: str
    generated_at: datetime
    days: List[DayPlan]
    cost_breakdown: Dict[str, float]
    status: str = "draft"
    booking_id: Optional[str] = None

class ChatRequest(BaseModel):
    user_message: str

class BookRequest(BaseModel):
    payment_method: Optional[str] = "mock"

# Utility / fallback helpers

def compute_cost_breakdown(days: List[DayPlan]) -> Dict[str, float]:
    place_cost = 0.0
    for d in days:
        for slot, places in d.slots.items():
            for p in places:
                place_cost += p.estimated_cost or 0.0
    transit = len(days) * 300
    hotel = len(days) * 1200
    total = place_cost + transit + hotel
    return {"places": place_cost, "transit": transit, "hotel": hotel, "total_estimated": total}

# Google Maps integration (best-effort; fallback to stub)

async def search_places_gmaps(destination: str, theme: str, limit: int = 3) -> List[Place]:
    """
    Try Google Maps Places API. If not, use stub data.
    """
    if not GOOGLE_MAPS_API_KEY or not googlemaps:
        logger.info("Google Maps not configured — using stub places.")
        # stub
        return [
            Place(
                name=f"{theme.capitalize()} Spot {i} in {destination}",
                place_id=f"stub-{destination[:3]}-{theme[:3]}-{i}",
                description=f"Demo {theme} place #{i} in {destination}",
                opening_hours="09:00-18:00",
                estimated_cost=150 + i * 50,
                rating=4.0 - 0.2 * i
            ) for i in range(1, limit+1)
        ]
    # googlemaps present
    gmaps_client = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)
    # run blocking call in thread
    def _search():
        try:
            res = gmaps_client.places(query=f"{theme} in {destination}", language="en")
            return res
        except Exception as e:
            logger.warning(f"Maps places search failed: {e}")
            return {}

    resp = await asyncio.to_thread(_search)
    results = resp.get("results", []) if resp else []
    places: List[Place] = []
    for r in results[:limit]:
        place_id = r.get("place_id")
        # fetch details
        def _details():
            try:
                return gmaps_client.place(place_id, fields=["opening_hours", "rating", "formatted_address", "name"])
            except Exception as e:
                logger.warning(f"Place details failed: {e}")
                return {}
        details = await asyncio.to_thread(_details)
        det = details.get("result", {}) if details else {}
        places.append(Place(
            name=r.get("name"),
            place_id=place_id,
            description=det.get("formatted_address") or r.get("vicinity") or "",
            opening_hours=str(det.get("opening_hours", "")),
            estimated_cost=200,  # placeholder: could refine via categories / price_level
            rating=det.get("rating")
        ))
    if not places:
        # fallback stub
        return await search_places_gmaps(destination, theme, limit)  # will hit stub path if no key
    return places

async def estimate_transit_cost(origin: str, dest: str) -> float:
    """
    Try Google Directions -> estimate cost. Fallback to fixed.
    """
    if not GOOGLE_MAPS_API_KEY or not googlemaps:
        return 200.0
    gmaps_client = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)
    def _dirs():
        try:
            return gmaps_client.directions(origin, dest, mode="driving")
        except Exception as e:
            logger.warning(f"Directions API failed: {e}")
            return []
    directions = await asyncio.to_thread(_dirs)
    if not directions:
        return 200.0
    distance_m = directions[0]["legs"][0]["distance"]["value"]
    return max(100.0, (distance_m / 1000.0) * 20.0)

# ---------------------------
# Vertex AI / Gemini integration (best-effort; fallback to stub)
# ---------------------------

async def gemini_generate_itinerary(pref: Preferences, place_map: Dict[str, List[Place]]) -> List[DayPlan]:
    """
    Use Vertex AI (Gemini) to produce itinerary JSON. If Vertex isn't configured, produce a deterministic stub.
    """
    # If Vertex client not available, use stub generation
    if not aiplatform:
        logger.info("Vertex AI SDK not available — using stub itinerary generation.")
        return await _stub_generate_itinerary(pref, place_map)

    # Try to generate using Vertex AI TextGenerationModel
    try:
        # Use the TextGenerationModel if available in SDK
        try:
            # Newer SDK: TextGenerationModel
            ModelClass = getattr(aiplatform, "TextGenerationModel", None)
            if ModelClass:
                model = ModelClass.from_pretrained(os.getenv("VERTEX_MODEL", "text-bison@001"))
                prompt = _build_itinerary_prompt(pref, place_map)
                # generate synchronously, so call in thread
                def _generate():
                    response = model.generate(prompt=prompt, temperature=0.2, max_output_tokens=1024)
                    return response
                resp = await asyncio.to_thread(_generate)
                # try parse JSON from response.text
                text = ""
                # SDK response shapes vary; being defensive
                if hasattr(resp, "text"):
                    text = resp.text
                elif hasattr(resp, "generated_text"):
                    text = resp.generated_text
                else:
                    # try candidate content
                    try:
                        text = resp[0].text
                    except Exception:
                        text = str(resp)
                itinerary_json = _safe_json_loads(text)
                if itinerary_json and "days" in itinerary_json:
                    days = [DayPlan(**d) for d in itinerary_json["days"]]
                    return days
                else:
                    # fallback stub
                    logger.warning("Vertex output didn't contain expected JSON — falling back to stub.")
                    return await _stub_generate_itinerary(pref, place_map)
            else:
                # older SDK pattern: aiplatform.gapic?
                logger.warning("TextGenerationModel not found in aiplatform SDK — using stub.")
                return await _stub_generate_itinerary(pref, place_map)
        except Exception as e:
            logger.warning(f"Exception while calling Vertex SDK: {e}")
            return await _stub_generate_itinerary(pref, place_map)
    except Exception as e:
        logger.warning(f"Vertex AI not used: {e}")
        return await _stub_generate_itinerary(pref, place_map)

def _build_itinerary_prompt(pref: Preferences, place_map: Dict[str, List[Place]]) -> str:
    # Build a prompt with candidate places serialized as JSON for the LLM
    candidate_summary = {}
    for k, v in place_map.items():
        candidate_summary[k] = [p.dict() for p in v]
    prompt = f"""
    You are a trip-planning assistant for EaseMyTrip. Produce a strict JSON object with a top-level key "days" that is a list.
    Each day must be of form:
    {{"date":"YYYY-MM-DD", "slots":{{"morning":[{{...}}],"afternoon":[{{...}}],"evening":[{{...}}]}}}}

    Constraints:
    - Use only candidate places provided below.
    - Include for each place: name, place_id (if any), description, estimated_cost, rating.
    - Keep the JSON compact and valid (no extra commentary).
    User preferences:
    {pref.json()}
    Candidate places:
    {json.dumps(candidate_summary, default=str, indent=2)}
    Return ONLY the JSON object.
    """
    return prompt

def _safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    # try to find first '{' and last '}' to extract JSON substring (defensive)
    try:
        start = text.index("{")
        end = text.rfind("}")
        json_text = text[start:end+1]
        return json.loads(json_text)
    except Exception as e:
        logger.warning(f"Could not parse JSON from LLM output: {e}")
        return None

async def _stub_generate_itinerary(pref: Preferences, place_map: Dict[str, List[Place]]) -> List[DayPlan]:
    days = []
    total_days = max(1, (pref.end_date - pref.start_date).days + 1)
    # rotate themes into slots
    themes = list(place_map.keys())
    for d in range(total_days):
        day_date = pref.start_date + timedelta(days=d)
        morning = [p for p in place_map.get("heritage", [])[:1]]
        afternoon = [p for p in place_map.get("adventure", [])[:1]]
        evening = [p for p in place_map.get("nightlife", [])[:1]]
        days.append(DayPlan(date=day_date, slots={"morning": morning, "afternoon": afternoon, "evening": evening}))
    return days

# ---------------------------
# ElevenLabs TTS integration (HTTP) - best-effort, fallback to text-only
# ---------------------------

def elevenlabs_generate_tts_file(itinerary: Dict[str, Any], output_dir: str = "/tmp") -> str:
    """
    Generate TTS audio via ElevenLabs REST API.
    Returns filepath to mp3.
    If ELEVENLABS_API_KEY not provided, returns a 'not-available' placeholder filepath.
    """
    if not ELEVENLABS_API_KEY:
        logger.info("ElevenLabs key not configured — skipping TTS and returning placeholder.")
        # write a small txt file to indicate placeholder
        placeholder = os.path.join(output_dir, f"tts_{itinerary['id']}_placeholder.txt")
        with open(placeholder, "w", encoding="utf-8") as f:
            f.write("TTS not available. ElevenLabs API key not configured.")
        return placeholder

    # Build short summary text
    text = f"Hello {itinerary['user_name']}. Your trip to {itinerary['destination']} from {itinerary['start_date']} to {itinerary['end_date']} includes {len(itinerary['days'])} days of activities. "
    for d in itinerary["days"]:
        morning_names = ", ".join([p["name"] for p in d["slots"].get("morning", [])]) if d["slots"].get("morning") else "no morning activities"
        text += f"On {d['date']}, morning: {morning_names}. "

    # ElevenLabs TTS API v1: POST /v1/text-to-speech/{voice_id}
    voice_id = ELEVENLABS_VOICE_ID or "alloy"
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, stream=True, timeout=30)
        if resp.status_code != 200:
            logger.warning(f"ElevenLabs TTS failed {resp.status_code}: {resp.text}")
            # fallback to placeholder
            placeholder = os.path.join(output_dir, f"tts_{itinerary['id']}_error.txt")
            with open(placeholder, "w", encoding="utf-8") as f:
                f.write(f"TTS generation failed: {resp.status_code} {resp.text}")
            return placeholder
        # Save audio content
        mp3_path = os.path.join(output_dir, f"tts_{itinerary['id']}.mp3")
        with open(mp3_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return mp3_path
    except Exception as e:
        logger.warning(f"Exception calling ElevenLabs: {e}")
        placeholder = os.path.join(output_dir, f"tts_{itinerary['id']}_exception.txt")
        with open(placeholder, "w", encoding="utf-8") as f:
            f.write(f"TTS exception: {e}")
        return placeholder

# FastAPI app & endpoints
app = FastAPI(title="EaseMyTrip - Personalized Trip Planner (Prototype)")

@app.get("/")
def root():
    return {"message": "EaseMyTrip demo API running. Use /create_itinerary -> /get_itinerary -> /chat -> /book -> /tts endpoints."}

@app.post("/create_itinerary", response_model=Itinerary)
async def create_itinerary(pref: Preferences):
    """
    Orchestrator:
    1. Search places (Maps)
    2. Call Gemini (Vertex) to assemble day-wise itinerary JSON
    3. Compute cost
    4. Store and return
    """
    logger.info(f"Creating itinerary for {pref.user_name} -> {pref.destination} ({pref.start_date} - {pref.end_date})")
    itinerary_id = str(uuid4())
    place_map: Dict[str, List[Place]] = {}

    # gather candidate places in parallel for themes
    themes = pref.themes or ["heritage", "adventure", "nightlife"]
    tasks = [search_places_gmaps(pref.destination, t, limit=3) for t in themes]
    results = await asyncio.gather(*tasks)
    for t, r in zip(themes, results):
        place_map[t] = r

    # call Gemini (Vertex) to produce days
    days = await gemini_generate_itinerary(pref, place_map)

    # compute simple cost
    cost_breakdown = compute_cost_breakdown(days)

    it = Itinerary(
        id=itinerary_id,
        user_name=pref.user_name,
        destination=pref.destination,
        start_date=pref.start_date,
        end_date=pref.end_date,
        budget=pref.budget,
        themes=pref.themes,
        language=pref.language,
        generated_at=datetime.utcnow(),
        days=days,
        cost_breakdown=cost_breakdown,
        status="draft"
    )
    ITINERARIES[itinerary_id] = it.dict()
    logger.info(f"Itinerary {itinerary_id} created.")
    return it

@app.get("/get_itinerary/{itinerary_id}", response_model=Itinerary)
async def get_itinerary(itinerary_id: str):
    data = ITINERARIES.get(itinerary_id)
    if not data:
        raise HTTPException(status_code=404, detail="Itinerary not found")
    return Itinerary(**data)

@app.post("/chat/{itinerary_id}")
async def chat_itinerary(itinerary_id: str, request: ChatRequest):
    """
    Send user message to Gemini (chat with itinerary context).
    Fallback: apply small rule-based edits if Vertex not available or LLM helped produce patch.
    """
    it = ITINERARIES.get(itinerary_id)
    if not it:
        raise HTTPException(status_code=404, detail="Itinerary not found")
    user_msg = request.user_message.strip()
    logger.info(f"Chat for {itinerary_id}: {user_msg}")

    # If Vertex available, attempt to call it; otherwise apply basic rules
    if aiplatform:
        try:
            # Build prompt with itinerary and user question
            prompt = f"""
            You are an assistant that edits/generated itineraries in JSON.
            Current itinerary JSON:
            {json.dumps(it, default=str, indent=2)}

            User asks:
            {user_msg}

            Return a JSON object with a single key "patch" which is either:
            - {"op":"replace","path":"/days/0/slots/morning","value":[...]} style patch operations OR
            - Provide "explanation": "..." and "patch": [...]
            Return ONLY JSON.
            """
            # attempt to call Vertex text generation (best-effort)
            try:
                ModelClass = getattr(aiplatform, "TextGenerationModel", None)
                if ModelClass:
                    model = ModelClass.from_pretrained(os.getenv("VERTEX_MODEL", "text-bison@001"))
                    def _generate():
                        return model.generate(prompt=prompt, temperature=0.2, max_output_tokens=512)
                    resp = await asyncio.to_thread(_generate)
                    # try extract text
                    text = ""
                    if hasattr(resp, "text"):
                        text = resp.text
                    elif hasattr(resp, "generated_text"):
                        text = resp.generated_text
                    else:
                        text = str(resp)
                    parsed = _safe_json_loads(text)
                    if parsed and "patch" in parsed:
                        patch = parsed["patch"]
                        if isinstance(patch, list):
                            for p in patch:
                                if p.get("op") == "remove" and p.get("path", "").endswith("evening"):
                                    for d in it["days"]:
                                        d["slots"]["evening"] = []
                            # store
                            it["cost_breakdown"] = compute_cost_breakdown([DayPlan(**d) for d in it["days"]])
                            ITINERARIES[itinerary_id] = it
                            return {"reply": parsed.get("explanation", "Applied patch."), "itinerary": it}
                        else:
                            return {"reply": parsed.get("explanation", "Produced patch but not applied."), "itinerary": it}
                    else:
                        # fallback
                        logger.info("No actionable patch returned by LLM; using rule-based fallback.")
                else:
                    logger.info("Vertex TextGenerationModel not found in SDK; using fallback rules.")
            except Exception as e:
                logger.warning(f"Error calling Vertex in chat endpoint: {e}")
        except Exception as e:
            logger.warning(f"Vertex chat path failed: {e}")

    # Fallback / simple rule-based edits
    msg = user_msg.lower()
    response_text = "No LLM change applied. Fallback actions: "
    if "remove nightlife" in msg or "skip nightlife" in msg or "no nightlife" in msg:
        for d in it["days"]:
            d["slots"]["evening"] = []
        response_text += "Removed nightlife from evenings."
    elif "move" in msg and "morning" in msg:
        # move first evening item to morning for day 1 if exists
        day0 = it["days"][0]
        ev = day0["slots"].get("evening", [])
        if ev:
            moved = ev.pop(0)
            day0["slots"]["morning"].insert(0, moved)
            response_text += f"Moved '{moved['name']}' to morning on {day0['date']}."
        else:
            response_text += "No evening item to move."
    else:
        response_text += "No known rule matched. (You can say 'remove nightlife' or 'move X to morning')"

    # recompute cost and save
    it["cost_breakdown"] = compute_cost_breakdown([DayPlan(**d) for d in it["days"]])
    ITINERARIES[itinerary_id] = it
    return {"reply": response_text, "itinerary": it}

@app.post("/book/{itinerary_id}")
async def book_itinerary(itinerary_id: str, payload: BookRequest):
    it = ITINERARIES.get(itinerary_id)
    if not it:
        raise HTTPException(status_code=404, detail="Itinerary not found")
    if it.get("status") == "booked":
        raise HTTPException(status_code=400, detail="Already booked")

    booking_id = str(uuid4())
    # Simulated orchestration (hotel, cab, experiences)
    confirmations = {
        "hotel": {"status": "confirmed", "provider_ref": f"HTL-{booking_id[:8]}"},
        "cab": {"status": "confirmed", "provider_ref": f"CAB-{booking_id[:8]}"},
        "experience": {"status": "confirmed", "provider_ref": f"EXP-{booking_id[:8]}"},
    }
    BOOKINGS[booking_id] = {
        "itinerary_id": itinerary_id,
        "booked_at": datetime.utcnow().isoformat(),
        "payment_method": payload.payment_method,
        "confirmations": confirmations
    }
    it["status"] = "booked"
    it["booking_id"] = booking_id
    ITINERARIES[itinerary_id] = it
    logger.info(f"Itinerary {itinerary_id} booked -> booking {booking_id}")
    return {"message": "Booking simulated and confirmed", "booking_id": booking_id, "confirmations": confirmations}

@app.post("/webhook/live-update")
async def webhook_live_update(payload: Dict[str, Any]):
    """
    Accept live events such as closures or weather and annotate affected itineraries.
    Example payloads:
    - {"type":"closure","place_id":"...","start":"2025-10-02","end":"2025-10-03"}
    - {"type":"weather","date":"2025-10-02","severity":"heavy_rain"}
    """
    logger.info(f"Received webhook: {payload}")
    changed = []
    for id_, it in ITINERARIES.items():
        updated = False
        if payload.get("type") == "closure" and payload.get("place_id"):
            pid = payload["place_id"]
            for d in it["days"]:
                for slot, places in d["slots"].items():
                    for p in places:
                        if p.get("place_id") == pid:
                            p["note"] = f"Closed on {payload.get('start')}. LLM suggested alternatives."
                            updated = True
        elif payload.get("type") == "weather" and payload.get("date"):
            for d in it["days"]:
                if str(d["date"]) == payload["date"]:
                    d["note"] = f"Weather alert: {payload.get('severity')}. Consider indoor options."
                    updated = True
        if updated:
            ITINERARIES[id_] = it
            changed.append(id_)
    return {"changed_itineraries": changed}

@app.get("/tts/{itinerary_id}")
async def itinerary_tts(itinerary_id: str):
    it = ITINERARIES.get(itinerary_id)
    if not it:
        raise HTTPException(status_code=404, detail="Itinerary not found")
    # Ensure consistent dict form
    audio_path = elevenlabs_generate_tts_file(it)
    return {"audio_file": audio_path}

@app.get("/list_itineraries")
async def list_itineraries():
    return {"count": len(ITINERARIES), "ids": list(ITINERARIES.keys())}

@app.get("/get_booking/{booking_id}")
async def get_booking(booking_id: str):
    b = BOOKINGS.get(booking_id)
    if not b:
        raise HTTPException(status_code=404, detail="Booking not found")
    return b

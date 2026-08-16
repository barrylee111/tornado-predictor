const BASE = '/api'

export async function fetchPrediction(lat, lon) {
  const r = await fetch(`${BASE}/predict?lat=${lat}&lon=${lon}`)
  if (!r.ok) throw new Error(`Prediction failed: ${r.status}`)
  return r.json()
}

export async function fetchWeather(lat, lon) {
  const r = await fetch(`${BASE}/weather?lat=${lat}&lon=${lon}`)
  if (!r.ok) throw new Error(`Weather fetch failed: ${r.status}`)
  return r.json()
}

export async function fetchTTS(text, voiceId) {
  const r = await fetch(`${BASE}/tts`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, voice_id: voiceId }),
  })
  if (!r.ok) throw new Error(`TTS failed: ${r.status}`)
  return r.json()
}

// Open-Meteo geocoding — handles city names AND zip codes, no key needed
export async function geocodeQuery(query) {
  if (!query || query.trim().length < 2) return []
  const params = new URLSearchParams({
    name: query.trim(),
    count: 8,
    language: 'en',
    format: 'json',
    country_code: 'US',
  })
  const r = await fetch(`https://geocoding-api.open-meteo.com/v1/search?${params}`)
  if (!r.ok) return []
  const data = await r.json()
  return (data.results || []).map(loc => ({
    id: loc.id,
    label: [loc.name, loc.admin1].filter(Boolean).join(', '),
    lat: loc.latitude,
    lon: loc.longitude,
    zip: loc.postcodes?.[0] ?? null,
  }))
}

// Zippopotam.us — direct zip → lat/lon lookup for exact 5-digit zip entry
export async function geocodeZip(zip) {
  try {
    const r = await fetch(`https://api.zippopotam.us/us/${zip}`)
    if (!r.ok) return null
    const data = await r.json()
    const place = data.places?.[0]
    if (!place) return null
    return {
      label: `${place['place name']}, ${place['state abbreviation']} ${zip}`,
      lat: parseFloat(place.latitude),
      lon: parseFloat(place.longitude),
      zip,
    }
  } catch {
    return null
  }
}

import { useState, useEffect } from 'react'

// Default to center of Tornado Alley if geolocation denied
const FALLBACK = { lat: 35.5, lon: -97.5, label: 'Oklahoma City, OK' }

export function useGeolocation() {
  const [location, setLocation] = useState(null)
  const [status, setStatus] = useState('pending') // pending | granted | denied | unsupported

  useEffect(() => {
    if (!navigator.geolocation) {
      setStatus('unsupported')
      setLocation(FALLBACK)
      return
    }

    navigator.geolocation.getCurrentPosition(
      pos => {
        setLocation({ lat: pos.coords.latitude, lon: pos.coords.longitude, label: 'Your location' })
        setStatus('granted')
      },
      () => {
        setStatus('denied')
        setLocation(FALLBACK)
      },
      { timeout: 8000, maximumAge: 60000 }
    )
  }, [])

  return { location, status }
}

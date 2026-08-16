import { useState, useEffect } from 'react'

const RAINVIEWER_API = 'https://api.rainviewer.com/public/weather-maps.json'

export function useRainViewer() {
  const [radarUrl, setRadarUrl] = useState(null)
  const [timestamps, setTimestamps] = useState([])
  const [currentIndex, setCurrentIndex] = useState(-1)

  useEffect(() => {
    let cancelled = false
    async function load() {
      try {
        const r = await fetch(RAINVIEWER_API)
        const data = await r.json()
        const past = data.radar?.past ?? []
        if (!past.length || cancelled) return
        setTimestamps(past)
        setCurrentIndex(past.length - 1)
        const latest = past[past.length - 1]
        setRadarUrl(buildUrl(latest.path))
      } catch (e) {
        console.warn('RainViewer fetch failed:', e)
      }
    }
    load()
    // Refresh every 5 minutes
    const interval = setInterval(load, 5 * 60 * 1000)
    return () => { cancelled = true; clearInterval(interval) }
  }, [])

  function buildUrl(path) {
    // color=2 (NOAA style), smooth=1, snow=0
    return `https://tilecache.rainviewer.com${path}/256/{z}/{x}/{y}/2/1_0.png`
  }

  function stepFrame(dir) {
    setCurrentIndex(prev => {
      const next = Math.min(Math.max(0, prev + dir), timestamps.length - 1)
      if (timestamps[next]) setRadarUrl(buildUrl(timestamps[next].path))
      return next
    })
  }

  return { radarUrl, stepFrame, frameCount: timestamps.length, currentIndex }
}

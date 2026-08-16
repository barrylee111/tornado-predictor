import { useState, useCallback } from 'react'
import { fetchPrediction, fetchWeather } from '../api/client'

export function usePrediction() {
  const [prediction, setPrediction] = useState(null)
  const [weather, setWeather] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const predict = useCallback(async (lat, lon) => {
    setLoading(true)
    setError(null)
    try {
      const [pred, wx] = await Promise.all([
        fetchPrediction(lat, lon),
        fetchWeather(lat, lon),
      ])
      setPrediction(pred)
      setWeather(wx)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [])

  return { prediction, weather, loading, error, predict }
}

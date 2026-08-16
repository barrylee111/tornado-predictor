import { useState, useEffect, useCallback } from 'react'
import { CloudLightning, MapPin, RefreshCw } from 'lucide-react'
import { RadarMap } from './components/RadarMap'
import { ZipSearch } from './components/ZipSearch'
import { RiskPanel } from './components/RiskPanel'
import { AtmosphericReadings } from './components/AtmosphericReadings'
import { VoiceButton } from './components/VoiceButton'
import { useGeolocation } from './hooks/useGeolocation'
import { usePrediction } from './hooks/usePrediction'

export default function App() {
  const { location: geoLoc, status: geoStatus } = useGeolocation()
  const { prediction, weather, loading, error, predict } = usePrediction()

  const [selectedLoc, setSelectedLoc] = useState(null)

  // Auto-predict on geolocation load
  useEffect(() => {
    if (geoLoc && !selectedLoc) {
      setSelectedLoc(geoLoc)
      predict(geoLoc.lat, geoLoc.lon)
    }
  }, [geoLoc])

  const handleSelect = useCallback((loc) => {
    setSelectedLoc(loc)
    predict(loc.lat, loc.lon)
  }, [predict])

  const handleMapClick = useCallback((lat, lon) => {
    const loc = { lat, lon, label: `${lat.toFixed(3)}°, ${lon.toFixed(3)}°` }
    setSelectedLoc(loc)
    predict(lat, lon)
  }, [predict])

  const handleRefresh = () => {
    if (selectedLoc) predict(selectedLoc.lat, selectedLoc.lon)
  }

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-storm-950">
      {/* Sidebar */}
      <aside className="w-80 shrink-0 flex flex-col bg-storm-900 border-r border-storm-700 z-10">
        {/* Header */}
        <div className="px-4 py-4 border-b border-storm-700">
          <div className="flex items-center gap-2 mb-1">
            <CloudLightning size={20} className="text-sky-400" />
            <span className="font-bold text-white tracking-tight">TornadoWatch</span>
          </div>
          <p className="text-xs text-gray-500">2-hour tornado risk · NEXRAD radar</p>
        </div>

        {/* Location */}
        <div className="px-4 py-3 border-b border-storm-700 space-y-2">
          <ZipSearch
            currentLabel={selectedLoc?.label}
            onSelect={handleSelect}
          />
          {selectedLoc && (
            <div className="flex items-center justify-between text-xs text-gray-500">
              <div className="flex items-center gap-1.5">
                <MapPin size={11} />
                <span className="truncate max-w-[170px]">{selectedLoc.label}</span>
              </div>
              <button
                onClick={handleRefresh}
                className="flex items-center gap-1 hover:text-gray-300 transition-colors"
                title="Refresh prediction"
              >
                <RefreshCw size={11} className={loading ? 'animate-spin' : ''} />
                <span>Refresh</span>
              </button>
            </div>
          )}
          {geoStatus === 'denied' && (
            <div className="text-xs text-amber-500">
              Location access denied — defaulted to Oklahoma City.
            </div>
          )}
        </div>

        {/* Prediction */}
        <div className="px-4 py-4 border-b border-storm-700">
          <RiskPanel prediction={prediction} loading={loading} error={error} />
        </div>

        {/* Voice button */}
        {prediction && (
          <div className="px-4 py-3 border-b border-storm-700">
            <VoiceButton
              narrative={prediction.narrative}
              riskLevel={prediction.risk_level}
              disabled={loading}
            />
          </div>
        )}

        {/* Atmospheric readings */}
        <div className="px-4 py-4 flex-1 overflow-y-auto">
          <AtmosphericReadings prediction={prediction} weather={weather} />
        </div>

        {/* Footer */}
        <div className="px-4 py-3 border-t border-storm-700 text-xs text-gray-600 space-y-0.5">
          <div>Radar: RainViewer · Data: NOAA / Open-Meteo</div>
          <div>Model target: 2-hour coordinate prediction</div>
        </div>
      </aside>

      {/* Map */}
      <main className="flex-1 relative">
        <RadarMap
          selectedLat={selectedLoc?.lat}
          selectedLon={selectedLoc?.lon}
          prediction={prediction}
          onMapClick={handleMapClick}
        />
      </main>
    </div>
  )
}

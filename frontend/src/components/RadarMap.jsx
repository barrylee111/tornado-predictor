import { useEffect, useRef } from 'react'
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet'
import L from 'leaflet'
import { useRainViewer } from '../hooks/useRainViewer'
import { ChevronLeft, ChevronRight } from 'lucide-react'

// Fix Leaflet default icon path issue with bundlers
delete L.Icon.Default.prototype._getIconUrl
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
})

function riskIcon(riskLevel) {
  const colors = { LOW: '#10b981', ELEVATED: '#eab308', HIGH: '#f97316', EXTREME: '#ef4444' }
  const color = colors[riskLevel] ?? colors.LOW
  const pulse = (riskLevel === 'HIGH' || riskLevel === 'EXTREME')
    ? `<div style="position:absolute;inset:-6px;border-radius:50%;border:2px solid ${color};opacity:0.5;animation:risk-pulse 1.4s ease-in-out infinite;"></div>` : ''
  return L.divIcon({
    html: `<div style="position:relative;display:flex;align-items:center;justify-content:center;">
      ${pulse}
      <div style="width:18px;height:18px;border-radius:50%;background:${color};border:2px solid white;box-shadow:0 0 8px ${color}88;"></div>
    </div>`,
    className: '',
    iconSize: [24, 24],
    iconAnchor: [12, 12],
  })
}

function FlyTo({ lat, lon }) {
  const map = useMap()
  useEffect(() => {
    if (lat != null && lon != null) {
      map.flyTo([lat, lon], Math.max(map.getZoom(), 8), { duration: 1.2 })
    }
  }, [lat, lon, map])
  return null
}

function MapClickHandler({ onMapClick }) {
  const map = useMap()
  useEffect(() => {
    map.on('click', e => onMapClick(e.latlng.lat, e.latlng.lng))
    return () => map.off('click')
  }, [map, onMapClick])
  return null
}

export function RadarMap({ selectedLat, selectedLon, prediction, onMapClick }) {
  const { radarUrl, stepFrame, frameCount, currentIndex } = useRainViewer()

  const center = [selectedLat ?? 37, selectedLon ?? -95]

  return (
    <div className="relative w-full h-full">
      <MapContainer
        center={center}
        zoom={selectedLat ? 8 : 5}
        className="w-full h-full"
        zoomControl={false}
      >
        {/* Dark base map */}
        <TileLayer
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          attribution='&copy; <a href="https://carto.com/">CARTO</a>'
          subdomains="abcd"
          maxZoom={19}
        />

        {/* RainViewer radar overlay */}
        {radarUrl && (
          <TileLayer
            key={radarUrl}
            url={radarUrl}
            opacity={0.65}
            zIndex={200}
            attribution='Radar &copy; <a href="https://rainviewer.com">RainViewer</a>'
          />
        )}

        {/* Fly to selected location */}
        {selectedLat && <FlyTo lat={selectedLat} lon={selectedLon} />}

        {/* Click handler */}
        <MapClickHandler onMapClick={onMapClick} />

        {/* Marker */}
        {selectedLat && (
          <Marker
            position={[selectedLat, selectedLon]}
            icon={riskIcon(prediction?.risk_level)}
          >
            {prediction && (
              <Popup className="tornado-popup">
                <div className="text-sm font-medium text-gray-900">
                  {prediction.risk_level} risk
                  <br />
                  <span className="text-gray-600 font-normal text-xs">
                    {Math.round(prediction.prob_2h * 100)}% in 2h
                  </span>
                </div>
              </Popup>
            )}
          </Marker>
        )}
      </MapContainer>

      {/* Radar scrubber */}
      {frameCount > 0 && (
        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex items-center gap-2 bg-storm-900/90 border border-storm-600 rounded-full px-3 py-1.5 backdrop-blur-sm z-[1000]">
          <button
            onClick={() => stepFrame(-1)}
            className="text-gray-400 hover:text-white transition-colors"
          >
            <ChevronLeft size={16} />
          </button>
          <span className="text-xs text-gray-400 font-mono min-w-[80px] text-center">
            Radar {currentIndex + 1}/{frameCount}
          </span>
          <button
            onClick={() => stepFrame(1)}
            className="text-gray-400 hover:text-white transition-colors"
          >
            <ChevronRight size={16} />
          </button>
        </div>
      )}

      {/* Map legend */}
      <div className="absolute top-3 right-3 bg-storm-900/90 border border-storm-600 rounded-lg p-2 text-xs text-gray-400 z-[1000] backdrop-blur-sm space-y-1">
        {[['LOW','emerald'],['ELEVATED','yellow'],['HIGH','orange'],['EXTREME','red']].map(([lvl, c]) => (
          <div key={lvl} className="flex items-center gap-2">
            <div className={`w-2.5 h-2.5 rounded-full bg-${c}-500`} />
            <span>{lvl}</span>
          </div>
        ))}
        <div className="border-t border-storm-600 mt-1 pt-1 text-gray-500">Click map to predict</div>
      </div>
    </div>
  )
}

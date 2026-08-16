import { Thermometer, Droplets, Wind, Activity } from 'lucide-react'

function Reading({ label, value, unit, icon: Icon, highlight }) {
  return (
    <div className={`flex items-center justify-between py-1.5 border-b border-storm-700 last:border-0 ${highlight ? 'text-yellow-300' : ''}`}>
      <div className="flex items-center gap-2 text-xs text-gray-400">
        {Icon && <Icon size={12} />}
        <span>{label}</span>
      </div>
      <div className="text-xs font-mono font-semibold text-white">
        {value != null && value !== 0 ? `${typeof value === 'number' ? value.toFixed(0) : value} ${unit}` : '—'}
      </div>
    </div>
  )
}

export function AtmosphericReadings({ prediction, weather }) {
  const atm = prediction?.atmospheric_summary ?? {}
  const wx = weather ?? {}

  const cape = atm.cape ?? wx.cape_jkg ?? null
  const cin = atm.cin ?? null
  const srh = atm.srh_03km ?? null
  const shear = atm.shear_06km ?? null
  const lifted = atm.lifted_index ?? wx.lifted_index ?? null
  const temp = wx.temperature_c ?? null
  const dewpoint = wx.dewpoint_c ?? null
  const windspeed = wx.windspeed_ms != null ? (wx.windspeed_ms * 1.944).toFixed(0) : null // m/s → knots

  const capeHigh = cape != null && cape > 1500
  const srhHigh  = srh != null && srh > 150

  return (
    <div className="bg-storm-800 rounded-lg p-3 border border-storm-600">
      <div className="text-xs uppercase tracking-wider text-gray-500 mb-2">Atmospheric Conditions</div>
      <Reading label="CAPE"        value={cape}     unit="J/kg"  icon={Activity}    highlight={capeHigh} />
      <Reading label="CIN"         value={cin}      unit="J/kg"  icon={Activity} />
      <Reading label="Helicity 0–3km" value={srh}   unit="m²/s²" icon={Wind}       highlight={srhHigh} />
      <Reading label="Bulk Shear"  value={shear}    unit="m/s"   icon={Wind} />
      <Reading label="Lifted Index" value={lifted}  unit="K"     icon={Activity} />
      <Reading label="Temperature" value={temp != null ? temp.toFixed(1) : null} unit="°C" icon={Thermometer} />
      <Reading label="Dewpoint"    value={dewpoint != null ? dewpoint.toFixed(1) : null} unit="°C" icon={Droplets} />
      <Reading label="Wind"        value={windspeed} unit="kts"  icon={Wind} />
    </div>
  )
}

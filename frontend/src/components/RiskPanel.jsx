import { Wind, Zap, Clock, TrendingUp } from 'lucide-react'

const RISK_COLORS = {
  LOW:      { bar: 'bg-emerald-500', text: 'text-emerald-400', glow: '' },
  ELEVATED: { bar: 'bg-yellow-500',  text: 'text-yellow-400',  glow: '' },
  HIGH:     { bar: 'bg-orange-500',  text: 'text-orange-400',  glow: 'risk-pulse' },
  EXTREME:  { bar: 'bg-red-500',     text: 'text-red-400',     glow: 'risk-pulse' },
}

function ProbBar({ label, value, color }) {
  const pct = Math.round(value * 100)
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs text-gray-400">
        <span>{label}</span>
        <span className="font-mono font-semibold text-white">{pct}%</span>
      </div>
      <div className="h-1.5 bg-storm-700 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-700 ${color}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}

export function RiskPanel({ prediction, loading, error }) {
  if (loading) {
    return (
      <div className="flex flex-col gap-3 animate-pulse">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="h-8 bg-storm-700 rounded-lg" />
        ))}
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-red-950 border border-red-800 rounded-lg p-3 text-red-300 text-sm">
        {error}
      </div>
    )
  }

  if (!prediction) {
    return (
      <div className="text-gray-500 text-sm text-center py-4">
        Click the map or search a location to get a prediction.
      </div>
    )
  }

  const colors = RISK_COLORS[prediction.risk_level] ?? RISK_COLORS.LOW

  return (
    <div className="space-y-4">
      {/* Risk badge */}
      <div className={`flex items-center justify-between rounded-lg px-4 py-3 risk-${prediction.risk_level} ${colors.glow}`}>
        <div>
          <div className="text-xs uppercase tracking-widest text-gray-400 mb-0.5">Tornado Risk</div>
          <div className={`text-2xl font-bold ${colors.text}`}>{prediction.risk_level}</div>
        </div>
        <Zap size={28} className={colors.text} />
      </div>

      {/* Probability bars */}
      <div className="bg-storm-800 rounded-lg p-3 space-y-3 border border-storm-600">
        <ProbBar label="Next 1 hour" value={prediction.prob_1h} color={colors.bar} />
        <ProbBar label="Next 2 hours" value={prediction.prob_2h} color={colors.bar} />
      </div>

      {/* EF estimate */}
      <div className="bg-storm-800 rounded-lg p-3 border border-storm-600 flex items-center gap-3">
        <Wind size={18} className="text-sky-400 shrink-0" />
        <div>
          <div className="text-xs text-gray-400">Estimated intensity if tornado occurs</div>
          <div className="text-white font-semibold">
            EF{Math.round(prediction.ef_estimate)} scale
            <span className="ml-2 text-xs text-gray-400 font-normal">
              ({efDescription(prediction.ef_estimate)})
            </span>
          </div>
        </div>
      </div>

      {/* Valid time */}
      <div className="flex items-center gap-2 text-xs text-gray-500">
        <Clock size={12} />
        <span>Valid {new Date(prediction.valid_time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })} UTC</span>
        <span className="ml-auto font-mono opacity-60">v:{prediction.model_version}</span>
      </div>
    </div>
  )
}

function efDescription(ef) {
  const ef0 = Math.round(ef)
  const desc = ['Minor', 'Moderate', 'Significant', 'Severe', 'Devastating', 'Incredible']
  return desc[Math.min(Math.max(ef0, 0), 5)] ?? 'Unknown'
}

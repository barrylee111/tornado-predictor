import { useState, useRef } from 'react'
import { Volume2, VolumeX, Loader } from 'lucide-react'
import { fetchTTS } from '../api/client'

// Bill Paxton–adjacent ElevenLabs voice: "Adam" (warm, authoritative American male)
const VOICE_ID = 'pNInz6obpgDQGcFmaJgB'

export function VoiceButton({ narrative, riskLevel, disabled }) {
  const [state, setState] = useState('idle') // idle | loading | playing | error
  const audioRef = useRef(null)

  const riskColors = {
    LOW:      'bg-emerald-800 hover:bg-emerald-700 border-emerald-600',
    ELEVATED: 'bg-yellow-800  hover:bg-yellow-700  border-yellow-600',
    HIGH:     'bg-orange-800  hover:bg-orange-700  border-orange-600',
    EXTREME:  'bg-red-800     hover:bg-red-700     border-red-600',
  }
  const colorClass = riskColors[riskLevel] ?? riskColors.LOW

  async function handleClick() {
    if (state === 'loading') return

    if (state === 'playing') {
      audioRef.current?.pause()
      setState('idle')
      return
    }

    setState('loading')
    try {
      const res = await fetchTTS(narrative, VOICE_ID)

      if (res.provider === 'elevenlabs' && res.audio_url) {
        // ElevenLabs: play base64 audio
        const audio = new Audio(res.audio_url)
        audioRef.current = audio
        audio.onended = () => setState('idle')
        audio.onerror = () => setState('error')
        await audio.play()
        setState('playing')
      } else {
        // Browser fallback: Web Speech API
        browserSpeak(narrative, () => setState('idle'))
        setState('playing')
      }
    } catch (e) {
      console.error('TTS error:', e)
      // Final fallback: always try Web Speech
      browserSpeak(narrative, () => setState('idle'))
      setState('playing')
    }
  }

  return (
    <button
      onClick={handleClick}
      disabled={disabled || !narrative}
      className={`
        w-full flex items-center justify-center gap-2.5 py-3 px-4 rounded-lg border
        font-semibold text-sm text-white transition-all duration-200
        disabled:opacity-40 disabled:cursor-not-allowed
        ${colorClass}
      `}
    >
      {state === 'loading' && <Loader size={16} className="animate-spin" />}
      {state === 'playing' && <VolumeX size={16} />}
      {(state === 'idle' || state === 'error') && <Volume2 size={16} />}
      <span>
        {state === 'loading' ? 'Getting report…'
          : state === 'playing' ? 'Stop'
          : 'Hear the forecast'}
      </span>
    </button>
  )
}

function browserSpeak(text, onEnd) {
  if (!window.speechSynthesis) { onEnd(); return }
  window.speechSynthesis.cancel()
  const utt = new SpeechSynthesisUtterance(text)
  utt.rate = 0.92
  utt.pitch = 0.88
  utt.volume = 1
  // Prefer a deep American male voice
  const voices = window.speechSynthesis.getVoices()
  const preferred = voices.find(v =>
    v.lang.startsWith('en') && v.name.toLowerCase().includes('male')
  ) ?? voices.find(v => v.lang.startsWith('en')) ?? null
  if (preferred) utt.voice = preferred
  utt.onend = onEnd
  utt.onerror = onEnd
  window.speechSynthesis.speak(utt)
}

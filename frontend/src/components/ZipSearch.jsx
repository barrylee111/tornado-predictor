import { useState, useEffect, useRef, useCallback } from 'react'
import { Search, MapPin, X } from 'lucide-react'
import { geocodeQuery, geocodeZip } from '../api/client'

const ZIP_RE = /^\d{5}$/

function useDebounce(value, delay) {
  const [debounced, setDebounced] = useState(value)
  useEffect(() => {
    const t = setTimeout(() => setDebounced(value), delay)
    return () => clearTimeout(t)
  }, [value, delay])
  return debounced
}

export function ZipSearch({ currentLabel, onSelect }) {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState([])
  const [open, setOpen] = useState(false)
  const [activeIdx, setActiveIdx] = useState(0)
  const [searching, setSearching] = useState(false)
  const inputRef = useRef(null)
  const debounced = useDebounce(query, 300)

  useEffect(() => {
    if (!debounced.trim()) {
      setResults([])
      return
    }
    let cancelled = false
    async function search() {
      setSearching(true)
      try {
        let found = []
        if (ZIP_RE.test(debounced.trim())) {
          // Exact zip — use Zippopotam for precise lookup first
          const exact = await geocodeZip(debounced.trim())
          if (exact) found = [exact]
        }
        // Always augment with Open-Meteo geocoding (catches city names + partial zips)
        const geo = await geocodeQuery(debounced)
        const ids = new Set(found.map(f => f.label))
        found = [...found, ...geo.filter(g => !ids.has(g.label))]
        if (!cancelled) {
          setResults(found.slice(0, 8))
          setActiveIdx(0)
          setOpen(found.length > 0)
        }
      } finally {
        if (!cancelled) setSearching(false)
      }
    }
    search()
    return () => { cancelled = true }
  }, [debounced])

  function select(loc) {
    setQuery('')
    setOpen(false)
    onSelect(loc)
  }

  function handleKey(e) {
    if (!open) return
    if (e.key === 'ArrowDown') { e.preventDefault(); setActiveIdx(i => Math.min(i + 1, results.length - 1)) }
    if (e.key === 'ArrowUp')   { e.preventDefault(); setActiveIdx(i => Math.max(i - 1, 0)) }
    if (e.key === 'Enter')     { e.preventDefault(); if (results[activeIdx]) select(results[activeIdx]) }
    if (e.key === 'Escape')    { setOpen(false); inputRef.current?.blur() }
  }

  return (
    <div className="relative w-full">
      <div className="flex items-center gap-2 bg-storm-800 border border-storm-600 rounded-lg px-3 py-2.5 focus-within:border-sky-500 transition-colors">
        {searching
          ? <div className="w-4 h-4 border-2 border-sky-400 border-t-transparent rounded-full animate-spin shrink-0" />
          : <Search size={16} className="text-gray-400 shrink-0" />
        }
        <input
          ref={inputRef}
          type="text"
          value={query}
          onChange={e => { setQuery(e.target.value); setOpen(true) }}
          onKeyDown={handleKey}
          onFocus={() => results.length && setOpen(true)}
          onBlur={() => setTimeout(() => setOpen(false), 150)}
          placeholder={currentLabel ?? 'Search city or ZIP code…'}
          className="flex-1 bg-transparent text-sm text-white placeholder-gray-500 outline-none"
        />
        {query && (
          <button onClick={() => { setQuery(''); setResults([]); setOpen(false) }}>
            <X size={14} className="text-gray-500 hover:text-gray-300" />
          </button>
        )}
      </div>

      {open && results.length > 0 && (
        <div className="zip-dropdown">
          {results.map((loc, i) => (
            <div
              key={loc.id ?? loc.label}
              className={`zip-dropdown-item ${i === activeIdx ? 'active' : ''}`}
              onMouseDown={() => select(loc)}
              onMouseEnter={() => setActiveIdx(i)}
            >
              <MapPin size={13} className="text-sky-400 shrink-0" />
              <span>{loc.label}</span>
              {loc.zip && <span className="ml-auto text-xs text-gray-500">{loc.zip}</span>}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

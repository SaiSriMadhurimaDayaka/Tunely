import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Activity, ArrowRight, ArrowUpRight, AudioLines, Bookmark, Check,
  ChevronDown, CircleDot, CloudRain, Compass, Disc3, Heart, History, Info, LoaderCircle,
  Eye, EyeOff, Music2, Pause, RefreshCw, Search, Sparkles, Sun, Waves, X, Zap,
} from 'lucide-react'
import { youtubeMusicDestination } from './youtubeMusic.js'

// Local API traffic uses Vite's same-origin proxy so Flask's SameSite session
// cookie works while the development server binds to IPv4 loopback.
const API = import.meta.env.VITE_API_BASE_URL || ''
const MOODS = [
  { value: 'calm', label: 'Calm', description: 'Peaceful and gentle', icon: Waves, color: 'calm' },
  { value: 'happy', label: 'Happy', description: 'Bright and uplifting', icon: Sun, color: 'happy' },
  { value: 'sad', label: 'Sad', description: 'Reflective and expressive', icon: CloudRain, color: 'sad' },
  { value: 'stressed', label: 'Stressed', description: 'Choose the response you want', icon: Activity, color: 'stressed' },
  { value: 'energetic', label: 'Energetic', description: 'Lively and dynamic', icon: Zap, color: 'energetic' },
  { value: 'neutral', label: 'Neutral', description: 'Balanced and familiar', icon: CircleDot, color: 'neutral' },
  { value: 'romantic', label: 'Romantic', description: 'Intimate and warm', icon: Heart, color: 'romantic' },
  { value: 'motivated', label: 'Motivated', description: 'Encouraging and powerful', icon: ArrowUpRight, color: 'motivated' },
  { value: 'nostalgic', label: 'Nostalgic', description: 'A familiar, memory-filled sound', icon: History, color: 'nostalgic' },
]
const GOALS = [
  { value: 'relax', label: 'Relax', description: 'Let the day soften', icon: 'relax' },
  { value: 'focus', label: 'Focus', description: 'Keep your mind in the zone', icon: 'focus' },
  { value: 'energize', label: 'Energize', description: 'Turn up your momentum', icon: 'energy' },
  { value: 'feel-good', label: 'Feel-good', description: 'Find a brighter sound', icon: 'discover' },
  { value: 'wind down', label: 'Wind down', description: 'Ease into a quieter pace', icon: 'wind' },
  { value: 'discover something new', label: 'Discover', description: 'Find a new favorite', icon: 'discover' },
]
const RATINGS = [
  { label: 'Loved it', icon: '♥' }, { label: 'Liked it', icon: '＋' },
  { label: 'It was okay', icon: '·' }, { label: "Didn't like it", icon: '−' },
]
const MOOD_LEVELS = [
  { value: 1, label: 'Mild' }, { value: 2, label: 'Moderate' }, { value: 3, label: 'Intense' },
]
const FALLBACK_DISCOVERY_LANGUAGES = [
  'Telugu', 'Hindi', 'Tamil', 'Malayalam', 'Kannada', 'English',
].map(language => ({ language, search_supported: true, catalog_song_count: null }))
const INTENSITY_HINTS = {
  happy: ['Light, pleasant feel-good music', 'Cheerful, uplifting music', 'Highly energetic, celebratory music'],
  sad: ['Gentle, reflective music', 'Emotional, expressive music', 'Deeply expressive or cathartic music'],
  calm: ['Soft, unobtrusive music', 'Peaceful, immersive music', 'Deeply atmospheric music'],
  energetic: ['Lightly upbeat music', 'Lively music with a stronger rhythm', 'High-energy, powerful music'],
  stressed: ['Gentle, low-intensity music', 'A balance of calmness and engagement', 'A stronger response shaped by your goal'],
  romantic: ['Subtle and tender', 'Warm and intimate', 'Rich and expressive'],
  neutral: ['Light and unobtrusive', 'Balanced and engaging', 'Pronounced character'],
  motivated: ['A gentle lift', 'Encouraging and lively', 'Bold and powerful'],
  nostalgic: ['A soft familiar touch', 'Warm and evocative', 'Deeply expressive and memory-filled'],
}
const NAV = [
  { id: 'discover', label: 'Discover', icon: Compass },
  { id: 'saved', label: 'Saved tracks', icon: Bookmark },
  { id: 'history', label: 'Your history', icon: History },
  { id: 'insights', label: 'AI insights', icon: Activity },
]

function getSessionId() {
  let id = localStorage.getItem('tunely-session-id')
  if (!id) {
    id = globalThis.crypto?.randomUUID?.().replaceAll('-', '') || `${Date.now()}${Math.random().toString(36).slice(2)}`
    localStorage.setItem('tunely-session-id', id)
  }
  return id
}

async function api(path, options = {}) {
  const controller = new AbortController()
  const timeout = window.setTimeout(() => controller.abort(), 25000)
  try {
    const response = await fetch(`${API}${path}`, {
      ...options,
      signal: options.signal || controller.signal,
      credentials: 'include',
      headers: {
        'Content-Type': 'application/json',
        'X-Session-ID': getSessionId(),
        ...(options.headers || {}),
      },
    })
    const payload = await response.json().catch(() => ({}))
    if (!response.ok) {
      const serverMessage = payload?.error?.message
      if (serverMessage) throw new Error(serverMessage)
      if (response.status >= 500) {
        throw new Error(`Tunely could not reach Flask through Vite (HTTP ${response.status}). Start Flask on port 5002, then restart Vite so its proxy reconnects.`)
      }
      throw new Error(`Request failed (${response.status})`)
    }
    return payload.data
  } catch (err) {
    if (err.name === 'AbortError') throw new Error('The request took too long. Check that Flask is running and try again.')
    throw err
  } finally {
    window.clearTimeout(timeout)
  }
}

function prettyDuration(ms) {
  if (!ms) return '—'
  const total = Math.floor(ms / 1000)
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, '0')}`
}

function prettyFeature(value, feature) {
  if (value == null) return null
  if (feature === 'tempo') return `${Math.round(value)} BPM`
  if (feature === 'loudness') return `${value.toFixed(1)} dB`
  return `${Math.round(value * 100)}% ${feature}`
}

function MoodSelector({ value, onChange }) {
  return <div className="mood-grid" role="group" aria-label="Choose your current mood">
    {MOODS.map(mood => { const Icon = mood.icon; return <button key={mood.value} type="button" onClick={() => onChange(mood.value)}
      className={`mood-option mood-${mood.color} ${value === mood.value ? 'selected' : ''}`} aria-pressed={value === mood.value}>
      <span className="mood-symbol" aria-hidden="true"><Icon size={19} strokeWidth={2} /></span><span className="mood-option-copy"><strong>{mood.label}</strong><small>{mood.description}</small></span>
      {value === mood.value && <Check size={14} className="mood-check" />}
    </button>})}
  </div>
}

function GoalSelector({ value, onChange }) {
  return <div className="goal-grid" role="group" aria-label="Choose your listening goal">
    {GOALS.map((goal, index) => <button key={goal.value} type="button" onClick={() => onChange(goal.value)}
      className={`goal-option ${value === goal.value ? 'selected' : ''}`} aria-pressed={value === goal.value}>
      <span className={`goal-icon goal-icon-${index}`}>
        {index === 0 ? <Disc3 size={19} /> : index === 1 ? <Waves size={19} /> : index === 2 ? <AudioLines size={19} /> : index === 3 ? <Pause size={17} /> : <Sparkles size={18} />}
      </span>
      <span className="goal-copy"><strong>{goal.label}</strong><small>{goal.description}</small></span>
      <span className="goal-radio">{value === goal.value && <span />}</span>
    </button>)}
    <button className={`goal-clear ${value === '' ? 'selected' : ''}`} type="button" aria-pressed={value === ''} onClick={() => onChange('')}>No listening goal</button>
  </div>
}

function Artwork({ song, index = 0, compact = false }) {
  const palette = ['art-purple', 'art-blue', 'art-pink', 'art-teal', 'art-amber', 'art-indigo']
  const [failedImage, setFailedImage] = useState('')
  const title = song.song_name || song.title || '♪'
  const initials = title.split(/\s+/).slice(0, 2).map(word => word[0]).join('').toUpperCase()
  if (song.cover_image && failedImage !== song.cover_image) return <div className={`artwork ${compact ? 'artwork-small' : ''}`}><img src={song.cover_image} alt={`${title} artwork`} loading="lazy" onError={() => setFailedImage(song.cover_image)} /></div>
  return <div className={`artwork artwork-fallback ${palette[index % palette.length]} ${compact ? 'artwork-small' : ''}`} aria-label="Cover image unavailable">
    <div className="art-orbit orbit-one" /><div className="art-orbit orbit-two" />
    <span className="art-initials">{initials}</span><span className="art-mark"><AudioLines size={compact ? 15 : 21} /></span>
    {!compact && <span className="art-unavailable">COVER UNAVAILABLE</span>}
  </div>
}

function SongCard({ song, index, saved, feedback, onFavorite, onFeedback, feedbackBusy, allowFeedback = true }) {
  const title = song.song_name || song.title || 'Untitled recording'
  const listeningUrl = youtubeMusicDestination(song)
  const features = song.characteristics || {}
  const chips = ['energy', 'danceability', 'acousticness'].filter(key => features[key] != null).slice(0, 2)
  const genreText = song.genre_source === 'reviewed_dataset'
    ? 'Reviewed genre: ' + (song.genre || song.music_categories?.join(', ') || 'unclassified')
    : song.ai_genre
    ? `Mood-guided genre: ${song.ai_genre}`
    : song.genre || song.track_genre
    ? `Genre: ${song.genre || song.track_genre}`
    : song.genres?.length
      ? `Genre estimate from video text: ${song.genres.join(', ')}`
      : song.genre_candidates?.length
        ? `Mood-based style suggestions: ${song.genre_candidates.slice(0, 2).join(', ')}`
        : 'Genre estimate unavailable'
  return <article className={`song-card ${listeningUrl ? 'song-card-linkable' : ''}`}>
    {listeningUrl && <a className="song-card-click-target" href={listeningUrl} target="_blank" rel="noopener noreferrer"
      aria-label={`Open ${title}${song.artist ? ` by ${song.artist}` : ''} in YouTube Music in a new tab`} />}
    <div className="card-art-wrap">
      <Artwork song={song} index={index} />
      <span className="card-score"><Sparkles size={12} /> {song.youtube_video_id || song.itunes_track_id ? 'AI discovery match' : song.strategy === 'exploration' || song.score_method === 'context-seeded catalog exploration' ? 'exploration' : 'feature match'}</span>
      <button className={`favorite-button ${saved ? 'is-saved' : ''}`} type="button" aria-label={saved ? 'Remove from saved tracks' : 'Save track'} onClick={() => onFavorite(song)}>
        <Bookmark size={17} fill={saved ? 'currentColor' : 'none'} />
      </button>
    </div>
    <div className="song-card-body">
      <div className="song-title-row"><div className="song-title-group"><h3 title={title}>{title}</h3><p title={song.artist || song.channel_title}>{song.artist || (song.channel_title ? `Channel: ${song.channel_title}` : 'Artist unavailable')}</p></div><span className="duration">{prettyDuration(song.duration_ms)}</span></div>
      <div className="song-metadata"><span>{genreText}</span>{song.album_name && <><i /><span>{song.album_name}</span></>}{song.release_year && <><i /><span>{song.release_year}</span></>}{song.language && <><i /><span>{song.language_verified ? song.language : `Language estimate: ${song.language}`}</span></>}</div>
      {song.music_categories?.length > 0 && <div className="music-category-tags" aria-label="Classified music categories">{song.music_categories.map(label => <span key={label}>{label}</span>)}</div>}
      {song.emotion_labels?.length > 0 && <div className="emotion-tags" aria-label={`Lyric emotion signals: ${song.emotion_labels.join(', ')}`} title={`Lyrics-based signals · ${song.emotion_model_version || 'classifier'}`}>{song.emotion_labels.map(label => <span key={label}>{label}</span>)}</div>}
      {song.category_explanation && song.music_categories?.length > 0 && <p className="category-explanation">Category evidence: {song.category_explanation}</p>}
      <div className="song-context-tags">{song.selected_mood && <span>Requested mood: {song.selected_mood}{song.mood_intensity ? ` · ${song.mood_intensity}` : ''}</span>}{song.mood_verified && song.mood && <span>Verified mood tag: {song.mood}</span>}{song.language_evidence && <span title={song.language_evidence}>{song.language_verified ? 'Verified language' : 'Language estimate'}</span>}{song.mood_evidence?.length > 0 && <span>Mood text signal: {song.mood_evidence.join(', ')}</span>}{song.intensity_evidence?.length > 0 && <span>Mood match signal: {song.intensity_evidence.join(', ')}</span>}</div>
      {chips.length > 0 && <div className="feature-chips">{chips.map(key => <span key={key}>{prettyFeature(features[key], key)}</span>)}</div>}
      <p className="recommendation-reason"><Sparkles size={13} />{song.explanation || `Suggested for your ${song.selected_mood || 'selected'} mood based on the available song metadata.`}</p>
      {listeningUrl && <span className="song-card-open-hint"><ArrowUpRight size={12} />Open in YouTube Music</span>}
      <div className="card-divider" />
      {feedback ? <div className="feedback-confirm"><Check size={15} /> Feedback saved: <strong>{feedback}</strong></div> : (allowFeedback && song.recommendation_id ? <div className="feedback-row" aria-label="Rate this recommendation">
        <span>How does it feel?</span><div>{RATINGS.map(rating => <button key={rating.label} type="button" disabled={feedbackBusy} title={rating.label} aria-label={rating.label}
          className={`rating-button ${rating.label === "Didn't like it" ? 'rating-down' : ''}`} onClick={() => onFeedback(song, rating.label)}><span>{rating.icon}</span></button>)}</div>
      </div> : null)}
    </div>
  </article>
}

function StatCard({ icon: Icon, label, value, hint, tone = '' }) {
  return <div className="stat-card"><span className={`stat-icon ${tone}`}><Icon size={18} /></span><span className="stat-label">{label}</span><strong>{value}</strong><small>{hint}</small></div>
}

export default function App() {
  const [page, setPage] = useState('discover')
  const [mood, setMood] = useState('neutral')
  const [moodLevel, setMoodLevel] = useState(2)
  const [goal, setGoal] = useState('')
  const [preference, setPreference] = useState('')
  const [genre, setGenre] = useState('')
  const [language, setLanguage] = useState('')
  const [customEnergy, setCustomEnergy] = useState(false)
  const [energy, setEnergy] = useState(55)
  const [songs, setSongs] = useState([])
  const [categoryGuidance, setCategoryGuidance] = useState([])
  const [languages, setLanguages] = useState([])
  const [languageLoadError, setLanguageLoadError] = useState('')
  const [favorites, setFavorites] = useState([])
  const [history, setHistory] = useState([])
  const [insights, setInsights] = useState(null)
  const [loading, setLoading] = useState(false)
  const [feedbackBusy, setFeedbackBusy] = useState('')
  const [error, setError] = useState('')
  const [discoveryFailed, setDiscoveryFailed] = useState(false)
  const [notice, setNotice] = useState('')
  const [hasDiscovered, setHasDiscovered] = useState(false)
  const [apiStatus, setApiStatus] = useState('checking')
  const [catalogInfo, setCatalogInfo] = useState(null)
  const [currentUser, setCurrentUser] = useState(null)
  const [authBusy, setAuthBusy] = useState(false)
  const [authError, setAuthError] = useState('')
  const recommendationInFlight = useRef(false)

  const favoriteIds = useMemo(() => new Set(favorites.map(song => song.song_id)), [favorites])
  const currentPage = NAV.find(item => item.id === page)

  const refreshFavorites = useCallback(async () => {
    const [catalogFavorites, youtubeFavorites, itunesFavorites] = await Promise.all([
      api('/api/favorites'), api('/api/youtube-favorites'), api('/api/itunes-favorites'),
    ])
    setFavorites([...catalogFavorites, ...youtubeFavorites, ...itunesFavorites])
  }, [])

  useEffect(() => {
    api('/api/health').then(data => {
      setCatalogInfo(data)
      setApiStatus(data.catalog_ready ? 'connected' : 'empty')
    }).catch(() => setApiStatus('offline'))
    const languageRequest = api('/api/discovery/languages').then(data => {
      setLanguages(data || [])
      setLanguageLoadError('')
      return data || []
    }).catch(() => {
      setLanguages(FALLBACK_DISCOVERY_LANGUAGES)
      setLanguageLoadError('Showing the built-in language list because Flask did not provide its language options. Restart Flask before searching.')
      return FALLBACK_DISCOVERY_LANGUAGES
    })
    api('/api/auth/me').then(data => {
      setCurrentUser(data.user || null)
      if (data.user) api('/api/preferences').then(async preferences => {
        if (preferences.preferred_language) {
          const supportedLanguages = await languageRequest
          const canSearch = supportedLanguages.some(item => item.language.toLowerCase() === preferences.preferred_language.toLowerCase() && item.search_supported)
          if (canSearch) setLanguage(preferences.preferred_language)
          else setLanguage('')
        }
        if (preferences.favorite_genres?.[0]) setGenre(preferences.favorite_genres[0])
      }).catch(() => {})
    }).catch(() => {})
    refreshFavorites().catch(() => {})
  }, [refreshFavorites])

  const discover = async ({ refresh = false } = {}) => {
    if (recommendationInFlight.current) return
    const selectedIntensity = MOOD_LEVELS.find(level => level.value === moodLevel)
    if (!language || !mood || !selectedIntensity) {
      const missing = [!language && 'language', !mood && 'mood', !selectedIntensity && 'mood intensity'].filter(Boolean)
      setError(`Choose your ${missing.join(', ')} before finding music.`)
      setNotice('')
      return
    }
    recommendationInFlight.current = true
    const excludeTrackIds = refresh ? songs.map(song => String(song.itunes_track_id || '')).filter(Boolean) : []
    setHasDiscovered(true)
    setSongs([])
    setCategoryGuidance([])
    setLoading(true); setError(''); setNotice(''); setDiscoveryFailed(false)
    try {
      const intensity = selectedIntensity.label
      const result = await api('/api/mood-recommendations', { method: 'POST', body: JSON.stringify({ language, mood, intensity, goal, preference, refresh, exclude_track_ids: excludeTrackIds, limit: 8 }) })
      if (!['itunes', 'reviewed_dataset'].includes(result.api_source)) {
        throw new Error('The running Flask server is outdated and is still using the YouTube discovery endpoint. Stop it, restart with “python -m backend.app”, then retry.')
      }
      if (!Array.isArray(result.recommendations)) {
        throw new Error(result.message || 'The music catalog returned an invalid recommendation response. Please retry.')
      }
      setSongs(result.recommendations)
      setCategoryGuidance(result.inferred_category_labels || [])
      setHasDiscovered(true)
      if (result.error_code) { setDiscoveryFailed(true); setError(result.message || 'The music catalog request failed. Please retry.') }
      else if (!result.recommendations.length) setError(result.message || 'No matching songs found.')
      else setNotice(result.message || `Found ${result.recommendations.length} music matches.`)
    } catch (err) {
      setSongs([])
      setDiscoveryFailed(true)
      setError(/\b404\b|route.*not found/i.test(err.message)
        ? 'This Flask process does not have the recommendation route loaded yet. Restart it with “python -m backend.app”, then try again.'
        : err.message)
    }
    finally { recommendationInFlight.current = false; setLoading(false) }
  }

  const showPage = async (nextPage) => {
    setPage(nextPage); setError(''); setNotice('')
    if (nextPage === 'saved') {
      setLoading(true)
      try { await refreshFavorites() } catch (err) { setError(err.message) }
      finally { setLoading(false) }
    }
    if (nextPage === 'history') {
      setLoading(true)
      try { setHistory(await api('/api/history?limit=50')) } catch (err) { setError(err.message) }
      finally { setLoading(false) }
    }
    if (nextPage === 'insights') {
      setLoading(true)
      try { setInsights(await api('/api/insights')) } catch (err) { setError(err.message) }
      finally { setLoading(false) }
    }
  }

  const toggleFavorite = async (song) => {
    setError(''); setNotice('')
    const id = song.song_id
    const dynamicItunesResult = Boolean(song.itunes_track_id)
    const dynamicYoutubeResult = Boolean(song.youtube_video_id)
    const route = dynamicItunesResult ? '/api/itunes-favorites' : dynamicYoutubeResult ? '/api/youtube-favorites' : '/api/favorites'
    const isSaved = favoriteIds.has(id)
    try {
      if (isSaved) {
        const identifier = dynamicItunesResult ? song.itunes_track_id : dynamicYoutubeResult ? song.youtube_video_id : id
        await api(`${route}/${encodeURIComponent(identifier)}`, { method: 'DELETE' })
        setFavorites(current => current.filter(item => item.song_id !== id))
        setNotice('Removed from your saved tracks.')
      } else {
        const payload = dynamicItunesResult ? { track_id: song.itunes_track_id } : dynamicYoutubeResult ? { video_id: song.youtube_video_id } : { song_id: id }
        await api(route, { method: 'POST', body: JSON.stringify(payload) })
        await refreshFavorites()
        setNotice('Saved to your tracks.')
      }
    } catch (err) { setError(err.message) }
  }

  const submitFeedback = async (song, rating) => {
    if (feedbackBusy) return
    setFeedbackBusy(song.recommendation_id); setError('')
    try {
      const feedbackRoute = song.itunes_track_id ? '/api/itunes-feedback' : song.youtube_video_id ? '/api/youtube-feedback' : '/api/feedback'
      await api(feedbackRoute, { method: 'POST', body: JSON.stringify({ recommendation_id: song.recommendation_id, rating }) })
      setSongs(current => current.map(item => item.recommendation_id === song.recommendation_id ? { ...item, feedback: rating } : item))
      setNotice(song.youtube_video_id || song.itunes_track_id
        ? 'Feedback saved. External catalog search rankings are not trained or changed by these ratings yet.'
        : 'Thanks — your feedback will help tune future recommendations.')
    } catch (err) { setError(err.message) }
    finally { setFeedbackBusy('') }
  }

  const submitAuth = async (mode, fields) => {
    setAuthBusy(true); setAuthError('')
    try {
      const result = await api(`/api/auth/${mode}`, { method: 'POST', body: JSON.stringify(fields) })
      setCurrentUser(result.user)
      if (mode === 'login') {
        const preferences = await api('/api/preferences')
        const savedLanguage = languages.find(item => item.language.toLowerCase() === (preferences.preferred_language || '').toLowerCase() && item.song_count > 0)
        setLanguage(savedLanguage ? preferences.preferred_language : '')
        setGenre(preferences.favorite_genres?.[0] || '')
      }
      setPage('discover'); setNotice(`Welcome${result.user.name ? `, ${result.user.name}` : ''}. Your account is ready.`)
      refreshFavorites().catch(() => {})
    } catch (err) { setAuthError(err.message) }
    finally { setAuthBusy(false) }
  }

  const logout = async () => {
    try { await api('/api/auth/logout', { method: 'POST' }) }
    catch (err) { setError(err.message); return }
    setCurrentUser(null); setLanguage(''); setGenre(''); setPage('discover')
    setFavorites([]); setHistory([]); setNotice('You are signed out. This browser session is now separate.')
  }

  const selectLanguage = async (value) => {
    setLanguage(value)
    setError('')
    if (currentUser) {
      try { await api('/api/preferences', { method: 'PUT', body: JSON.stringify({ preferred_language: value === 'Unknown' ? '' : value, favorite_genres: genre ? [genre] : [] }) }) }
      catch (err) { setError(err.message) }
    }
  }

  const selectGenre = async (value) => {
    setGenre(value)
    if (currentUser) {
      try { await api('/api/preferences', { method: 'PUT', body: JSON.stringify({ preferred_language: language === 'Unknown' ? '' : language, favorite_genres: value ? [value] : [] }) }) }
      catch (err) { setError(err.message) }
    }
  }

  const renderCards = (items, mode = 'recommendations') => <div className="song-grid">
    {items.map((song, index) => {
      const record = mode === 'history' ? {
        ...song,
        recommendation_id: song.recommendation_id,
        explanation: song.rating ? `You rated this ${song.rating}.` : `Recommended for ${song.context?.goal || 'your discovery session'}.`,
        selected_mood: song.context?.mood ? MOODS.find(item => item.value === song.context.mood)?.label || song.context.mood : null,
        characteristics: {},
      } : song
      const recordId = record.song_id
      return <SongCard key={`${record.recommendation_id || recordId}-${index}`} song={record} index={index}
        saved={favoriteIds.has(recordId)} feedback={record.feedback || record.rating}
        allowFeedback={mode !== 'saved'} feedbackBusy={feedbackBusy === record.recommendation_id}
        onFavorite={toggleFavorite} onFeedback={submitFeedback} />
    })}
  </div>

  return <div className="app-shell">
    <aside className="sidebar">
      <a className="brand" href="#discover" onClick={event => { event.preventDefault(); showPage('discover') }}>
        <span className="brand-mark"><AudioLines size={19} strokeWidth={2.5} /></span><span>Tunely</span><span className="brand-dot">.</span>
      </a>
      <div className="sidebar-label">YOUR SPACE</div>
      <nav className="primary-nav" aria-label="Main navigation">
        {NAV.map(({ id, label, icon: Icon }) => <button type="button" key={id} aria-label={label} onClick={() => showPage(id)} className={`nav-link ${page === id ? 'active' : ''}`}>
          <Icon size={18} strokeWidth={1.8} /><span>{label}</span>{id === 'saved' && favorites.length > 0 && <b>{favorites.length}</b>}
        </button>)}
      </nav>
      <div className="sidebar-feature">
        <div className="sidebar-wave"><span /><span /><span /><span /><span /><span /><span /></div>
        <strong>Soundtrack your moment</strong><p>Your mood and your listening goal are yours to choose.</p>
        <button type="button" onClick={() => { showPage('discover'); requestAnimationFrame(() => document.getElementById('selection-panel')?.scrollIntoView({ behavior: 'smooth', block: 'start' })) }}>Start a session <ArrowRight size={14} /></button>
      </div>
      <div className="sidebar-bottom"><span className={`status-dot status-${apiStatus}`} /><span>{apiStatus === 'connected' ? 'Catalog connected' : apiStatus === 'checking' ? 'Connecting…' : apiStatus === 'empty' ? 'Catalog is empty' : 'API unavailable'}</span><span className="sidebar-version">v0.4</span></div>
    </aside>

    <main className="main-area">
      <header className="topbar">
        <div className="breadcrumbs"><span>Workspace</span><span className="crumb-slash">/</span><strong>{currentPage?.label || (page === 'register' ? 'Create account' : page === 'login' ? 'Sign in' : 'Discover')}</strong></div>
        <div className="topbar-right">{currentUser ? <><span className="session-pill"><span className="session-dot" />{currentUser.name}</span><button className="auth-top-button" type="button" onClick={logout}>Sign out</button></> : <><span className="session-pill"><span className="session-dot" />Guest session</span><button className="auth-top-button" type="button" onClick={() => { setAuthError(''); setPage('login') }}>Sign in</button></>}<div className="avatar">{currentUser?.name?.[0]?.toUpperCase() || 'T'}</div></div>
      </header>

      <div className="page-content">
        {error && <div className="alert alert-error" role="alert"><Info size={17} /><span>{error}</span><button type="button" aria-label="Dismiss error" onClick={() => setError('')}><X size={16} /></button></div>}
        {notice && <div className="alert alert-success" role="status"><Check size={16} /><span>{notice}</span><button type="button" aria-label="Dismiss message" onClick={() => setNotice('')}><X size={16} /></button></div>}

        {(page === 'login' || page === 'register') && <AuthPage mode={page} onModeChange={mode => { setPage(mode); setAuthError('') }} onSubmit={submitAuth} busy={authBusy} error={authError} />}

        {page === 'discover' && <>
          <section className="selection-panel" id="selection-panel">
            <div className="section-heading"><div><span className="section-kicker">START WITH YOU</span><h2>Set the scene</h2></div><span className="step-count"><span>01</span> / 02</span></div>
            <div className="preferences-row">
              <label className="genre-field"><span>Song language <small>REQUIRED</small></span><div className="input-wrap"><Search size={15} /><select className="filter-select" value={language} onChange={event => selectLanguage(event.target.value)} aria-label="Choose the language to search for" required><option value="">Choose a language</option>{languages.filter(item => item.search_supported).map(item => <option key={item.language} value={item.language}>{item.catalog_song_count == null ? item.language : item.language + ' · ' + item.catalog_song_count + ' songs'}</option>)}</select></div><small className="language-note">Recommendations use the Language column from your reviewed catalog.</small>{languageLoadError && <small className="language-load-error" role="alert">{languageLoadError}</small>}{!language && !languageLoadError && <small className="language-validation" role="status">Choose a language before finding music.</small>}</label>
            </div>
            <div className="form-divider" />
            <div className="form-section"><div className="form-label-row"><div><h3>How are you feeling right now?</h3><p>Your mood helps personalize your recommendations. You’re always in control.</p></div><span className="optional-tag">YOUR CALL</span></div><MoodSelector value={mood} onChange={value => { setMood(value); setError('') }} />
              <div className="mood-intensity"><div className="mood-intensity-copy"><strong>How strongly should the music reflect this mood?</strong><small>{INTENSITY_HINTS[mood]?.[moodLevel - 1] || 'Choose how pronounced you want this mood in the music.'}</small></div><div className="mood-levels" role="group" aria-label="Mood intensity level">{MOOD_LEVELS.map(level => <button key={level.value} type="button" aria-pressed={moodLevel === level.value} aria-label={`Mood intensity: ${level.label}`} className={`mood-level ${moodLevel === level.value ? 'selected' : ''}`} onClick={() => { setMoodLevel(level.value); setError('') }}><span className="level-dot">{moodLevel === level.value && <i />}</span><span>{level.label}</span></button>)}</div></div>
            </div>
            <div className="form-divider" />
            <div className="form-section"><div className="form-label-row"><div><h3>What would you like to listen to?</h3><p>Describe the sound or theme in your own words. This text helps the genre model rank your reviewed songs.</p></div><span className="optional-tag">OPTIONAL</span></div><label className="preference-input-label" htmlFor="listening-preference">Your listening preference</label><input id="listening-preference" className="preference-input" type="text" maxLength={500} value={preference} onChange={event => setPreference(event.target.value)} placeholder="e.g. heartbreak songs, peaceful devotional music, powerful hero elevation" /><div className="goal-subheading">Or choose a quick listening goal</div><GoalSelector value={goal} onChange={setGoal} /></div>
            <div className="energy-preference"><div className="catalog-limitation"><Info size={15} /><span>Tunely ranks songs from your reviewed catalog using its language and genre labels, your mood, intensity, and listening goal.</span></div></div>
            <div className="selection-footer"><p><span className="privacy-dot" />Your choices stay in this browser session.</p><button className="primary-button" type="button" onClick={() => discover()} disabled={loading}>
              {loading ? <><LoaderCircle size={17} className="spin" />Finding your sound…</> : <>Find My Music <ArrowRight size={17} /></>}
            </button></div>
          </section>

          <section className="how-section"><div className="how-heading"><div><span className="section-kicker">A BETTER KIND OF DISCOVERY</span><h2>Music that meets you here.</h2></div><p>No assumptions. Just thoughtful suggestions shaped by the sound you want and what you tell us you like.</p></div>
            <div className="how-grid"><div className="how-card"><span className="how-number">01</span><span className="how-icon"><Compass size={20} /></span><h3>You set the direction</h3><p>Choose your language, mood, and how strongly you want the music to reflect it.</p></div><div className="how-card"><span className="how-number">02</span><span className="how-icon"><AudioLines size={20} /></span><h3>Genres are chosen for you</h3><p>Tunely selects mood-aligned genres and ranks available catalog songs using language evidence, metadata, and mood signals.</p></div><div className="how-card"><span className="how-number">03</span><span className="how-icon"><Heart size={20} /></span><h3>Your feedback teaches</h3><p>Ratings are stored for your session. Catalog language estimates are not treated as verified track metadata.</p></div></div>
          </section>

          {hasDiscovered && <section className="results-section" id="results">
            <div className="results-heading"><div><span className="section-kicker">MADE FOR THIS MOMENT</span><h2>Your picks <span className="count-badge">{songs.length}</span></h2><p>Ranked using language evidence, mood and intensity context, genre suitability, and catalog metadata. Popularity is not used.</p></div>
              <div className="results-controls"><button className="text-button" onClick={() => document.getElementById('selection-panel')?.scrollIntoView({ behavior: 'smooth' })} type="button"><ChevronDown size={15} />Change choices</button><button className="outline-button" onClick={() => discover({ refresh: true })} type="button" disabled={loading}><RefreshCw size={15} className={loading ? 'spin' : ''} /> Refresh</button></div></div>
            {loading ? <div className="loading-state"><LoaderCircle size={25} className="spin" /><span>Finding songs for your language, mood, and listening preference…</span></div> : songs.length ? <>{categoryGuidance.length > 0 && <p className="category-guidance">Your mood and listening choice prioritize {categoryGuidance.join(' · ')} from your reviewed catalog.</p>}{renderCards(songs)}</> : <EmptyState icon={Music2} title="No confident matches yet" text={error || 'We could not find songs for this selection. Try another language or adjust your choices.'} action={discoveryFailed ? 'Retry search' : 'Adjust your choices'} onAction={discoveryFailed ? discover : () => document.querySelector('.selection-panel')?.scrollIntoView({ behavior: 'smooth' })} />}
          </section>}
          <footer className="page-footer"><span>© 2026 Tunely</span><span>Made for discovery, not diagnosis.</span><span>{catalogInfo?.song_count ?? 0} catalog tracks · recommendations shown in Tunely</span></footer>
        </>}

        {page === 'saved' && <section className="library-page"><PageTitle eyebrow="YOUR COLLECTION" title="Saved tracks" description="The songs you want to come back to." icon={Bookmark} /><div className="library-summary"><span className="summary-icon"><Bookmark size={18} /></span><span><strong>{favorites.length} saved {favorites.length === 1 ? 'track' : 'tracks'}</strong><small>Kept in this browser session</small></span></div>
          {loading ? <Loading /> : favorites.length ? renderCards(favorites, 'saved') : <EmptyState icon={Bookmark} title="Your collection starts here" text="Save a song from your recommendations and it will show up here." action="Find some music" onAction={() => showPage('discover')} />}
          <footer className="page-footer"><span>© 2026 Tunely</span><span>Saved tracks are linked to your account when signed in.</span></footer>
        </section>}

        {page === 'history' && <section className="library-page"><PageTitle eyebrow="YOUR LISTENING JOURNEY" title="Your history" description="Every recommendation session, with your ratings alongside." icon={History} />
          {loading ? <Loading /> : history.length ? <><div className="history-summary"><span><strong>{history.length}</strong> recent recommendations</span><span>Newest first</span></div>{renderCards(history, 'history')}</> : <EmptyState icon={History} title="Nothing here yet" text="Your recommendations will appear here as you explore." action="Start discovering" onAction={() => showPage('discover')} />}
          <footer className="page-footer"><span>© 2026 Tunely</span><span>History is linked to your account when signed in.</span></footer>
        </section>}

        {page === 'insights' && <section className="library-page"><PageTitle eyebrow="YOUR TUNELY PROFILE" title="AI insights" description="A transparent look at what your feedback is teaching the recommender." icon={Activity} />
          {loading ? <Loading /> : insights && <>
            <div className="insight-hero"><div className="insight-orb"><AudioLines size={28} /></div><div><span className="section-kicker">LEARNING FROM REAL FEEDBACK</span><h2>{insights.feedback_and_q_learning.feedback_count > 0 ? "Your choices are shaping the next one." : "The model is starting from zero."}</h2><p>{insights.feedback_and_q_learning.feedback_count > 0 ? `You have shared ${insights.feedback_and_q_learning.feedback_count} ${insights.feedback_and_q_learning.feedback_count === 1 ? "rating" : "ratings"} so far.` : "Nothing is pre-trained on personal feedback. Rate recommendations to begin."}</p></div><div className="insight-glow" /></div>
            <div className="stats-grid"><StatCard icon={Heart} label="Feedback given" value={insights.feedback_and_q_learning.feedback_count} hint="Real ratings from your account or session" tone="stat-pink" /><StatCard icon={Compass} label="Goals explored" value={insights.selected_goals?.length || 0} hint="Distinct listening goals" tone="stat-purple" /><StatCard icon={Activity} label="Learned values" value={insights.feedback_and_q_learning.q_values?.length || 0} hint="State and strategy pairs" tone="stat-blue" /></div>
            <div className="insights-columns"><div className="insight-panel"><div className="panel-title"><div><span className="section-kicker">YOUR SESSIONS</span><h3>Goals you've explored</h3></div><Compass size={17} /></div>{insights.selected_goals?.length ? <div className="goal-insight-list">{insights.selected_goals.map((item, i) => <div key={item.goal || i} className="goal-insight"><span className={`goal-insight-mark mark-${i % 4}`}><Music2 size={14} /></span><span>{item.goal || 'Unspecified goal'}</span><b>{item.recommendations}</b></div>)}</div> : <p className="empty-inline">No recommendation sessions recorded yet.</p>}<div className="panel-subhead">Mood intensity you selected</div>{insights.mood_levels?.length ? <div className="goal-insight-list">{insights.mood_levels.map((item, i) => <div key={`${item.mood}-${item.level}-${i}`} className="goal-insight"><span className={`goal-insight-mark mark-${i % 4}`}><Activity size={13} /></span><span>{item.mood || 'Unspecified'} · {MOOD_LEVELS.find(level => level.value === item.level)?.label || `level ${item.level ?? '—'}`}</span><b>{item.selections}</b></div>)}</div> : <p className="empty-inline">No mood levels recorded yet.</p>}</div>
              <div className="insight-panel"><div className="panel-title"><div><span className="section-kicker">POSITIVE FEEDBACK</span><h3>Characteristics in liked tracks</h3></div><Heart size={17} /></div>{insights.positive_feedback_characteristics?.rated_songs ? <><p className="panel-caption">Averages from {insights.positive_feedback_characteristics.rated_songs} positively rated tracks.</p><FeatureBars features={insights.positive_feedback_characteristics.average_audio_features_on_positive_feedback} /></> : <p className="empty-inline">No positive feedback yet. Rate a few tracks to build this view.</p>}</div></div>
            <div className="model-note"><Info size={17} /><p><strong>What these insights mean.</strong> Tunely learns which recommendation strategy works better for your selected mood and goal. Small samples are early signals, not statistically reliable conclusions. The system does not diagnose emotions or claim music changes how you feel.</p></div>
          </>}
          <footer className="page-footer"><span>© 2026 Tunely</span><span>Insights are based only on this session's stored interactions.</span></footer>
        </section>}
      </div>
    </main>

  </div>
}

function PageTitle({ eyebrow, title, description, icon: Icon }) {
  return <div className="page-title"><div className="page-title-icon"><Icon size={21} /></div><div><span className="section-kicker">{eyebrow}</span><h1>{title}</h1><p>{description}</p></div></div>
}

function EmptyState({ icon: Icon, title, text, action, onAction }) {
  return <div className="empty-state"><div className="empty-icon"><Icon size={25} /></div><h3>{title}</h3><p>{text}</p><button className="outline-button" onClick={onAction} type="button">{action} <ArrowRight size={15} /></button></div>
}

function Loading() {
  return <div className="loading-state"><LoaderCircle size={25} className="spin" /><span>Loading your listening space…</span></div>
}

function FeatureBars({ features = {} }) {
  if (!features || !Object.keys(features).length) return <p className="empty-inline">Audio characteristics are unavailable for these tracks.</p>
  return <div className="feature-bars">{Object.entries(features).map(([key, value]) => <div className="feature-bar" key={key}><div><span>{key}</span><b>{Math.round(value * 100)}%</b></div><span className="bar-track"><i style={{ width: `${Math.max(3, Math.min(100, value * 100))}%` }} /></span></div>)}</div>
}

function AuthPage({ mode, onModeChange, onSubmit, busy, error }) {
  const registering = mode === 'register'
  const [showPassword, setShowPassword] = useState(false)
  const [formError, setFormError] = useState('')
  const [fields, setFields] = useState({ name: '', email: '', password: '', confirmPassword: '' })
  const set = key => event => setFields(current => ({ ...current, [key]: event.target.value }))
  const handleSubmit = event => {
    event.preventDefault(); setFormError('')
    if (registering && fields.password !== fields.confirmPassword) {
      setFormError('Your passwords do not match.')
      return
    }
    onSubmit(mode, { name: fields.name, email: fields.email, password: fields.password })
  }
  return <section className="auth-page">
    <div className="auth-orb auth-orb-one" /><div className="auth-orb auth-orb-two" />
    <form className="auth-card" onSubmit={handleSubmit}>
      <div className="auth-mark"><AudioLines size={22} /></div>
      <span className="section-kicker">YOUR MUSIC, YOUR SPACE</span>
      <h1>{registering ? 'Create your account' : 'Welcome back'}</h1>
      <p>{registering ? 'Save your music preferences and pick up where you left off.' : 'Sign in to restore your saved tracks and preferences.'}</p>
      {(error || formError) && <div className="auth-error" role="alert">{formError || error}</div>}
      {registering && <label className="auth-field">Name<input autoComplete="name" maxLength="100" required value={fields.name} onChange={set('name')} placeholder="Your name" /></label>}
      <label className="auth-field">Email<input type="email" autoComplete="email" maxLength="254" required value={fields.email} onChange={set('email')} placeholder="you@example.com" /></label>
      <label className="auth-field">Password<div className="password-input"><input type={showPassword ? 'text' : 'password'} autoComplete={registering ? 'new-password' : 'current-password'} minLength="8" maxLength="128" required value={fields.password} onChange={set('password')} placeholder="At least 8 characters" /><button type="button" aria-label={showPassword ? 'Hide password' : 'Show password'} onClick={() => setShowPassword(value => !value)}>{showPassword ? <EyeOff size={17} /> : <Eye size={17} />}</button></div></label>
      {registering && <label className="auth-field">Confirm password<input type={showPassword ? 'text' : 'password'} autoComplete="new-password" minLength="8" maxLength="128" required value={fields.confirmPassword} onChange={set('confirmPassword')} placeholder="Re-enter your password" /></label>}
      {registering && <small className="auth-hint">Use 8–128 characters. Passwords are stored as secure hashes.</small>}
      <button className="primary-button auth-submit" type="submit" disabled={busy}>{busy ? <><LoaderCircle size={17} className="spin" />{registering ? 'Creating account…' : 'Signing in…'}</> : <>{registering ? 'Create account' : 'Sign in'} <ArrowRight size={17} /></>}</button>
      <p className="auth-switch">{registering ? 'Already have an account?' : 'New to Tunely?'} <button type="button" onClick={() => onModeChange(registering ? 'login' : 'register')}>{registering ? 'Sign in' : 'Create an account'}</button></p>
    </form>
  </section>
}

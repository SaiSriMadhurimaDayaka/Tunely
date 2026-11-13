const VIDEO_ID = /^[A-Za-z0-9_-]{11}$/

/** Resolve only real supplied video IDs; otherwise search using actual metadata. */
export function youtubeMusicDestination(song = {}) {
  let videoId = song.youtube_video_id
  if (!VIDEO_ID.test(videoId || '')) {
    try {
      const stored = new URL(song.youtube_music_url)
      if (stored.protocol === 'https:' && ['music.youtube.com', 'youtube.com', 'www.youtube.com'].includes(stored.hostname)) {
        videoId = stored.searchParams.get('v')
      } else if (stored.protocol === 'https:' && stored.hostname === 'youtu.be') {
        videoId = stored.pathname.slice(1)
      }
    } catch { /* Fall back to a title/artist search if no verified video URL exists. */ }
  }
  if (VIDEO_ID.test(videoId || '')) return `https://music.youtube.com/watch?v=${encodeURIComponent(videoId)}`

  const title = song.song_name || song.title
  if (typeof title !== 'string' || !title.trim()) return null
  const artist = typeof song.artist === 'string' ? song.artist.trim() : ''
  const query = [title.trim(), artist].filter(Boolean).join(' ')
  return `https://music.youtube.com/search?q=${encodeURIComponent(query)}`
}

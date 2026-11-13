import test from 'node:test'
import assert from 'node:assert/strict'
import { youtubeMusicDestination } from '../src/youtubeMusic.js'

test('a real YouTube video ID opens its YouTube Music song URL', () => {
  assert.equal(youtubeMusicDestination({ youtube_video_id: 'Abcdef_1234', song_name: 'Song' }),
    'https://music.youtube.com/watch?v=Abcdef_1234')
})

test('a verified stored YouTube URL is converted to YouTube Music', () => {
  assert.equal(youtubeMusicDestination({ youtube_music_url: 'https://www.youtube.com/watch?v=Abcdef_1234' }),
    'https://music.youtube.com/watch?v=Abcdef_1234')
})

test('missing IDs produce a properly encoded title and artist search', () => {
  const destination = youtubeMusicDestination({ song_name: "Don't Stop, 東京 & Me", artist: "Zoë O'Neil" })
  assert.equal(new URL(destination).hostname, 'music.youtube.com')
  assert.equal(new URL(destination).pathname, '/search')
  assert.equal(new URL(destination).searchParams.get('q'), "Don't Stop, 東京 & Me Zoë O'Neil")
})

test('invalid IDs are never used and missing titles produce no destination', () => {
  const destination = youtubeMusicDestination({ youtube_video_id: 'invented', song_name: 'Actual title', artist: 'Actual artist' })
  assert.equal(new URL(destination).searchParams.get('q'), 'Actual title Actual artist')
  assert.equal(youtubeMusicDestination({ youtube_video_id: 'invented' }), null)
})

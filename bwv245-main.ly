% BWV 245 - O große Lieb, o Lieb ohn alle Maße
% Arranged for SATB (4 staves)
\version "2.25.26"

% Global settings
global = {
  \key g \minor
  \time 4/4
  %% \tempo 4 = 70
}

% Soprano
sopranoMusic = {
  \global
  {
    \partial 4
    g'4
    | % 1
    g' g' fis' \fermata d'
    | % 2
    g' a' bes' bes'
    | % 3
    c'' ( bes' ) a' \fermata a'
    | % 4
    bes' c'' d''8 ( c'' bes'4 )
    | % 5
    ees'' ees'' d'' des''8 ( c'' )
    | % 6
    c''2 bes'4 \fermata bes'
    | % 7
    a' g' f' d'8 ( ees' )
    | % 8
    f'4 f' g' f'
    | % 9
    ees'2 d'4 \fermata d''
    | % 10
    c'' bes'8 ( a' ) a'2
    | % 11
    g'2. \fermata \bar "|."
  }
}

bassLyrics = \lyricmode {
  O
  gro -- ße Lieb, o
  Lieb' ohn al -- le
  Ma -- ße, die
  dich ge -- bracht -- auf
  die -- se Mar -- ter stra -- ße, ich
  leb -- te mit der
  Welt in Lust und
  Freu -- den, und
  du mußt -- lei -- den.

}

% Alto
altoMusic = {
  \global
  {
    \partial 4
    d'4
    | % 1
    ees'8 ( d' ) c'4 d' a
    | % 2
    d'8 ( e' ) fis'4 g' g'
    | % 3
    a' ( g' ) fis' fis'
    | % 4
    g' a' bes' f'
    | % 5
    g'8 ( a' bes'4 ) bes' bes'
    | % 6
    bes' ( a' ) f' g'8 ( f' )
    | % 7
    ees' ( d' ) e'4 d' a
    | % 8
    d'8 ( ees' ) f'4 f'8 ( ees' ) ees' ( d' )
    | % 9
    d' ( c'16 bes c'4 ) bes bes'
    | % 10
    a' g' g' ( fis' )
    | % 11
    d'2.
  }
}

% Tenor
tenorMusic = {
  \global
  {
    \partial 4
    bes4
    | % 1
    bes a8 ( g ) a4 fis
    | % 2
    g c' d' d'
    | % 3
    ees' ( d' ) d' d'
    | % 4
    d' ees' f'8 ees' d' c'
    | % 5
    bes4 bes8 c' d' bes g'4
    | % 6
    f'4. ees'8 d'4 d'
    | % 7
    c' bes a f
    | % 8
    bes bes bes a
    | % 9
    bes ( f ) f f'
    | % 10
    ees' d' e' ( d'8 c' )
    | % 11
    b2.
  }
}

% Bass
bassMusic = {
  \global
  {
    \partial 4
    g4
    | % 1
    c8 ( d ) ees4 d c
    | % 2
    bes, a, g, g
    | % 3
    fis ( g ) d d
    | % 4
    g c' bes aes
    | % 5
    g fis f e
    | % 6
    f2 bes,4 g,
    | % 7
    c cis d d8 ( c )
    | % 8
    bes, ( c ) d ( bes, ) ees4 f
    | % 9
    g ( a ) bes bes
    | % 10
    fis g cis ( d )
    | % 11
    g,2.
  }
}

% Score layout
bwv =
\new ChoirStaff <<
  \new Staff \with {
    instrumentName = "Soprano"
    shortInstrumentName = "S."
  } {
    \new Voice = "soprano" {
      \set Voice.midiInstrument = #"flute"
      \sopranoMusic
    }
  }

  \new Staff \with {
    instrumentName = "Alto"
    shortInstrumentName = "A."
  } {
    \new Voice = "alto" {
      \set Voice.midiInstrument = #"oboe"
      \altoMusic
    }
  }

  \new Staff \with {
    instrumentName = "Tenor"
    shortInstrumentName = "T."
    \clef "treble_8"
  } {
    \new Voice = "tenor" {
      \set Voice.midiInstrument = #"clarinet"
      \tenorMusic
    }
  }

  \new Staff \with {
    instrumentName = "Bass"
    shortInstrumentName = "B."
    \clef bass
  } {
    \new Voice = "bass" {
      \set Voice.midiInstrument = #"bassoon"
      \bassMusic
    }
  }

  \new Lyrics \lyricsto "bass" \bassLyrics
>>


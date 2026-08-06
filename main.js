import { HandLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import * as Tone from "tone";

// ═══════════════════════════════════════════════════════════════
//  CONSTANTS & CONFIG
// ═══════════════════════════════════════════════════════════════

const FINGERS = {
  index: { pip: 6, tip: 8 },
  middle: { pip: 10, tip: 12 },
  ring: { pip: 14, tip: 16 },
  pinky: { pip: 18, tip: 20 },
};

const DEGREE_SEMITONES = { 1: 0, 2: 2, 3: 4, 4: 5, 5: 7, 6: 9, 7: -1 };
const NUMERAL_TO_DEGREE = { I: 1, II: 2, III: 3, IV: 4, V: 5, VI: 6, VII: 7 };

const MAJOR_SCALE = {
  A:  ["A","B","C#","D","E","F#","G#"],
  Bb: ["Bb","C","D","Eb","F","G","A"],
  B:  ["B","C#","D#","E","F#","G#","A#"],
  C:  ["C","D","E","F","G","A","B"],
  Db: ["Db","Eb","F","Gb","Ab","Bb","C"],
  D:  ["D","E","F#","G","A","B","C#"],
  Eb: ["Eb","F","G","Ab","Bb","C","D"],
  E:  ["E","F#","G#","A","B","C#","D#"],
  F:  ["F","G","A","Bb","C","D","E"],
  Gb: ["Gb","Ab","Bb","Cb","Db","Eb","F"],
  G:  ["G","A","B","C","D","E","F#"],
  Ab: ["Ab","Bb","C","Db","Eb","F","G"]
};

const DRUM_PATTERNS = [
  { name: "House",  kick: [1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0], snare: [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0], hihat: [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1], clap:  [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0] },
  { name: "Techno", kick: [1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,1,0], snare: [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0], hihat: [0,0,1,0, 0,0,1,0, 0,0,1,0, 0,0,1,0], clap:  [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0] },
  { name: "Dubstep",kick: [1,0,0,0, 0,0,0,0, 1,0,0,0, 0,0,0,0], snare: [0,0,0,0, 0,0,0,0, 1,0,0,0, 0,0,0,0], hihat: [1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,1,0], clap:  [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0] },
  { name: "Trance", kick: [1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0], snare: [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0], hihat: [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1], clap:  [0,0,0,0, 0,0,1,0, 0,0,0,0, 0,0,1,0] },
  { name: "Drum&Bass", kick: [1,0,0,1, 0,0,1,0, 1,0,0,1, 0,0,1,0], snare: [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0], hihat: [1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,1,0], clap:  [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0] },
  { name: "Trap",   kick: [1,0,0,0, 0,0,0,0, 1,0,0,0, 0,0,1,0], snare: [0,0,0,0, 0,0,0,0, 1,0,0,0, 0,0,0,0], hihat: [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1], clap:  [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0] }
];

const PRESETS = {
  house: {
    lead: { osc: "sawtooth", attack: 0.01, decay: 0.2, sustain: 0.5, release: 0.5, filterBase: 200, filterOct: 4, volume: -6 },
    bass: { osc: "square", attack: 0.01, decay: 0.3, sustain: 0.8, release: 0.3, filterBase: 100, filterOct: 2, volume: -3 },
    pad:  { osc: "triangle", attack: 0.5, decay: 0.5, sustain: 0.8, release: 2, volume: -10 },
    drums: { kickPitch: 0.05, kickOct: 10, snareDecay: 0.2 },
    effects: { reverbDecay: 2.5, reverbWet: 0.25, delayTime: "8n", delayFeedback: 0.35, chorusRate: 2.5, chorusDepth: 0.4, distortion: 0.15 },
    filter: { cutoff: 2000, Q: 1 },
    lfo: { rate: 4, amount: 0.3 }
  },
  techno: {
    lead: { osc: "square", attack: 0.001, decay: 0.1, sustain: 0.3, release: 0.2, filterBase: 300, filterOct: 3, volume: -8 },
    bass: { osc: "sawtooth", attack: 0.001, decay: 0.4, sustain: 0.9, release: 0.1, filterBase: 80, filterOct: 3, volume: -2 },
    pad:  { osc: "sawtooth", attack: 0.8, decay: 0.3, sustain: 0.6, release: 1.5, volume: -12 },
    drums: { kickPitch: 0.02, kickOct: 12, snareDecay: 0.15 },
    effects: { reverbDecay: 1.5, reverbWet: 0.15, delayTime: "16n", delayFeedback: 0.5, chorusRate: 1, chorusDepth: 0.2, distortion: 0.3 },
    filter: { cutoff: 3000, Q: 2 },
    lfo: { rate: 8, amount: 0.5 }
  },
  dubstep: {
    lead: { osc: "fmsawtooth", attack: 0.05, decay: 0.4, sustain: 0.4, release: 0.8, filterBase: 100, filterOct: 5, volume: -6 },
    bass: { osc: "fmsquare", attack: 0.05, decay: 0.5, sustain: 0.7, release: 0.4, filterBase: 60, filterOct: 4, volume: -1 },
    pad:  { osc: "sine", attack: 1, decay: 0.5, sustain: 0.9, release: 3, volume: -14 },
    drums: { kickPitch: 0.08, kickOct: 8, snareDecay: 0.3 },
    effects: { reverbDecay: 3, reverbWet: 0.35, delayTime: "4n", delayFeedback: 0.6, chorusRate: 3, chorusDepth: 0.6, distortion: 0.4 },
    filter: { cutoff: 800, Q: 4 },
    lfo: { rate: 2, amount: 0.8 }
  },
  trance: {
    lead: { osc: "sawtooth", attack: 0.01, decay: 0.3, sustain: 0.6, release: 1, filterBase: 400, filterOct: 5, volume: -5 },
    bass: { osc: "sawtooth", attack: 0.01, decay: 0.2, sustain: 0.9, release: 0.2, filterBase: 120, filterOct: 3, volume: -3 },
    pad:  { osc: "triangle", attack: 1.5, decay: 0.3, sustain: 0.7, release: 4, volume: -12 },
    drums: { kickPitch: 0.03, kickOct: 10, snareDecay: 0.25 },
    effects: { reverbDecay: 5, reverbWet: 0.4, delayTime: "8n", delayFeedback: 0.45, chorusRate: 4, chorusDepth: 0.7, distortion: 0.1 },
    filter: { cutoff: 4000, Q: 0.8 },
    lfo: { rate: 6, amount: 0.2 }
  },
  ambient: {
    lead: { osc: "sine", attack: 0.5, decay: 1, sustain: 0.8, release: 3, filterBase: 500, filterOct: 2, volume: -10 },
    bass: { osc: "sine", attack: 0.3, decay: 0.8, sustain: 0.9, release: 2, filterBase: 80, filterOct: 1, volume: -6 },
    pad:  { osc: "sine", attack: 2, decay: 1, sustain: 0.9, release: 6, volume: -16 },
    drums: { kickPitch: 0.1, kickOct: 4, snareDecay: 0.5 },
    effects: { reverbDecay: 8, reverbWet: 0.6, delayTime: "4n", delayFeedback: 0.7, chorusRate: 1, chorusDepth: 0.8, distortion: 0.05 },
    filter: { cutoff: 3000, Q: 0.5 },
    lfo: { rate: 0.5, amount: 0.1 }
  }
};

// ═══════════════════════════════════════════════════════════════
//  DOM REFERENCES
// ═══════════════════════════════════════════════════════════════

const videoEl = document.getElementById("webcam");
const canvasEl = document.getElementById("overlay");
const ctx = canvasEl.getContext("2d");
const spectrumCanvas = document.getElementById("spectrumCanvas");
const specCtx = spectrumCanvas.getContext("2d");

const chordDisplayEl = document.getElementById("chordDisplay");
const volumeBarEls = Array.from(document.querySelectorAll(".vol-bar"));
const qualityDisplayEl = document.getElementById("qualityDisplay");
const startOverlayEl = document.getElementById("startOverlay");
const startBtn = document.getElementById("startBtn");
const helpButton = document.getElementById("helpButton");
const helpModal = document.getElementById("helpModal");
const closeHelp = document.getElementById("closeHelp");

const keySelectEl = document.getElementById("keySelect");
const toneSelectEl = document.getElementById("toneSelect");
const bpmSlider = document.getElementById("bpmSlider");
const bpmValue = document.getElementById("bpmValue");
const presetSelect = document.getElementById("presetSelect");
const playBtn = document.getElementById("playBtn");
const recBtn = document.getElementById("recBtn");
const clearBtn = document.getElementById("clearBtn");

const arpToggle = document.getElementById("arpToggle");
const looperToggle = document.getElementById("looperToggle");
const arpRateSelect = document.getElementById("arpRate");
const arpModeSelect = document.getElementById("arpMode");
const lfoRateSlider = document.getElementById("lfoRate");
const lfoRateVal = document.getElementById("lfoRateVal");
const wobbleSlider = document.getElementById("wobbleAmt");
const wobbleVal = document.getElementById("wobbleVal");

const patternSelect = document.getElementById("patternSelect");
const sequencerGrid = document.getElementById("sequencerGrid");

const leftHandInd = document.getElementById("leftHandInd");
const rightHandInd = document.getElementById("rightHandInd");
const gestureStatus = document.getElementById("gestureStatus");

// Effect knobs
const masterFilterCutoff = document.getElementById("masterFilterCutoff");
const masterFilterQ = document.getElementById("masterFilterQ");
const distortionAmt = document.getElementById("distortionAmt");
const distortionWet = document.getElementById("distortionWet");
const reverbDecay = document.getElementById("reverbDecay");
const reverbWet = document.getElementById("reverbWet");
const delayTime = document.getElementById("delayTime");
const delayFeedback = document.getElementById("delayFeedback");
const chorusRate = document.getElementById("chorusRate");
const chorusDepth = document.getElementById("chorusDepth");
const compThresh = document.getElementById("compThresh");
const compRatio = document.getElementById("compRatio");

// ═══════════════════════════════════════════════════════════════
//  GLOBAL STATE
// ═══════════════════════════════════════════════════════════════

let currentTonicFreq = Number(keySelectEl.value);
let currentKeyName = keySelectEl.selectedOptions[0].dataset.note;
let currentWaveform = toneSelectEl.value;

let isPlaying = false;
let isRecording = false;
let isArpOn = false;
let isLooperOn = false;
let currentPreset = "house";
let currentPattern = 0;
let seqState = JSON.parse(JSON.stringify(DRUM_PATTERNS[0]));

// Looper
let loopEvents = [];
let loopPart = null;
let loopLength = "2m"; // 2 measures

// Arpeggiator
let arpNotes = [];
let arpIndex = 0;
let arpDirection = 1;
let arpPart = null;

// Gesture stabilizers
const CHORD_HOLD_TIME_MS = 300;
const VIBE_NULL_WINDOW_MS = 50;
let stableChordState = null;
let candidateChordState = null;
let candidateChordSince = 0;
let lastChordSeenValidTime = 0;

// Current playing notes tracking
let currentLeadNotes = [];
let currentBassFreq = null;
let currentPadNotes = [];

// ═══════════════════════════════════════════════════════════════
//  TONE.JS ELECTRONIC MUSIC ENGINE
// ═══════════════════════════════════════════════════════════════

class ElectronicMusicEngine {
  constructor() {
    this.initialized = false;
    this.transportStarted = false;
  }

  async init() {
    await Tone.start();
    Tone.Transport.bpm.value = 128;

    // ── Master Chain ──
    this.masterCompressor = new Tone.Compressor(-24, 4);
    this.masterLimiter = new Tone.Limiter(-1);
    this.masterBus = new Tone.Gain(0.85);

    // Master effects
    this.masterFilter = new Tone.Filter(2000, "lowpass");
    this.masterFilter.Q.value = 1;

    // Bitcrusher for lo-fi distortion
    this.bitcrusher = new Tone.BitCrusher(4, 4); // bits, normfreq
    this.bitcrusher.wet.value = 0; // start dry

    // Flanger for modulation effects
    this.flanger = new Tone.FeedbackDelay("8n", 0.2); // Using FeedbackDelay as Flanger substitute
    this.flanger.wet.value = 0; // start dry

    // Chain master effects: bus -> filter -> bitcrusher -> flanger -> compressor -> limiter -> destination
    this.masterBus.chain(
      this.masterFilter,
      this.bitcrusher,
      this.flanger,
      this.masterCompressor,
      this.masterLimiter,
      Tone.Destination
    );

    // Analysers for visualization
    this.fftAnalyser = new Tone.Analyser("fft", 2048);
    this.waveAnalyser = new Tone.Analyser("waveform", 1024);
    this.masterBus.fan(this.fftAnalyser, this.waveAnalyser);

    // ── Lead Track ──
    this.leadSynth = new Tone.MonoSynth({
      oscillator: { type: "sawtooth" },
      envelope: { attack: 0.01, decay: 0.2, sustain: 0.5, release: 0.5 },
      filterEnvelope: { attack: 0.01, decay: 0.2, sustain: 0.5, baseFrequency: 200, octaves: 4, exponent: 2 },
      volume: -6
    });
    this.leadDistortion = new Tone.Distortion(0.15);
    // New effects for expressive control
    this.leadBitcrusher = new Tone.BitCrusher(4, 4); // bits, normalfrequency
    this.leadWaveshaper = new Tone.WaveShaper();
    // Set a subtle curve for warmth (almost linear)
    this.leadWaveshaper.curve = new Float32Array([-0.99, -0.5, 0, 0.5, 0.99]);
    this.leadWaveshaper.oversample = '4x';
    this.leadFlanger = null; // Flanger not available in this Tone.js version
    this.leadPhaser = new Tone.Phaser();
    this.leadPhaser.frequency.value = 0; // no modulation by default
    this.leadTremolo = new Tone.Tremolo();
    this.leadTremolo.depth.value = 0; // no effect by default
    this.leadFilter = new Tone.Filter(2000, "lowpass");
    this.leadDelay = new Tone.FeedbackDelay("8n", 0.35);
    this.leadReverb = new Tone.Reverb({ decay: 2.5, wet: 0.25, preDelay: 0.01 });
    this.leadPanner = new Tone.Panner(0);
    // Chain: distortion -> bitcrusher -> waveshaper -> phaser -> tremolo -> filter -> delay -> reverb -> panner -> master
    this.leadSynth.chain(
      this.leadDistortion,
      this.leadBitcrusher,
      this.leadWaveshaper,
      this.leadPhaser,
      this.leadTremolo,
      this.leadFilter,
      this.leadDelay,
      this.leadReverb,
      this.leadPanner,
      this.masterBus
    );

    // ── Bass Track ──
    this.bassSynth = new Tone.MonoSynth({
      oscillator: { type: "square" },
      envelope: { attack: 0.01, decay: 0.3, sustain: 0.8, release: 0.3 },
      filterEnvelope: { attack: 0.01, decay: 0.1, sustain: 0.5, baseFrequency: 100, octaves: 2 },
      volume: -3
    });
    this.bassFilter = new Tone.Filter(800, "lowpass");
    this.bassDistortion = new Tone.Distortion(0.2);
    this.bassPanner = new Tone.Panner(0);
    this.bassSynth.chain(this.bassDistortion, this.bassFilter, this.bassPanner, this.masterBus);

    // ── Pad Track ──
    this.padSynth = new Tone.PolySynth(Tone.Synth, {
      oscillator: { type: "triangle" },
      envelope: { attack: 0.5, decay: 0.5, sustain: 0.8, release: 2 },
      volume: -10
    });
    this.padChorus = new Tone.Chorus(4, 2.5, 0.4).start();
    this.padReverb = new Tone.Reverb({ decay: 4, wet: 0.4 });
    this.padPanner = new Tone.Panner(0);
    this.padSynth.chain(this.padChorus, this.padReverb, this.padPanner, this.masterBus);

    // ── Drums ──
    this.kick = new Tone.MembraneSynth({
      pitchDecay: 0.05,
      octaves: 10,
      oscillator: { type: "sine" },
      envelope: { attack: 0.001, decay: 0.4, sustain: 0.01, release: 1.4 },
      volume: 0
    });

    this.snare = new Tone.NoiseSynth({
      noise: { type: "white" },
      envelope: { attack: 0.001, decay: 0.2, sustain: 0 },
      volume: -2
    });

    this.hihat = new Tone.MetalSynth({
      frequency: 200,
      envelope: { attack: 0.001, decay: 0.1, release: 0.01 },
      harmonicity: 5.1,
      modulationIndex: 32,
      resonance: 4000,
      octaves: 1.5,
      volume: -12
    });

    this.clap = new Tone.NoiseSynth({
      noise: { type: "pink" },
      envelope: { attack: 0.001, decay: 0.3, sustain: 0 },
      volume: -4
    });

    this.kick.connect(this.masterBus);
    this.snare.connect(this.masterBus);
    this.hihat.connect(this.masterBus);
    this.clap.connect(this.masterBus);

    // ── Master Effects ──
    this.masterFilter = new Tone.Filter(2000, "lowpass");
    this.masterFilter.Q.value = 1;

    // Bitcrusher for lo-fi distortion
    this.bitcrusher = new Tone.BitCrusher(4, 4); // bits, normfreq
    this.bitcrusher.wet.value = 0; // start dry

    // Flanger for modulation effects
    this.flanger = new Tone.FeedbackDelay("8n", 0.2); // Using FeedbackDelay as Flanger substitute
    this.flanger.wet.value = 0; // start dry

    // Envelope follower for auto-wah effects
    this.envelopeFollower = new Tone.Follower(0.1, 400);
    this.autoFilter = new Tone.Filter(800, "lowpass");
    this.autoFilter.Q.value = 2;
    this.envelopeFollower.connect(this.autoFilter.frequency);
    this.autoFilterDepth = 1000; // How much the filter moves

    // LFO for wobble
    this.lfo = new Tone.LFO("4n", 400, 2000);
    this.lfoFilter = new Tone.Filter(1000, "lowpass");
    this.lfo.connect(this.lfoFilter.frequency);
    this.lfo.start();

    // Chain master effects: bus -> filter -> bitcrusher -> flanger -> compressor -> limiter -> destination
    this.masterBus.chain(
      this.masterFilter,
      this.bitcrusher,
      this.flanger,
      this.masterCompressor,
      this.masterLimiter,
      Tone.Destination
    );

    // ── Sequencer ──
    this.setupSequencer();

    // ── Arpeggiator ──
    this.setupArpeggiator();

    this.initialized = true;
    this.applyPreset("house");
  }

  setupSequencer() {
    this.seq = new Tone.Sequence((time, step) => {
      const p = seqState;
      if (p.kick[step])  this.kick.triggerAttackRelease("C1", "8n", time);
      if (p.snare[step]) this.snare.triggerAttackRelease("8n", time);
      if (p.hihat[step]) this.hihat.triggerAttackRelease("32n", time, 0.3);
      if (p.clap[step])  this.clap.triggerAttackRelease("8n", time);

      Tone.Draw.schedule(() => {
        updateStepIndicator(step);
      }, time);
    }, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], "16n");
  }

  setupArpeggiator() {
    this.arpPart = new Tone.Sequence((time, _) => {
      if (!isArpOn || arpNotes.length === 0) return;
      const note = arpNotes[arpIndex];
      if (note) {
        this.leadSynth.triggerAttackRelease(note, arpRateSelect.value, time);
      }
      this.advanceArp();
    }, [0], "8n");
  }

  advanceArp() {
    const mode = arpModeSelect.value;
    if (mode === "up") {
      arpIndex = (arpIndex + 1) % arpNotes.length;
    } else if (mode === "down") {
      arpIndex = (arpIndex - 1 + arpNotes.length) % arpNotes.length;
    } else if (mode === "updown") {
      arpIndex += arpDirection;
      if (arpIndex >= arpNotes.length - 1) { arpDirection = -1; }
      if (arpIndex <= 0) { arpDirection = 1; }
    } else if (mode === "random") {
      arpIndex = Math.floor(Math.random() * arpNotes.length);
    }
  }

  setArpNotes(freqs) {
    if (!freqs || freqs.length === 0) {
      arpNotes = [];
      return;
    }
    arpNotes = freqs.map(f => {
      try { return Tone.Frequency(f).toNote(); } catch { return null; }
    }).filter(n => n);
    arpIndex = 0;
    arpDirection = 1;
  }

  playLeadNotes(freqs, volume01) {
    if (!freqs || freqs.length === 0) return;
    const notes = freqs.map(f => {
      try { return Tone.Frequency(f).toNote(); } catch { return null; }
    }).filter(n => n);
    if (notes.length === 0) return;

    const note = notes[0];
    const key = note;
    if (this._lastLeadKey === key) {
      this.leadSynth.volume.rampTo(Tone.gainToDb(Math.max(0.01, volume01)), 0.05);
      return;
    }
    this._lastLeadKey = key;
    this.leadSynth.triggerAttack(note);
    this.leadSynth.volume.rampTo(Tone.gainToDb(Math.max(0.01, volume01)), 0.05);
  }

  playBassNote(freq, volume01) {
    if (!freq) return;
    const note = Tone.Frequency(freq / 2).toNote();
    if (this._lastBassNote === note) {
      this.bassSynth.volume.rampTo(Tone.gainToDb(Math.max(0.01, volume01 * 0.7)), 0.05);
      return;
    }
    this._lastBassNote = note;
    this.bassSynth.triggerAttack(note);
    this.bassSynth.volume.rampTo(Tone.gainToDb(Math.max(0.01, volume01 * 0.7)), 0.05);
  }

  playPadNotes(freqs, volume01) {
    if (!freqs || freqs.length === 0) return;
    const notes = freqs.map(f => {
      try { return Tone.Frequency(f).toNote(); } catch { return null; }
    }).filter(n => n);
    if (notes.length === 0) return;

    const key = notes.join(",");
    if (this._lastPadKey === key) {
      this.padSynth.volume.rampTo(Tone.gainToDb(Math.max(0.01, volume01 * 0.5)), 0.1);
      return;
    }
    this._lastPadKey = key;
    this.padSynth.triggerAttack(notes);
    this.padSynth.volume.rampTo(Tone.gainToDb(Math.max(0.01, volume01 * 0.5)), 0.1);
  }

  stopLead() {
    if (this._lastLeadKey) {
      this.leadSynth.triggerRelease();
      this._lastLeadKey = null;
    }
  }
  stopBass() {
    if (this._lastBassNote) {
      this.bassSynth.triggerRelease();
      this._lastBassNote = null;
    }
  }
  stopPad() {
    if (this._lastPadKey) {
      this.padSynth.releaseAll();
      this._lastPadKey = null;
    }
  }

  updateMasterFilter(cutoff, Q) {
    this.masterFilter.frequency.rampTo(cutoff, 0.05);
    this.masterFilter.Q.rampTo(Q, 0.05);
  }

  updateDistortion(amount, wet) {
    this.leadDistortion.distortion = amount;
    this.leadDistortion.wet.value = wet;
    this.bassDistortion.distortion = amount * 0.5;
  }

  updateReverb(decay, wet) {
    this.leadReverb.decay = decay;
    this.leadReverb.wet.value = wet;
    this.padReverb.decay = decay * 1.5;
    this.padReverb.wet.value = Math.min(wet * 1.2, 1);
  }

  updateDelay(timeVal, feedback) {
    const timeMap = ["32n","16n","8n","4n","2n"];
    const idx = Math.min(Math.floor(timeVal / 25), 4);
    this.leadDelay.delayTime.value = timeMap[idx];
    this.leadDelay.feedback.value = feedback / 100;
  }

  updateChorus(rate, depth) {
    this.padChorus.frequency.value = rate;
    this.padChorus.depth = depth / 100;
  }

  updateCompressor(thresh, ratio) {
    this.masterCompressor.threshold.value = thresh;
    this.masterCompressor.ratio.value = ratio;
  }

  updateLFO(rate, amount01) {
    if (!this.lfo || !this.lfo.frequency) return;
    this.lfo.frequency.value = rate;
    const minF = 200 + (1 - amount01) * 800;
    const maxF = 200 + amount01 * 4800;
    this.lfo.min = minF;
    this.lfo.max = maxF;
  }

  updateBitcrusher(amount) {
    // amount: 0 to 1, where 0 is no effect, 1 is full effect
    // We'll map to bits: 24 (clean) down to 4 (very lo-fi)
    const bits = 24 - Math.round(amount * 20); // 24 to 4
    this.bitcrusher.bits.value = bits;
    // Also adjust the wetness to mix in the effect
    this.bitcrusher.wet.value = amount * 0.8; // max 80% wet to avoid too much distortion
  }

  updateFlanger(amount) {
    // amount: 0 to 1
    this.flanger.wet.value = amount;
    // Also adjust some parameters for more interesting effect at higher amounts
    const depth = 0.2 + amount * 0.5; // 0.2 to 0.7
    const feedback = 0.1 + amount * 0.4; // 0.1 to 0.5
    this.flanger.effect.depth = depth;
    this.flanger.effect.feedback = feedback;
  }

  updateAutoFilter(amount) {
    // amount: 0 to 1
    // We'll use this to control the depth of the envelope follower modulation
    if (this.envDepth) {
      this.envDepth.gain.value = amount; // 0 to 1
    }
    // Also adjust the base frequency of the auto filter based on amount
    // More amount = wider sweep range
    const baseFreq = 200 + amount * 800; // 200 to 1000 Hz
    this.autoFilter.frequency.value = baseFreq;
  }

  updateFilterFromTilt(tiltFactor) {
    let targetFreq = 1200;
    let targetQ = 0.7;
    if (tiltFactor < 0) {
      const intensity = Math.abs(tiltFactor);
      targetFreq = 1200 - intensity * 950;
      targetQ = 0.7 + intensity * 1.5;
    } else if (tiltFactor > 0) {
      targetFreq = 1200 + tiltFactor * 3800;
      targetQ = 0.7 + tiltFactor * 4.5;
    }
    this.leadFilter.frequency.rampTo(targetFreq, 0.04);
    this.leadFilter.Q.rampTo(targetQ, 0.04);
    this.bassFilter.frequency.rampTo(Math.min(targetFreq * 0.4, 2000), 0.04);
  }

  // New effect update methods
  updateBitcrusher(amount) {
    // Map 0-100 to reasonable bitcrush range (4-24 bits)
    const bits = Math.floor(4 + (amount / 100) * 20);
    this.leadBitcrusher.bits = bits;
    // Also adjust the master bitcrusher for overall effect
    this.bitcrusher.bits = bits;
  }

  updateFlanger(amount) {
    // Map 0-100 to feedback delay parameters
    const feedback = amount / 100; // 0 to 1
    this.flanger.feedback.value = feedback;
  }

  updateAutoFilter(amount) {
    // Map 0-100 to filter frequency modulation amount (0-2000 Hz)
    const depth = (amount / 100) * 2000;
    this.autoFilterDepth = depth;
  }

  setLeadWaveform(type) {
    const map = { "sawtooth": "sawtooth", "square": "square", "triangle": "triangle", "sine": "sine", "fmsaw": "fmsawtooth", "pulse": "pulse" };
    this.leadSynth.oscillator.type = map[type] || "sawtooth";
  }

  applyPreset(name) {
    const p = PRESETS[name];
    if (!p) return;
    currentPreset = name;

    this.leadSynth.oscillator.type = p.lead.osc;
    this.leadSynth.envelope.attack = p.lead.attack;
    this.leadSynth.envelope.decay = p.lead.decay;
    this.leadSynth.envelope.sustain = p.lead.sustain;
    this.leadSynth.envelope.release = p.lead.release;
    this.leadSynth.filterEnvelope.baseFrequency = p.lead.filterBase;
    this.leadSynth.filterEnvelope.octaves = p.lead.filterOct;
    this.leadSynth.volume.value = p.lead.volume;

    this.bassSynth.oscillator.type = p.bass.osc;
    this.bassSynth.envelope.attack = p.bass.attack;
    this.bassSynth.envelope.decay = p.bass.decay;
    this.bassSynth.envelope.sustain = p.bass.sustain;
    this.bassSynth.envelope.release = p.bass.release;
    this.bassSynth.filterEnvelope.baseFrequency = p.bass.filterBase;
    this.bassSynth.filterEnvelope.octaves = p.bass.filterOct;
    this.bassSynth.volume.value = p.bass.volume;

    this.padSynth.set({ oscillator: { type: p.pad.osc } });
    this.padSynth.set({ envelope: { attack: p.pad.attack, decay: p.pad.decay, sustain: p.pad.sustain, release: p.pad.release } });
    this.padSynth.volume.value = p.pad.volume;

    this.kick.set({ pitchDecay: p.drums.kickPitch, octaves: p.drums.kickOct });
    this.snare.set({ envelope: { decay: p.drums.snareDecay } });

    this.updateReverb(p.effects.reverbDecay, p.effects.reverbWet);
    this.updateDelay(p.effects.delayTime === "4n" ? 75 : p.effects.delayTime === "8n" ? 50 : 25, p.effects.delayFeedback * 100);
    this.updateChorus(p.effects.chorusRate, p.effects.chorusDepth * 100);
    this.updateDistortion(p.effects.distortion, p.effects.distortion);
    this.updateMasterFilter(p.filter.cutoff, p.filter.Q);
    this.updateLFO(p.lfo.rate, p.lfo.amount);

    // New effect controls
    this.updateBitcrusher(p.effects.bitcrush || 0);
    this.updateFlanger(p.effects.flanger || 0);
    this.updateAutoFilter(p.effects.autoFilter || 0);

    masterFilterCutoff.value = p.filter.cutoff;
    masterFilterQ.value = p.filter.Q;
    distortionAmt.value = p.effects.distortion * 100;
    distortionWet.value = p.effects.distortion * 100;
    reverbDecay.value = p.effects.reverbDecay;
    reverbWet.value = p.effects.reverbWet * 100;
    delayTime.value = p.effects.delayTime === "4n" ? 75 : p.effects.delayTime === "8n" ? 50 : 25;
    delayFeedback.value = p.effects.delayFeedback * 100;
    chorusRate.value = p.effects.chorusRate;
    chorusDepth.value = p.effects.chorusDepth * 100;
    lfoRateSlider.value = p.lfo.rate;
    wobbleSlider.value = p.lfo.amount * 100;

    // New effect sliders (we'll add these to the UI later)
    // bitcrusherAmt.value = p.effects.bitcrush * 100;
    // flangerAmt.value = p.effects.flanger * 100;
    // autoFilterAmt.value = p.effects.autoFilter * 100;
  }

  startTransport() {
    if (!this.transportStarted) {
      Tone.Transport.start();
      this.seq.start(0);
      this.arpPart.start(0);
      this.transportStarted = true;
    }
  }

  stopTransport() {
    Tone.Transport.stop();
    this.seq.stop();
    this.arpPart.stop();
    this.transportStarted = false;
  }

  toggleTransport() {
    if (this.transportStarted) {
      this.stopTransport();
      return false;
    } else {
      this.startTransport();
      return true;
    }
  }
}

const engine = new ElectronicMusicEngine();

// ═══════════════════════════════════════════════════════════════
//  CHORD LOGIC
// ═══════════════════════════════════════════════════════════════

function isFingerExtended(landmarks, name) {
  const { pip, tip } = FINGERS[name];
  return landmarks[tip].y < landmarks[pip].y;
}

function isThumbExtended(landmarks, handedness) {
  const thumbTip = landmarks[4];
  const thumbIp = landmarks[3];
  if (handedness === "Right") return thumbTip.x > thumbIp.x;
  return thumbTip.x < thumbIp.x;
}

function getChordQuality(landmarks) {
  const wrist = landmarks[0];
  const middleMcp = landmarks[9];
  return middleMcp.x > wrist.x ? "minor" : "major";
}

function classifyChord(landmarks, handedness) {
  const thumb = isThumbExtended(landmarks, handedness);
  const index = isFingerExtended(landmarks, "index");
  const middle = isFingerExtended(landmarks, "middle");
  const ring = isFingerExtended(landmarks, "ring");
  const pinky = isFingerExtended(landmarks, "pinky");
  const quality = getChordQuality(landmarks); // true for major, false for minor

  // Special extended finger combinations
  // All fingers extended - 7th chord matching base quality
  if (thumb && index && middle && ring && pinky) {
    return quality ? "Imaj7" : "im7";
  }

  // Index + Middle + Ring - Sus4 (replaces 3rd with 4th)
  if (index && middle && ring && !thumb && !pinky) {
    const degreeMap = { 2: "IIsus4", 3: "IIIsus4", 4: "IVsus4", 5: "Vsus4", 1: "Isus4" };
    const count = [index, middle, ring].filter(Boolean).length;
    const base = degreeMap[count] || "Isus4";
    return quality ? base : base.toLowerCase();
  }

  // Index + Ring - Sus2 (replaces 3rd with 2nd)
  if (index && ring && !thumb && !middle && !pinky) {
    const degreeMap = { 2: "IIsus2", 3: "IIIsus2", 4: "IVsus2", 5: "Vsus2", 1: "Isus2" };
    const count = [index, ring].filter(Boolean).length;
    const base = degreeMap[count] || "Isus2";
    return quality ? base : base.toLowerCase();
  }

  // Index + Middle - 7th chord matching base quality
  if (index && middle && !thumb && !ring && !pinky) {
    const degreeMap = { 2: "II7", 3: "III7", 4: "IV7", 5: "V7", 1: "I7" };
    const count = [index, middle].filter(Boolean).length;
    const base = degreeMap[count] || "I7";
    // For minor mode, this becomes m7 instead of just 7
    return quality ? base : base.replace(/7$/, "m7");
  }

  // Index + Middle + Pinky - 7th chord matching base quality (alternate fingering)
  if (index && middle && pinky && !thumb && !ring) {
    const degreeMap = { 2: "II7", 3: "III7", 4: "IV7", 5: "V7", 1: "I7" };
    const count = [index, middle, pinky].filter(Boolean).length;
    const base = degreeMap[count] || "I7";
    return quality ? base : base.replace(/7$/, "m7");
  }

  // Existing VI/VII detection
  if (index && pinky && !middle && !ring && !thumb) return quality === "major" ? "VI" : "vi";
  if (index && pinky && !middle && !ring && thumb) return quality === "major" ? "VII" : "vii";

  // Basic triads (1-5 fingers)
  const count = [thumb, index, middle, ring, pinky].filter(Boolean).length;
  const ROMAN = { 1: "I", 2: "II", 3: "III", 4: "IV", 5: "V" };
  const base = ROMAN[count];
  if (!base) return null;
  return quality ? base : base.toLowerCase();
}

function getHandHorizontalTilt(landmarks, handedness) {
  if (!landmarks || landmarks.length < 18) return 0;
  try {
    const wrist = landmarks[0], middleMcp = landmarks[9], ringMcp = landmarks[13];
    if (!wrist || !middleMcp || !ringMcp) return 0;
    const minX = Math.min(middleMcp.x, ringMcp.x);
    const maxX = Math.max(middleMcp.x, ringMcp.x);
    const MAX_TRAVEL = 0.12;
    let tiltFactor = 0;
    if (wrist.x < minX) tiltFactor = (wrist.x - minX) / MAX_TRAVEL;
    else if (wrist.x > maxX) tiltFactor = (wrist.x - maxX) / MAX_TRAVEL;
    tiltFactor = Math.max(-1, Math.min(1, tiltFactor));
    if (handedness === "Right") tiltFactor = -tiltFactor;
    return tiltFactor;
  } catch (e) { return 0; }
}

function getVolumeFromHeight(landmarks) {
  const wrist = landmarks[0];
  const TOP = 0.05, BOTTOM = 0.95;
  const clamped = Math.max(TOP, Math.min(BOTTOM, wrist.y));
  return 1 - (clamped - TOP) / (BOTTOM - TOP);
}

function getRightHandQualityIndex(landmarks) {
  return [
    isFingerExtended(landmarks, "index"),
    isFingerExtended(landmarks, "middle"),
    isFingerExtended(landmarks, "ring"),
    isFingerExtended(landmarks, "pinky")
  ].filter(Boolean).length;
}

// New gesture detection functions for enhanced expressiveness
function getPalmOpenness(landmarks) {
  if (!landmarks || landmarks.length < 21) return 0;
  try {
    // Measure spread between thumb tip and pinky tip
    const thumbTip = landmarks[4];
    const pinkyTip = landmarks[20];
    if (!thumbTip || !pinkyTip) return 0;

    const dx = thumbTip.x - pinkyTip.x;
    const dy = thumbTip.y - pinkyTip.y;
    const distance = Math.sqrt(dx * dx + dy * dy);

    // Normalize to 0-1 range (typical spread is 0.1-0.3)
    return Math.min(1, Math.max(0, (distance - 0.05) / 0.25));
  } catch (e) {
    return 0;
  }
}

function getFingerSpread(landmarks) {
  if (!landmarks || landmarks.length < 21) return 0;
  try {
    // Measure average spread between consecutive fingertips
    const tips = [4, 8, 12, 16, 20]; // thumb, index, middle, ring, pinky
    let totalSpread = 0;
    let count = 0;

    for (let i = 0; i < tips.length - 1; i++) {
      const tip1 = landmarks[tips[i]];
      const tip2 = landmarks[tips[i + 1]];
      if (!tip1 || !tip2) continue;

      const dx = tip1.x - tip2.x;
      const dy = tip1.y - tip2.y;
      const distance = Math.sqrt(dx * dx + dy * dy);
      totalSpread += distance;
      count++;
    }

    const avgSpread = count > 0 ? totalSpread / count : 0;
    // Normalize to 0-1 range (typical spread is 0.02-0.08)
    return Math.min(1, Math.max(0, (avgSpread - 0.01) / 0.07));
  } catch (e) {
    return 0;
  }
}

function getWristRotation(landmarks) {
  if (!landmarks || landmarks.length < 21) return 0;
  try {
    // Use wrist and middle finger MCP to estimate roll
    const wrist = landmarks[0];
    const middleMcp = landmarks[9];
    const pinkyMcp = landmarks[13];
    const indexMcp = landmarks[5];

    if (!wrist || !middleMcp || !pinkyMcp || !indexMcp) return 0;

    // Calculate the angle of the hand relative to horizontal
    // Using the line between index and pinky MCP as reference
    const dx = pinkyMcp.x - indexMcp.x;
    const dy = pinkyMcp.y - indexMcp.y;
    const angle = Math.atan2(dy, dx);

    // Convert to -1 to 1 range (where 0 is flat, negative is clockwise roll)
    // Normalize by pi/2 to get reasonable range
    return Math.max(-1, Math.min(1, angle / (Math.PI / 2)));
  } catch (e) {
    return 0;
  }
}

function getDegreeFreq(degree) {
  const semitones = DEGREE_SEMITONES[degree];
  let tonic = currentTonicFreq;
  if (tonic === 369.99 || tonic === 392.00 || tonic === 415.30) tonic /= 2;
  return tonic * Math.pow(2, semitones / 12);
}

function getChordName(roman, isMajorMode) {
  if (!roman || roman === "--") return "";

  // Handle extended chord types in the display
  const upperRoman = roman.toUpperCase();

  // Map internal representations to display names
  const chordMap = {
    // Triads
    "I": "I", "II": "II", "III": "III", "IV": "IV", "V": "V",
    "VI": "VI", "VII": "VII",
    // Suspended
    "SUS2": "sus2", "SUS4": "sus4",
    // 7th chords
    "7": "7", "M7": "maj7", "MAJ7": "maj7",
    // Extended (we'll use these for our new chord types)
    "IMAJ7": "Imaj7", "IM7": "Imaj7",
    "II7": "II7", "IIM7": "IImaj7", "IIMAJ7": "IImaj7",
    "III7": "III7", "IIIM7": "IIImaj7", "IIIMAJ7": "IIImaj7",
    "IV7": "IV7", "IVM7": "IVmaj7", "IVMAJ7": "IVmaj7",
    "V7": "V7", "VM7": "Vmaj7", "VMAJ7": "Vmaj7",
    "VI7": "VI7", "VIM7": "VIm7", "VIMAJ7": "VImaj7",
    "VII7": "VII7", "VIM7": "VIm7", "VIIMAJ7": "VIImaj7",
    // Minor versions
    "IM7": "Im7", "IIM7": "IIm7", "IIM7": "IIm7", "IIM7": "IIm7",
    "IIM7": "IIm7", "IIM7": "IIm7", "IIIM7": "IIIm7",
    "IVM7": "IVm7", "VM7": "Vm7", "VIM7": "VIm7"
  };

  // Check if this is a known extended chord
  if (chordMap[upperRoman]) {
    let base = chordMap[upperRoman];
    // Adjust case for minor chords
    if (!isMajorMode && base.charAt(0) >= 'A' && base.charAt(0) <= 'Z') {
      // If it starts with uppercase letter and we're in minor mode,
      // make the first letter lowercase for minor chords
      base = base.charAt(0).toLowerCase() + base.substring(1);
    }
    return base;
  }

  // Handle regular triads
  const degree = NUMERAL_TO_DEGREE[upperRoman];
  if (!degree) return "";
  const root = MAJOR_SCALE[currentKeyName][degree - 1];
  return isMajorMode ? root : root + "m";
}

function getChordTones(numeralStr, isMajorMode) {
  if (!numeralStr || numeralStr === "--") return null;

  // Handle chord quality prefixes/suffixes
  let cleanNumeral = numeralStr.toUpperCase();
  let isMajor = isMajorMode;
  let chordType = "triad"; // triad, sus2, sus4, maj7, m7, dom7, dim7

  // Check for chord type modifiers
  if (cleanNumeral.endsWith("MAJ7") || cleanNumeral.endsWith("MA7")) {
    cleanNumeral = cleanNumeral.replace(/MAJ7?$/, "");
    chordType = "maj7";
  } else if (cleanNumeral.endsWith("M7")) {
    cleanNumeral = cleanNumeral.replace(/M7$/, "");
    chordType = "maj7";
  } else if (cleanNumeral.endsWith("7")) {
    // Could be dom7 or minor 7th
    if (cleanNumeral.length > 1 && !isNaN(parseInt(cleanNumeral.substring(0, cleanNumeral.length - 1)))) {
      // Has a number before the 7, likely dominant 7
      cleanNumeral = cleanNumeral.slice(0, -1);
      chordType = "dom7";
    } else {
      // Just a numeral + 7, treat as dominant 7
      chordType = "dom7";
    }
  } else if (cleanNumeral.endsWith("SUS4")) {
    cleanNumeral = cleanNumeral.replace(/SUS4$/, "");
    chordType = "sus4";
  } else if (cleanNumeral.endsWith("SUS2")) {
    cleanNumeral = cleanNumeral.replace(/SUS2$/, "");
    chordType = "sus2";
  } else if (cleanNumeral.endsWith("DIM7")) {
    cleanNumeral = cleanNumeral.replace(/DIM7$/, "");
    chordType = "dim7";
    isMajor = false; // diminished is always minor-like
  }

  // Parse the numeral degree
  const degree = NUMERAL_TO_DEGREE[cleanNumeral];
  if (!degree) return null;
  const root = getDegreeFreq(degree);

  // Calculate intervals based on chord type
  const thirdInterval = isMajor && chordType !== "sus2" && chordType !== "sus4" ? 4 : 3;
  const fifthInterval = chordType === "dim7" ? 6 : 7; // diminished fifth for dim7
  const seventhInterval = chordType === "maj7" ? 11 : (chordType === "dom7" || chordType === "dim7") ? 10 : null;

  const third = root * Math.pow(2, thirdInterval / 12);
  const fifth = root * Math.pow(2, fifthInterval / 12);
  const seventh = seventhInterval !== null ? root * Math.pow(2, seventhInterval / 12) : null;
  const octaveRoot = root * 2;
  const octaveThird = third * 2;

  // Build return object based on chord type
  switch (chordType) {
    case "sus2":
      const second = root * Math.pow(2, 2 / 12);
      return { root, second, fifth, octaveRoot, octaveThird };
    case "sus4":
      const fourth = root * Math.pow(2, 5 / 12);
      return { root, fourth, fifth, octaveRoot, octaveThird };
    case "maj7":
      return { root, third, fifth, seventh, octaveRoot, octaveThird };
    case "dom7":
      return { root, third, fifth, seventh, octaveRoot, octaveThird };
    case "dim7":
      const dimThird = root * Math.pow(2, 3 / 12); // minor third
      const dimFifth = root * Math.pow(2, 6 / 12); // diminished fifth
      const dimSeventh = root * Math.pow(2, 9 / 12); // diminished seventh
      return { root: dimThird, third: dimFifth, fifth: dimSeventh, octaveRoot: root * 2, octaveThird: dimFifth * 2 };
    default: // triad
      return { root, third, fifth, octaveRoot, octaveThird };
  }
}

function getSolidNotes(tones, rightHandCount, isMajorMode) {
  if (!tones) return [];

  // Determine what type of chord we have based on available properties
  const hasSeventh = tones.seventh !== undefined;
  const hasSecond = tones.second !== undefined;
  const hasFourth = tones.fourth !== undefined;
  const isSus2 = hasSecond && !hasSeventh && !hasFourth;
  const isSus4 = hasFourth && !hasSeventh && !hasSecond;
  const isSeventhChord = hasSeventh && !hasSecond && !hasFourth;

  // For diminished, we use a different structure in our getChordTones
  const isDim = tones.root !== tones.third && Math.abs(tones.third - tones.root) === 3; // minor third interval

  // Extract tone values with fallbacks
  const root = tones.root || 0;
  const second = tones.second || root * Math.pow(2, 2/12);
  const third = tones.third || (isMajorMode ? root * Math.pow(2, 4/12) : root * Math.pow(2, 3/12));
  const fourth = tones.fourth || root * Math.pow(2, 5/12);
  const fifth = tones.fifth || root * Math.pow(2, 7/12);
  const sixth = tones.sixth || root * Math.pow(2, 9/12); // For dim7
  const seventh = tones.seventh || (isMajorMode ? root * Math.pow(2, 11/12) : root * Math.pow(2, 10/12));
  const octaveRoot = tones.octaveRoot || root * 2;
  const octaveThird = tones.octaveThird || (isMajorMode ? third * 2 : third * 2);

  if (isDim) {
    // For diminished seventh, we have a special structure from getChordTones
    // root = minor third, third = diminished fifth, fifth = diminished seventh
    const dimRoot = tones.root;      // actually minor third
    const dimThird = tones.third;    // actually diminished fifth
    const dimFifth = tones.fifth;    // actually diminished seventh
    const octaveRoot = tones.octaveRoot || root * 2;
    const octaveThird = tones.octaveThird || dimThird * 2;

    if (isMajorMode) {
      switch (rightHandCount) {
        case 1: return [dimRoot, dimThird, octaveRoot, octaveThird];
        case 2: return [dimThird, dimFifth, octaveRoot, octaveThird];
        case 3: return [dimRoot, dimThird, dimFifth, octaveRoot];
        case 4: return [dimRoot, dimThird, dimFifth, octaveThird];
        default: return [dimRoot, dimThird, octaveRoot, octaveThird];
      }
    } else {
      switch (rightHandCount) {
        case 1: return [dimRoot, dimThird, octaveRoot, octaveThird];
        case 2: return [dimThird, dimFifth, octaveRoot, octaveThird];
        case 3: return [dimRoot, dimThird, dimFifth, octaveRoot];
        case 4: return [dimRoot, dimThird, dimFifth, octaveThird];
        default: return [dimRoot, dimThird, octaveRoot, octaveThird];
      }
    }
  }

  if (isSus2) {
    if (isMajorMode) {
      switch (rightHandCount) {
        case 1: return [root, second, fifth, octaveRoot, octaveThird];
        case 2: return [second, fifth, octaveRoot, octaveThird];
        case 3: return [root, second, fifth, octaveRoot];
        case 4: return [root, second, fifth, octaveThird];
        default: return [root, second, fifth, octaveRoot, octaveThird];
      }
    } else {
      switch (rightHandCount) {
        case 1: return [root, second, fifth, octaveRoot, octaveThird];
        case 2: return [second, fifth, octaveRoot, octaveThird];
        case 3: return [root, second, fifth, octaveRoot];
        case 4: return [root, second, fifth, octaveThird];
        default: return [root, second, fifth, octaveRoot, octaveThird];
      }
    }
  }

  if (isSus4) {
    if (isMajorMode) {
      switch (rightHandCount) {
        case 1: return [root, fourth, fifth, octaveRoot, octaveThird];
        case 2: return [fourth, fifth, octaveRoot, octaveThird];
        case 3: return [root, fourth, fifth, octaveRoot];
        case 4: return [root, fourth, fifth, octaveThird];
        default: return [root, fourth, fifth, octaveRoot, octaveThird];
      }
    } else {
      switch (rightHandCount) {
        case 1: return [root, fourth, fifth, octaveRoot, octaveThird];
        case 2: return [fourth, fifth, octaveRoot, octaveThird];
        case 3: return [root, fourth, fifth, octaveRoot];
        case 4: return [root, fourth, fifth, octaveThird];
        default: return [root, fourth, fifth, octaveRoot, octaveThird];
      }
    }
  }

  if (isSeventhChord) {
    // Handle 7th chords
    if (isMajorMode) {
      switch (rightHandCount) {
        case 1: return [root, third, fifth, seventh, octaveRoot];
        case 2: return [third, fifth, seventh, octaveRoot, octaveThird];
        case 3: return [root, third, fifth, seventh, octaveRoot];
        case 4: return [root, third, fifth, seventh, octaveThird];
        default: return [root, third, fifth, seventh, octaveRoot];
      }
    } else {
      // For minor mode with 7th, we need to know if it's m7 or dim7 or maj7
      // For simplicity, we'll treat minor mode 7th as minor 7th
      switch (rightHandCount) {
        case 1: return [root, third, fifth, seventh, octaveRoot];
        case 2: return [third, fifth, seventh, octaveRoot, octaveThird];
        case 3: return [root, third, fifth, seventh, octaveRoot];
        case 4: return [root, third, fifth, seventh, octaveThird];
        default: return [root, third, fifth, seventh, octaveRoot];
      }
    }
  }

  // Default to triad behavior (original code)
  if (isMajorMode) {
    switch (rightHandCount) {
      case 1: return [root, fifth, octaveRoot, octaveThird];
      case 2: return [third, fifth, octaveRoot, octaveThird];
      case 3: return [root, third, fifth, seventh !== undefined ? seventh : octaveRoot];
      case 4: return [root, third, fifth, seventh !== undefined ? seventh : octaveThird];
      default: return [root, fifth, octaveRoot, octaveThird];
    }
  } else {
    switch (rightHandCount) {
      case 1: return [root, fifth, octaveRoot, octaveThird];
      case 2: return [third, fifth, octaveRoot, octaveThird];
      case 3: return [root, third, fifth, seventh !== undefined ? seventh : octaveRoot];
      case 4: return [root, third, fifth, seventh !== undefined ? seventh : octaveThird];
      default: return [root, fifth, octaveRoot, octaveThird];
    }
  }
}

// ═══════════════════════════════════════════════════════════════
//  GESTURE STABILIZER
// ═══════════════════════════════════════════════════════════════

function sameChordState(a, b) {
  if (a === null && b === null) return true;
  if (a === null || b === null) return false;
  return a.chord === b.chord &&
         a.isMajorMode === b.isMajorMode &&
         a.qualityIndex === b.qualityIndex &&
         a.thumbDown === b.thumbDown &&
         Math.abs(a.palmOpenness - b.palmOpenness) < 0.1 &&
         Math.abs(a.fingerSpread - b.fingerSpread) < 0.1 &&
         Math.abs(a.wristRotation - b.wristRotation) < 0.1;
}

function stabilizeChordState(rawState, now) {
  if (rawState !== null) lastChordSeenValidTime = now;
  let effectiveState = rawState;
  if (rawState === null && now - lastChordSeenValidTime < VIBE_NULL_WINDOW_MS) {
    effectiveState = candidateChordState;
  }
  if (!sameChordState(effectiveState, candidateChordState)) {
    candidateChordState = effectiveState;
    candidateChordSince = now;
  }
  if (now - candidateChordSince >= CHORD_HOLD_TIME_MS) {
    stableChordState = candidateChordState;
  }
  return stableChordState;
}

// ═══════════════════════════════════════════════════════════════
//  CAMERA & MEDIAPIPE
// ═══════════════════════════════════════════════════════════════

async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480 },
    audio: false,
  });
  videoEl.srcObject = stream;
  return new Promise((resolve) => {
    videoEl.onloadedmetadata = () => { videoEl.play(); resolve(); };
  });
}

async function setupHandLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );

  const options = {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.65,
    minTrackingConfidence: 0.6,
    minHandPresenceConfidence: 0.6,
  };

  try {
    return await HandLandmarker.createFromOptions(vision, options);
  } catch (gpuErr) {
    console.warn("GPU delegate failed, falling back to CPU:", gpuErr);
    options.baseOptions.delegate = "CPU";
    return await HandLandmarker.createFromOptions(vision, options);
  }
}

// ═══════════════════════════════════════════════════════════════
//  SEQUENCER UI
// ═══════════════════════════════════════════════════════════════

const TRACK_NAMES = ["kick", "snare", "hihat", "clap"];
const TRACK_LABELS = ["Kick", "Snare", "Hi-Hat", "Clap"];
const TRACK_COLORS = ["#ff006e", "#8338ec", "#3a86ff", "#00ff88"];

function buildSequencerGrid() {
  sequencerGrid.innerHTML = "";
  TRACK_NAMES.forEach((track, tIdx) => {
    const label = document.createElement("div");
    label.className = "seq-label";
    label.textContent = TRACK_LABELS[tIdx];
    label.style.color = TRACK_COLORS[tIdx];
    sequencerGrid.appendChild(label);

    for (let s = 0; s < 16; s++) {
      const step = document.createElement("div");
      step.className = "seq-step";
      step.dataset.track = track;
      step.dataset.step = s;
      if (seqState[track][s]) step.classList.add("active");
      step.addEventListener("click", () => {
        seqState[track][s] = seqState[track][s] ? 0 : 1;
        step.classList.toggle("active");
      });
      sequencerGrid.appendChild(step);
    }
  });
}

function updateStepIndicator(step) {
  document.querySelectorAll(".seq-step").forEach(el => {
    el.classList.toggle("current", Number(el.dataset.step) === step);
  });
}

// ═══════════════════════════════════════════════════════════════
//  SPECTRUM VISUALIZER
// ═══════════════════════════════════════════════════════════════

function drawSpectrum() {
  if (!engine.initialized) return;
  const w = spectrumCanvas.width = spectrumCanvas.offsetWidth;
  const h = spectrumCanvas.height = spectrumCanvas.offsetHeight;
  specCtx.clearRect(0, 0, w, h);

  const values = engine.fftAnalyser.getValue();
  const barCount = 64;
  const barW = w / barCount;

  for (let i = 0; i < barCount; i++) {
    const idx = Math.floor(i * (values.length / 2) / barCount);
    const val = values[idx];
    const db = Math.max(-100, val);
    const norm = (db + 100) / 100;
    const barH = norm * h * 0.9;
    const hue = 320 + (i / barCount) * 80;
    specCtx.fillStyle = `hsla(${hue}, 80%, 60%, ${0.3 + norm * 0.7})`;
    specCtx.fillRect(i * barW, h - barH, barW - 1, barH);
  }

  // Waveform overlay
  const wave = engine.waveAnalyser.getValue();
  specCtx.beginPath();
  specCtx.strokeStyle = "rgba(255,255,255,0.15)";
  specCtx.lineWidth = 1;
  for (let i = 0; i < wave.length; i++) {
    const x = (i / wave.length) * w;
    const y = h / 2 + (wave[i] * h * 0.3);
    if (i === 0) specCtx.moveTo(x, y);
    else specCtx.lineTo(x, y);
  }
  specCtx.stroke();
}

// ═══════════════════════════════════════════════════════════════
//  ENERGY WAVE VISUALIZATION
// ═══════════════════════════════════════════════════════════════

function drawEnergy(ctx, volume01, qualityIndex, tiltFactor, chordStr) {
  if (!ctx || qualityIndex === 0) return;
  const lineCount = qualityIndex;
  try {
    const centerY = ctx.canvas.height - 56;
    const canvasWidth = ctx.canvas.width;
    const maxThickness = 1 + volume01 * 8;
    const chaosScale = (tiltFactor + 1) / 2;
    const shakinessAmp = chaosScale * 25;
    const shakinessFreq = 0.05 + chaosScale * 0.15;

    let baseColorRGB = "150,150,150";
    let isChordActive = false;
    let isMajor = false;

    if (chordStr && chordStr !== "--") {
      isChordActive = true;
      const upperStr = chordStr.toUpperCase();
      isMajor = (chordStr === upperStr);
      const SCALE_COLORS = {
        "I": "232,161,61", "II": "210,50,120", "III": "180,40,150",
        "IV": "240,210,40", "V": "245,120,30", "VI": "230,40,40", "VII": "100,200,250"
      };
      baseColorRGB = SCALE_COLORS[upperStr] || "232,161,61";
    }
    const brightnessAlpha = isChordActive ? (isMajor ? 1 : 0.7) : 0.3;

    ctx.save();
    const time = performance.now() * 0.004;
    const [r, g, b] = baseColorRGB.split(",").map(s => parseInt(s));

    ctx.shadowBlur = 10 + volume01 * 20;
    ctx.shadowColor = `rgba(${r},${g},${b},${0.5 * brightnessAlpha})`;

    for (let l = 0; l < lineCount; l++) {
      ctx.beginPath();
      const lineYOffset = centerY + (l - (lineCount - 1) / 2) * 12;
      for (let x = 0; x <= canvasWidth; x += 10) {
        const baseSine = Math.sin(x * 0.005 + time + l * 0.5) * 20;
        const jitter = (Math.random() - 0.5) * shakinessAmp * Math.sin(x * shakinessFreq + time);
        const y = lineYOffset + baseSine + jitter;
        if (x === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = `rgba(${r},${g},${b},${brightnessAlpha})`;
      ctx.lineWidth = Math.max(1, maxThickness - l * 0.5);
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.stroke();
    }
    ctx.restore();
  } catch (e) {
    console.error("Wave animation failed:", e);
  }
}

// ═══════════════════════════════════════════════════════════════
//  VIDEO FRAME DRAWING
// ═══════════════════════════════════════════════════════════════

function computeCoverRect(srcW, srcH, dstW, dstH) {
  const srcRatio = srcW / srcH;
  const dstRatio = dstW / dstH;
  if (srcRatio > dstRatio) {
    const sHeight = srcH;
    const sWidth = srcH * dstRatio;
    return { sx: (srcW - sWidth) / 2, sy: 0, sWidth, sHeight };
  } else {
    const sWidth = srcW;
    const sHeight = srcW / dstRatio;
    return { sx: 0, sy: (srcH - sHeight) / 2, sWidth, sHeight };
  }
}

function drawFrame(results, canvasWidth, canvasHeight) {
  const srcW = videoEl.videoWidth;
  const srcH = videoEl.videoHeight;
  if (!srcW || !srcH) return;
  const { sx, sy, sWidth, sHeight } = computeCoverRect(srcW, srcH, canvasWidth, canvasHeight);
  const landmarksList = results && Array.isArray(results.landmarks) ? results.landmarks : [];

  ctx.save();
  ctx.clearRect(0, 0, canvasWidth, canvasHeight);
  ctx.translate(canvasWidth, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(videoEl, sx, sy, sWidth, sHeight, 0, 0, canvasWidth, canvasHeight);

  ctx.fillStyle = "#ffffff80";
  for (const landmarks of landmarksList) {
    if (!Array.isArray(landmarks)) continue;
    for (const point of landmarks) {
      if (!point || typeof point.x !== 'number' || typeof point.y !== 'number') continue;
      const videoPx = point.x * srcW;
      const videoPy = point.y * srcH;
      const canvasX = ((videoPx - sx) / sWidth) * canvasWidth;
      const canvasY = ((videoPy - sy) / sHeight) * canvasHeight;
      ctx.beginPath();
      ctx.arc(canvasX, canvasY, 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  ctx.restore();
}

// ═══════════════════════════════════════════════════════════════
//  VOLUME METER
// ═══════════════════════════════════════════════════════════════

function updateVolumeMeter(volume01) {
  const litCount = Math.round(volume01 * volumeBarEls.length);
  volumeBarEls.forEach((bar) => {
    const index = Number(bar.dataset.index);
    bar.classList.toggle("lit", index >= volumeBarEls.length - litCount);
  });
}

// ═══════════════════════════════════════════════════════════════
//  EFFECT LABEL UPDATES
// ═══════════════════════════════════════════════════════════════

function updateEffectLabels() {
  document.getElementById("filterCutoffVal").textContent = (masterFilterCutoff.value / 1000).toFixed(1) + "k";
  document.getElementById("filterQVal").textContent = Number(masterFilterQ.value).toFixed(1);
  document.getElementById("distVal").textContent = distortionAmt.value;
  document.getElementById("distWetVal").textContent = distortionWet.value;
  document.getElementById("revDecVal").textContent = Number(reverbDecay.value).toFixed(1);
  document.getElementById("revWetVal").textContent = reverbWet.value;
  document.getElementById("delTimeVal").textContent = delayTime.value;
  document.getElementById("delFbVal").textContent = delayFeedback.value;
  document.getElementById("chorRateVal").textContent = Number(chorusRate.value).toFixed(1);
  document.getElementById("chorDepthVal").textContent = chorusDepth.value;
  document.getElementById("compThreshVal").textContent = compThresh.value;
  document.getElementById("compRatioVal").textContent = Number(compRatio.value).toFixed(1);
  lfoRateVal.textContent = Number(lfoRateSlider.value).toFixed(1);
  wobbleVal.textContent = wobbleSlider.value;
  // New effect labels
  document.getElementById("bitcrusherVal").textContent = bitcrusherAmt.value;
  document.getElementById("bitcrusherMixVal").textContent = bitcrusherMix.value;
  document.getElementById("flangerVal").textContent = flangerAmt.value;
  document.getElementById("flangerFeedbackVal").textContent = flangerFeedback.value;
  document.getElementById("autoFilterVal").textContent = autoFilterAmt.value;
}

// ═══════════════════════════════════════════════════════════════
//  UI EVENT LISTENERS
// ═══════════════════════════════════════════════════════════════

startBtn.addEventListener("click", async () => {
  await engine.init();
  startOverlayEl.style.display = "none";
  canvasEl.classList.remove("dimmed");
  buildSequencerGrid();
});

keySelectEl.addEventListener("change", () => {
  currentTonicFreq = Number(keySelectEl.value);
  currentKeyName = keySelectEl.selectedOptions[0].dataset.note;
});

toneSelectEl.addEventListener("change", () => {
  currentWaveform = toneSelectEl.value;
  engine.setLeadWaveform(currentWaveform);
});

bpmSlider.addEventListener("input", () => {
  const bpm = Number(bpmSlider.value);
  bpmValue.textContent = bpm;
  if (engine.initialized) Tone.Transport.bpm.rampTo(bpm, 0.5);
});

presetSelect.addEventListener("change", () => {
  engine.applyPreset(presetSelect.value);
});

playBtn.addEventListener("click", () => {
  if (!engine.initialized) return;
  isPlaying = engine.toggleTransport();
  playBtn.classList.toggle("active", isPlaying);
  playBtn.textContent = isPlaying ? "⏸" : "▶";
});

recBtn.addEventListener("click", () => {
  if (!engine.initialized) return;
  isRecording = !isRecording;
  recBtn.classList.toggle("recording", isRecording);
});

clearBtn.addEventListener("click", () => {
  loopEvents = [];
  if (loopPart) { loopPart.dispose(); loopPart = null; }
});

arpToggle.addEventListener("click", () => {
  isArpOn = !isArpOn;
  arpToggle.classList.toggle("on", isArpOn);
  arpToggle.textContent = isArpOn ? "琶音器 ON" : "琶音器 OFF";
});

looperToggle.addEventListener("click", () => {
  isLooperOn = !isLooperOn;
  looperToggle.classList.toggle("on", isLooperOn);
  looperToggle.textContent = isLooperOn ? "循環錄音 ON" : "循環錄音 OFF";
});

const settingsBtn = document.getElementById("settingsBtn");
const settingsModal = document.getElementById("settingsModal");
const closeSettings = document.getElementById("closeSettings");
const languageSelect = document.getElementById("languageSelect");

const translations = {
  "zh-TW": {
    startHeader: "GESTURE SYNTH",
    startSubtext: "EDM Edition — 手勢控制電音合成器",
    startButton: "點擊開始",
    logoText: "GESTURE SYNTH",
    badgeText: "EDM v2.0",
    topBpm: "BPM",
    topPreset: "Preset",
    playBtnTitle: "播放 / 暫停",
    recBtnTitle: "錄製循環",
    clearBtnTitle: "清除循環",
    settingsBtnTitle: "設定",
    helpBtnTitle: "說明",
    leftHandStatus: "左手: 和弦選擇",
    rightHandStatus: "右手: 音色控制",
    lfoLabel: "LFO",
    wobbleLabel: "Wobble",
    patternLabel: "Pattern",
    settingsHeading: "設定",
    languageLabel: "語言"
  },
  "en-US": {
    startHeader: "GESTURE SYNTH",
    startSubtext: "EDM Edition — gesture-controlled synth",
    startButton: "Tap to start",
    logoText: "GESTURE SYNTH",
    badgeText: "EDM v2.0",
    topBpm: "BPM",
    topPreset: "Preset",
    playBtnTitle: "Play / Pause",
    recBtnTitle: "Record Loop",
    clearBtnTitle: "Clear Loop",
    settingsBtnTitle: "Settings",
    helpBtnTitle: "Help",
    leftHandStatus: "Left: chord select",
    rightHandStatus: "Right: timbre control",
    lfoLabel: "LFO",
    wobbleLabel: "Wobble",
    patternLabel: "Pattern",
    settingsHeading: "Settings",
    languageLabel: "Language"
  }
};

function applyTranslations(lang) {
  const dictionary = translations[lang] || translations["zh-TW"];
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    const key = el.getAttribute("data-i18n");
    if (dictionary[key]) el.textContent = dictionary[key];
  });
  document.querySelectorAll("[data-i18n-title]").forEach((el) => {
    const key = el.getAttribute("data-i18n-title");
    if (dictionary[key]) el.setAttribute("title", dictionary[key]);
  });
}

if (settingsBtn && settingsModal) {
  settingsBtn.addEventListener("click", () => {
    settingsModal.classList.remove("hidden");
  });
}
if (closeSettings && settingsModal) {
  closeSettings.addEventListener("click", () => {
    settingsModal.classList.add("hidden");
  });
}
if (settingsModal) {
  settingsModal.addEventListener("click", (e) => {
    if (e.target === settingsModal) settingsModal.classList.add("hidden");
  });
}
if (languageSelect) {
  languageSelect.addEventListener("change", () => {
    const selectedLang = languageSelect.value;
    document.documentElement.lang = selectedLang;
    localStorage.setItem("gestureSynthLanguage", selectedLang);
    applyTranslations(selectedLang);
  });
  const savedLang = localStorage.getItem("gestureSynthLanguage");
  const initialLang = savedLang || document.documentElement.lang || "zh-TW";
  languageSelect.value = initialLang;
  document.documentElement.lang = initialLang;
  applyTranslations(initialLang);
}

patternSelect.addEventListener("change", () => {
  currentPattern = Number(patternSelect.value);
  seqState = JSON.parse(JSON.stringify(DRUM_PATTERNS[currentPattern]));
  buildSequencerGrid();
});

// Effect knobs
masterFilterCutoff.addEventListener("input", () => {
  engine.updateMasterFilter(Number(masterFilterCutoff.value), Number(masterFilterQ.value));
  updateEffectLabels();
});
masterFilterQ.addEventListener("input", () => {
  engine.updateMasterFilter(Number(masterFilterCutoff.value), Number(masterFilterQ.value));
  updateEffectLabels();
});
distortionAmt.addEventListener("input", () => {
  engine.updateDistortion(Number(distortionAmt.value) / 100, Number(distortionWet.value) / 100);
  updateEffectLabels();
});
distortionWet.addEventListener("input", () => {
  engine.updateDistortion(Number(distortionAmt.value) / 100, Number(distortionWet.value) / 100);
  updateEffectLabels();
});
reverbDecay.addEventListener("input", () => {
  engine.updateReverb(Number(reverbDecay.value), Number(reverbWet.value) / 100);
  updateEffectLabels();
});
reverbWet.addEventListener("input", () => {
  engine.updateReverb(Number(reverbDecay.value), Number(reverbWet.value) / 100);
  updateEffectLabels();
});
delayTime.addEventListener("input", () => {
  engine.updateDelay(Number(delayTime.value), Number(delayFeedback.value));
  updateEffectLabels();
});
delayFeedback.addEventListener("input", () => {
  engine.updateDelay(Number(delayTime.value), Number(delayFeedback.value));
  updateEffectLabels();
});
chorusRate.addEventListener("input", () => {
  engine.updateChorus(Number(chorusRate.value), Number(chorusDepth.value));
  updateEffectLabels();
});
chorusDepth.addEventListener("input", () => {
  engine.updateChorus(Number(chorusRate.value), Number(chorusDepth.value));
  updateEffectLabels();
});
compThresh.addEventListener("input", () => {
  engine.updateCompressor(Number(compThresh.value), Number(compRatio.value));
  updateEffectLabels();
});
compRatio.addEventListener("input", () => {
  engine.updateCompressor(Number(compThresh.value), Number(compRatio.value));
  updateEffectLabels();
});
lfoRateSlider.addEventListener("input", () => {
  if (engine.initialized) {
    engine.updateLFO(Number(lfoRateSlider.value), Number(wobbleSlider.value) / 100);
  }
  updateEffectLabels();
});
wobbleSlider.addEventListener("input", () => {
  if (engine.initialized) {
    engine.updateLFO(Number(lfoRateSlider.value), Number(wobbleSlider.value) / 100);
  }
  updateEffectLabels();
});

// New effect controls
bitcrusherAmt.addEventListener("input", () => {
  const amount = Number(bitcrusherAmt.value) / 100;
  engine.updateBitcrusher(amount);
  // Apply mix
  const mix = Number(bitcrusherMix.value) / 100;
  engine.bitcrusher.wet.value = amount * 0.8 * mix;
  updateEffectLabels();
});
bitcrusherMix.addEventListener("input", () => {
  const amount = Number(bitcrusherAmt.value) / 100;
  const mix = Number(bitcrusherMix.value) / 100;
  engine.bitcrusher.wet.value = amount * 0.8 * mix;
  updateEffectLabels();
});
flangerAmt.addEventListener("input", () => {
  engine.updateFlanger(Number(flangerAmt.value) / 100);
  updateEffectLabels();
});
flangerFeedback.addEventListener("input", () => {
  if (engine.initialized) {
    engine.flanger.effect.feedback = Number(flangerFeedback.value) / 100;
  }
  updateEffectLabels();
});
autoFilterAmt.addEventListener("input", () => {
  engine.updateAutoFilter(Number(autoFilterAmt.value) / 100);
  updateEffectLabels();
});

helpButton.addEventListener("click", () => helpModal.classList.remove("hidden"));
closeHelp.addEventListener("click", (e) => { e.stopPropagation(); helpModal.classList.add("hidden"); });
helpModal.addEventListener("click", (e) => { if (e.target === helpModal) helpModal.classList.add("hidden"); });

// ═══════════════════════════════════════════════════════════════
//  PANEL COLLAPSE
// ═══════════════════════════════════════════════════════════════

const panelHandle = document.getElementById("panelHandle");
const bottomPanel = document.getElementById("bottomPanel");
const panelLabel = document.getElementById("panelLabel");
const centerHUD = document.getElementById("centerHUD");

let isPanelCollapsed = false;
let centerHUDPosition = { x: null, y: 72 };
let hasCenterHUDMoved = false;
let dragState = { active: false, startX: 0, startY: 0, offsetX: 0, offsetY: 0 };
let panelDragState = { active: false, startY: 0, startHeight: 0, moved: false };
let lastDetectionTimestamp = 0;
const DETECTION_INTERVAL_MS = 200;

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function updateCenterHUDPosition() {
  if (!centerHUD) return;
  if (!hasCenterHUDMoved) {
    centerHUD.style.left = "50%";
    centerHUD.style.top = `${centerHUDPosition.y}px`;
    centerHUD.style.transform = "translateX(-50%)";
    return;
  }
  centerHUD.style.left = `${centerHUDPosition.x}px`;
  centerHUD.style.top = `${centerHUDPosition.y}px`;
  centerHUD.style.transform = "none";
}

function getCenterHUDBounds() {
  if (!centerHUD) return { halfWidth: 0, height: 0 };
  return { halfWidth: centerHUD.offsetWidth / 2, height: centerHUD.offsetHeight };
}

function startCenterDrag(event) {
  if (!centerHUD) return;
  if (!hasCenterHUDMoved) {
    const rect = centerHUD.getBoundingClientRect();
    centerHUDPosition.x = rect.left;
    centerHUDPosition.y = rect.top;
    hasCenterHUDMoved = true;
  }
  dragState.active = true;
  dragState.startX = event.clientX !== undefined ? event.clientX : event.touches?.[0]?.clientX;
  dragState.startY = event.clientY !== undefined ? event.clientY : event.touches?.[0]?.clientY;
  dragState.offsetX = centerHUDPosition.x;
  dragState.offsetY = centerHUDPosition.y;
  centerHUD.classList.add("dragging");
  if (event.pointerId && centerHUD.setPointerCapture) {
    centerHUD.setPointerCapture(event.pointerId);
  }
  event.preventDefault();
  event.stopPropagation();
}

function moveCenterDrag(event) {
  if (!dragState.active) return;
  const clientX = event.clientX !== undefined ? event.clientX : event.touches?.[0]?.clientX;
  const clientY = event.clientY !== undefined ? event.clientY : event.touches?.[0]?.clientY;
  if (typeof clientX !== "number" || typeof clientY !== "number") return;
  const deltaX = clientX - dragState.startX;
  const deltaY = clientY - dragState.startY;
  const { halfWidth, height } = getCenterHUDBounds();
  centerHUDPosition.x = clamp(dragState.offsetX + deltaX, halfWidth + 20, window.innerWidth - halfWidth - 20);
  centerHUDPosition.y = clamp(dragState.offsetY + deltaY, 20, window.innerHeight - height - 20);
  updateCenterHUDPosition();
  event.preventDefault();
}

function endCenterDrag(event) {
  if (!dragState.active) return;
  dragState.active = false;
  if (centerHUD) {
    centerHUD.classList.remove("dragging");
    if (event?.pointerId && centerHUD.releasePointerCapture) {
      centerHUD.releasePointerCapture(event.pointerId);
    }
  }
}

function startPanelDrag(event) {
  if (!bottomPanel) return;
  panelDragState.active = true;
  panelDragState.moved = false;
  panelDragState.startY = event.clientY !== undefined ? event.clientY : event.touches?.[0]?.clientY;
  panelDragState.startHeight = bottomPanel.getBoundingClientRect().height;
  bottomPanel.style.transition = "none";
  if (event.pointerId && panelHandle.setPointerCapture) {
    panelHandle.setPointerCapture(event.pointerId);
  }
  event.preventDefault();
  event.stopPropagation();
}

function movePanelDrag(event) {
  if (!panelDragState.active || !bottomPanel) return;
  const clientY = event.clientY !== undefined ? event.clientY : event.touches?.[0]?.clientY;
  if (typeof clientY !== "number") return;
  const deltaY = clientY - panelDragState.startY;
  if (Math.abs(deltaY) > 6) panelDragState.moved = true;
  const maxHeight = Math.max(360, window.innerHeight - 80);
  const targetHeight = clamp(panelDragState.startHeight - deltaY, 40, maxHeight);
  bottomPanel.style.maxHeight = `${targetHeight}px`;
  const shouldCollapse = targetHeight <= 40;
  bottomPanel.classList.toggle("collapsed", shouldCollapse);
  spectrumCanvas.classList.toggle("panel-collapsed", shouldCollapse);
  gestureStatus.classList.toggle("panel-collapsed", shouldCollapse);
  helpButton.classList.toggle("panel-collapsed", shouldCollapse);
  panelLabel.textContent = shouldCollapse ? "展開面板" : "收起面板";
  event.preventDefault();
}

function endPanelDrag(event) {
  if (!panelDragState.active || !bottomPanel) return;
  panelDragState.active = false;
  // Calculate final state based on where drag ended
  const clientY = event.clientY !== undefined ? event.clientY : event.touches?.[0]?.clientY;
  let isNowCollapsed = false;

  if (typeof clientY === "number") {
    const deltaY = clientY - panelDragState.startY;
    const maxHeight = Math.max(360, window.innerHeight - 80);
    const targetHeight = clamp(panelDragState.startHeight - deltaY, 40, maxHeight);
    isNowCollapsed = targetHeight <= 40;
  } else {
    // Fallback to measuring if we can't get clientY
    const height = bottomPanel.getBoundingClientRect().height;
    isNowCollapsed = height <= 40;
  }

  // Update state for future reference
  isPanelCollapsed = isNowCollapsed;

  // Apply transition and update UI
  bottomPanel.style.transition = "max-height 0.35s cubic-bezier(0.4, 0, 0.2, 1), padding 0.35s ease";
  applyPanelCollapseState();

  if (event?.pointerId && panelHandle.releasePointerCapture) {
    panelHandle.releasePointerCapture(event.pointerId);
  }
}

function handlePanelClick() {
  if (panelDragState.moved) {
    panelDragState.moved = false;
    return;
  }
  isPanelCollapsed = !isPanelCollapsed;
  applyPanelCollapseState();
}

if (centerHUD) {
  centerHUD.addEventListener("pointerdown", startCenterDrag, { passive: false });
  centerHUD.addEventListener("mousedown", startCenterDrag, { passive: false });
  window.addEventListener("pointermove", moveCenterDrag, { passive: false });
  window.addEventListener("mousemove", moveCenterDrag, { passive: false });
  window.addEventListener("pointerup", endCenterDrag);
  window.addEventListener("mouseup", endCenterDrag);
  window.addEventListener("pointercancel", endCenterDrag);
  centerHUD.addEventListener("touchstart", startCenterDrag, { passive: false });
  window.addEventListener("touchmove", moveCenterDrag, { passive: false });
  window.addEventListener("touchend", endCenterDrag);
  window.addEventListener("touchcancel", endCenterDrag);
  window.addEventListener("resize", updateCenterHUDPosition);
  updateCenterHUDPosition();
}

if (panelHandle) {
  panelHandle.addEventListener("pointerdown", startPanelDrag, { passive: false });
  panelHandle.addEventListener("mousedown", startPanelDrag, { passive: false });
  window.addEventListener("pointermove", movePanelDrag, { passive: false });
  window.addEventListener("mousemove", movePanelDrag, { passive: false });
  window.addEventListener("pointerup", endPanelDrag);
  window.addEventListener("mouseup", endPanelDrag);
  window.addEventListener("pointercancel", endPanelDrag);
  panelHandle.addEventListener("touchstart", startPanelDrag, { passive: false });
  window.addEventListener("touchmove", movePanelDrag, { passive: false });
  window.addEventListener("touchend", endPanelDrag);
  window.addEventListener("touchcancel", endPanelDrag);
  panelHandle.addEventListener("click", handlePanelClick);
}

function applyPanelCollapseState(setHeight = true) {
  bottomPanel.classList.toggle("collapsed", isPanelCollapsed);
  spectrumCanvas.classList.toggle("panel-collapsed", isPanelCollapsed);
  gestureStatus.classList.toggle("panel-collapsed", isPanelCollapsed);
  helpButton.classList.toggle("panel-collapsed", isPanelCollapsed);
  // Only set height if not currently dragging (to avoid overriding user's drag position)
  if (setHeight && !panelDragState.active) {
    if (isPanelCollapsed) {
      bottomPanel.style.maxHeight = "40px"; // Match the collapse threshold used during drag
    } else {
      bottomPanel.style.maxHeight = "";
    }
  }
  panelLabel.textContent = isPanelCollapsed ? "展開面板" : "收起面板";
}

applyPanelCollapseState();


// ═══════════════════════════════════════════════════════════════
//  RESIZE
// ═══════════════════════════════════════════════════════════════

function resizeCanvas() {
  canvasEl.width = window.innerWidth;
  canvasEl.height = window.innerHeight;
}

// ═══════════════════════════════════════════════════════════════
//  MAIN LOOP
// ═══════════════════════════════════════════════════════════════

async function main() {
  await setupCamera();
  resizeCanvas();
  window.addEventListener("resize", resizeCanvas);

  const handLandmarker = await setupHandLandmarker();
  let lastVideoTime = -1;
  let cachedLeftLandmarks = null;
  let cachedRightLandmarks = null;
  let lastFrameResults = { landmarks: [] };

  function loop() {
    const timestampNow = performance.now();

    try {
      // ── 1. MediaPipe Update ──
      const shouldDetect = timestampNow - lastDetectionTimestamp >= DETECTION_INTERVAL_MS;
      if (shouldDetect && videoEl.readyState >= 2) {
        lastDetectionTimestamp = timestampNow;
        try {
          const results = handLandmarker.detectForVideo(videoEl, timestampNow);
          if (results) lastFrameResults = results;

          cachedLeftLandmarks = null;
          cachedRightLandmarks = null;

          if (results && Array.isArray(results.landmarks)) {
            results.landmarks.forEach((landmarks, i) => {
              const handedness = results.handedness?.[i]?.[0]?.categoryName;
              if (handedness === "Left") cachedLeftLandmarks = landmarks;
              if (handedness === "Right") cachedRightLandmarks = landmarks;
            });
          }
        } catch (err) {
          console.warn("HandLandmarker detection error:", err);
          lastFrameResults = { landmarks: [] };
          cachedLeftLandmarks = null;
          cachedRightLandmarks = null;
        }
      }

      if (videoEl.readyState >= 2) {
        drawFrame(lastFrameResults, canvasEl.width, canvasEl.height);
      }

      // ── 2. Gesture Status ──
      leftHandInd.classList.toggle("active", !!cachedLeftLandmarks);
      rightHandInd.classList.toggle("active", !!cachedRightLandmarks);

      // ── 3. Raw Gesture States ──
      let rawChord = null;
      let rawMode = true;
      let rawQualityIndex = 0;
      let rawThumbDown = false;
      let rawPalmOpenness = 0;
      let rawFingerSpread = 0;
      let rawWristRotation = 0;

      if (cachedLeftLandmarks) {
        const leftTilt = getHandHorizontalTilt(cachedLeftLandmarks, "Left");
        rawChord = classifyChord(cachedLeftLandmarks, "Left");
        rawMode = leftTilt >= 0;
      }

      if (cachedRightLandmarks) {
        rawQualityIndex = getRightHandQualityIndex(cachedRightLandmarks);
        rawThumbDown = isThumbExtended(cachedRightLandmarks, "Right");
        rawPalmOpenness = getPalmOpenness(cachedRightLandmarks);
        rawFingerSpread = getFingerSpread(cachedRightLandmarks);
        rawWristRotation = getWristRotation(cachedRightLandmarks);
      }

      let rawChordState = null;
      if (rawChord) {
        rawChordState = {
          chord: rawChord,
          isMajorMode: rawMode,
          qualityIndex: rawQualityIndex,
          thumbDown: rawThumbDown,
          palmOpenness: rawPalmOpenness,
          fingerSpread: rawFingerSpread,
          wristRotation: rawWristRotation
        };
      }

      // ── 4. Stabilize ──
      const stable = stabilizeChordState(rawChordState, timestampNow);
      let currentChord = null;
      let isMajorMode = true;
      let qualityIndex = 0;
      let thumbDown = false;
      let palmOpenness = 0;
      let fingerSpread = 0;
      let wristRotation = 0;

      if (stable) {
        currentChord = stable.chord;
        isMajorMode = stable.isMajorMode;
        qualityIndex = stable.qualityIndex;
        thumbDown = stable.thumbDown;
        palmOpenness = stable.palmOpenness;
        fingerSpread = stable.fingerSpread;
        wristRotation = stable.wristRotation;
      }

      // ── 5. UI Update ──
      if (currentChord) {
        const chordName = getChordName(currentChord, isMajorMode);
        chordDisplayEl.textContent = `${chordName}(${currentChord})`;
      } else {
        chordDisplayEl.textContent = "--";
      }

      const MAJOR_LABELS = { 1: "Major", 2: "Major 1st Inv", 3: "Major 7th", 4: "Dominant 7th" };
      const MINOR_LABELS = { 1: "Minor", 2: "Minor 1st Inv", 3: "Minor 7th", 4: "Diminished 7th" };
      const activeLabel = isMajorMode ? MAJOR_LABELS[qualityIndex] : MINOR_LABELS[qualityIndex];
      qualityDisplayEl.textContent = activeLabel ? `${activeLabel}${thumbDown ? " (-8ve)" : ""}` : "--";

      // ── 6. Audio Engine ──
      if (engine.initialized && cachedRightLandmarks) {
        try {
          const currentVolume = getVolumeFromHeight(cachedRightLandmarks);
          updateVolumeMeter(currentVolume);

          const horizontalTilt = getHandHorizontalTilt(cachedRightLandmarks, "Right");
          const tiltPercentage = Math.round(horizontalTilt * 100);
          document.getElementById("distortionDisplay").textContent = `Filter: ${tiltPercentage > 0 ? "+" : ""}${tiltPercentage}%`;

          engine.updateFilterFromTilt(horizontalTilt);

          // Map new gestures to effects
          engine.updateBitcrusher(rawPalmOpenness); // Palm openness -> bitcrush amount
          engine.updateFlanger(rawFingerSpread);    // Finger spread -> flanger depth
          engine.updateAutoFilter(rawWristRotation); // Wrist rotation -> auto filter amount

          if (currentChord && qualityIndex >= 1) {
            const tones = getChordTones(currentChord, isMajorMode);
            let notes = getSolidNotes(tones, qualityIndex, isMajorMode);
            if (thumbDown) notes = notes.map(f => f / 2);

            // Lead + Pad + Bass
            if (!isArpOn) {
              engine.playLeadNotes(notes, currentVolume);
            }
            engine.playPadNotes(notes, currentVolume * 0.6);
            if (tones) engine.playBassNote(tones.root, currentVolume);

            // Arpeggiator
            engine.setArpNotes(notes);

            // Looper recording
            if (isRecording && isLooperOn) {
              loopEvents.push({
                time: Tone.now(),
                notes: [...notes],
                volume: currentVolume,
                chord: currentChord,
                qualityIndex,
                isMajorMode,
                thumbDown
              });
            }
          } else {
            engine.stopLead();
            engine.stopBass();
            engine.stopPad();
            engine.setArpNotes([]);
          }
        } catch (err) {
          console.warn("Audio update error:", err);
          engine.stopLead();
          engine.stopBass();
          engine.stopPad();
          engine.setArpNotes([]);
        }
      } else if (engine.initialized) {
        engine.stopLead();
        engine.stopBass();
        engine.stopPad();
        engine.setArpNotes([]);
        updateVolumeMeter(0);
      }

      // ── 7. Visual Energy ──
      const volume = cachedRightLandmarks ? getVolumeFromHeight(cachedRightLandmarks) : 0;
      const tilt = cachedRightLandmarks ? getHandHorizontalTilt(cachedRightLandmarks, "Right") : 0;
      drawEnergy(ctx, volume, qualityIndex, tilt, currentChord);

      // ── 8. Spectrum Visualizer ──
      drawSpectrum();
    } catch (err) {
      console.error("Main loop failure:", err);
    } finally {
      requestAnimationFrame(loop);
    }
  }

  loop();
}

main().catch((err) => console.error(err));
#!/usr/bin/env python3
"""
Audio Peak Detection Script using Librosa
Detects audio onsets/peaks and exports timestamps to YAML format
"""

import librosa
import numpy as np
import yaml
import argparse
import os
from pathlib import Path

def detect_peaks(audio_file, method='default', sensitivity=0.5, ticks_per_second=1000):
    """
    Detect peaks in audio file using librosa
    
    Args:
        audio_file (str): Path to audio file
        method (str): Detection method ('default', 'energy', 'spectral_flux', 'complex_domain')
        sensitivity (float): Detection sensitivity (0.0 - 1.0)
        ticks_per_second (int): Time resolution in ticks per second
    
    Returns:
        list: Peak timestamps in ticks
    """
    
    print(f"Loading audio file: {audio_file}")
    
    # Load audio file
    y, sr = librosa.load(audio_file)
    duration = len(y) / sr
    
    print(f"Audio duration: {duration:.2f} seconds")
    print(f"Sample rate: {sr} Hz")
    print(f"Audio RMS level: {np.sqrt(np.mean(y**2)):.4f}")
    
    # Choose onset detection method
    if method == 'energy':
        # Use RMS energy for onset detection
        rms = librosa.feature.rms(y=y)[0]
        onset_envelope = np.diff(rms, prepend=rms[0])
        onset_envelope = np.maximum(0, onset_envelope)  # Only positive changes
    elif method == 'spectral_flux':
        # Use spectral flux
        stft = librosa.stft(y)
        onset_envelope = librosa.onset.onset_strength(S=np.abs(stft), sr=sr)
    elif method == 'complex_domain':
        # Use complex domain method
        onset_envelope = librosa.onset.onset_strength(y=y, sr=sr, feature=librosa.feature.chroma_stft)
    else:  # default
        onset_envelope = librosa.onset.onset_strength(y=y, sr=sr)
    
    print(f"Onset envelope max: {np.max(onset_envelope):.4f}")
    print(f"Onset envelope mean: {np.mean(onset_envelope):.4f}")
    
    # Convert sensitivity (0-1) to actual delta value
    # For classical music, we need much higher thresholds to avoid micro-variations
    max_envelope = np.max(onset_envelope)
    mean_envelope = np.mean(onset_envelope)
    
    if max_envelope > 0:
        # Much more conservative scaling for classical music
        # sensitivity 0.1 should give us roughly the right number of peaks
        delta = mean_envelope + (max_envelope - mean_envelope) * (sensitivity * 2 + 0.3)
    else:
        delta = 0.1
    
    print(f"Using delta threshold: {delta:.4f}")
    
    # Detect onset frames with conservative parameters for classical music
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_envelope,
        sr=sr,
        units='time',
        delta=delta,
        pre_max=3,      # Restored conservative values
        post_max=3,       
        pre_avg=3,      
        post_avg=5,     
        wait=20         # Increased wait time to avoid too many close peaks
    )
    
    print(f"Initial detection: {len(onset_frames)} peaks")
    
    # Additional filtering: remove peaks that are too close together
    # For classical music, we rarely want peaks closer than ~0.3 seconds
    min_gap = 0.2  # Minimum 200ms between peaks
    if len(onset_frames) > 1:
        filtered_onsets = [onset_frames[0]]  # Keep first peak
        for onset in onset_frames[1:]:
            if onset - filtered_onsets[-1] >= min_gap:
                filtered_onsets.append(onset)
        onset_frames = np.array(filtered_onsets)
    
    print(f"After filtering (min {min_gap}s gap): {len(onset_frames)} peaks")
    
    # If still no peaks, try alternative approach
    if len(onset_frames) == 0:
        print("No peaks detected with onset method, trying peak picking on envelope...")
        
        # Use peak picking on the onset envelope directly
        peaks = librosa.util.peak_pick(
            onset_envelope,
            pre_max=3,
            post_max=3,
            pre_avg=3,
            post_avg=5,
            delta=np.max(onset_envelope) * 0.01,  # Very low threshold
            wait=10
        )
        
        # Convert frame indices to time
        times = librosa.frames_to_time(peaks, sr=sr)
        onset_frames = times
        print(f"Peak picking found {len(onset_frames)} peaks")
    
    # Convert to ticks
    peak_ticks = []
    for onset_time in onset_frames:
        tick = int(onset_time * ticks_per_second)
        peak_ticks.append(tick)
    
    # Add end of file
    end_tick = int(duration * ticks_per_second)
    peak_ticks.append(end_tick)
    
    return peak_ticks, duration

def export_to_yaml(peaks, duration, output_file, metadata=None):
    """
    Export peak data to YAML file
    
    Args:
        peaks (list): Peak timestamps in ticks
        duration (float): Total audio duration in seconds
        output_file (str): Output YAML file path
        metadata (dict): Additional metadata to include
    """
    
    data = {
        'audio_analysis': {
            'total_peaks': len(peaks) - 1,  # Subtract 1 for end marker
            'duration_seconds': round(duration, 3),
            'duration_ticks': peaks[-1],
            'ticks_per_second': metadata.get('ticks_per_second', 1000) if metadata else 1000,
            'detection_method': metadata.get('method', 'default') if metadata else 'default',
            'sensitivity': metadata.get('sensitivity', 0.5) if metadata else 0.5
        },
        'peaks': []
    }
    
    # Add peak entries
    for i, tick in enumerate(peaks[:-1]):  # Exclude the last entry (end marker)
        peak_data = {
            'index': i,
            'tick': tick,
            'time_seconds': round(tick / (metadata.get('ticks_per_second', 1000) if metadata else 1000), 3)
        }
        data['peaks'].append(peak_data)
    
    # Add end marker
    data['end_of_file'] = {
        'tick': peaks[-1],
        'time_seconds': round(duration, 3)
    }
    
    # Write YAML file
    with open(output_file, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, indent=2)
    
    print(f"Results exported to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Detect audio peaks and export to YAML')
    parser.add_argument('input_file', help='Input audio file path')
    parser.add_argument('-o', '--output', help='Output YAML file (default: input_name_peaks.yaml)')
    parser.add_argument('-m', '--method', choices=['default', 'energy', 'spectral_flux', 'complex_domain'],
                       default='default', help='Peak detection method')
    parser.add_argument('-s', '--sensitivity', type=float, default=0.5,
                       help='Detection sensitivity (0.0-1.0, default: 0.5)')
    parser.add_argument('-t', '--ticks-per-second', type=int, default=1000,
                       help='Time resolution in ticks per second (default: 1000)')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found")
        return
    
    # Generate output filename if not provided
    if args.output:
        output_file = args.output
    else:
        input_path = Path(args.input_file)
        output_file = input_path.with_suffix('.yaml').name.replace(input_path.suffix.replace('.', '_'), '_peaks')
    
    try:
        # Detect peaks
        peaks, duration = detect_peaks(
            args.input_file,
            method=args.method,
            sensitivity=args.sensitivity,
            ticks_per_second=args.ticks_per_second
        )
        
        # Prepare metadata
        metadata = {
            'method': args.method,
            'sensitivity': args.sensitivity,
            'ticks_per_second': args.ticks_per_second,
            'source_file': args.input_file
        }
        
        # Export to YAML
        export_to_yaml(peaks, duration, output_file, metadata)
        
        print(f"\nSummary:")
        print(f"  Input: {args.input_file}")
        print(f"  Output: {output_file}")
        print(f"  Peaks detected: {len(peaks) - 1}")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Resolution: {args.ticks_per_second} ticks/second")
        
    except Exception as e:
        print(f"Error processing audio: {e}")

if __name__ == "__main__":
    main()
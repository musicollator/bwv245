#!/usr/bin/env python3
"""
Visualize audio with detected peaks overlaid
"""

import librosa
import numpy as np
import matplotlib.pyplot as plt
import yaml
import argparse
from pathlib import Path

def load_peaks_from_yaml(yaml_file):
    """Load peak data from YAML file"""
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    
    # Extract peak times in seconds
    peak_times = []
    if 'peaks' in data:
        for peak in data['peaks']:
            peak_times.append(peak['time_seconds'])
    
    # Add end time if available
    if 'end_of_file' in data:
        end_time = data['end_of_file']['time_seconds']
    else:
        end_time = None
    
    metadata = data.get('audio_analysis', {})
    
    return peak_times, end_time, metadata

def plot_audio_with_peaks(audio_file, yaml_file, output_plot=None):
    """Plot audio waveform with detected peaks"""
    
    print(f"Loading audio: {audio_file}")
    print(f"Loading peaks: {yaml_file}")
    
    # Load audio
    y, sr = librosa.load(audio_file)
    duration = len(y) / sr
    time = np.linspace(0, duration, len(y))
    
    # Load peaks
    peak_times, end_time, metadata = load_peaks_from_yaml(yaml_file)
    
    print(f"Audio duration: {duration:.2f}s")
    print(f"Peaks to plot: {len(peak_times)}")
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot 1: Full waveform with peaks
    ax1.plot(time, y, color='lightblue', alpha=0.7, linewidth=0.5)
    ax1.set_title(f'Audio Waveform with Detected Peaks\n{Path(audio_file).name}')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)
    
    # Add peak markers
    for i, peak_time in enumerate(peak_times):
        ax1.axvline(x=peak_time, color='red', alpha=0.8, linewidth=1.5)
        # Add peak numbers for first 20 peaks to avoid clutter
        if i < 20:
            ax1.text(peak_time, ax1.get_ylim()[1] * 0.9, str(i), 
                    rotation=90, fontsize=8, ha='right', va='top')
    
    # Add end marker
    if end_time:
        ax1.axvline(x=end_time, color='green', alpha=0.8, linewidth=2, 
                   linestyle='--', label='End of file')
    
    # Plot 2: Onset strength function (if we can recreate it)
    try:
        onset_envelope = librosa.onset.onset_strength(y=y, sr=sr)
        times_envelope = librosa.frames_to_time(np.arange(len(onset_envelope)), sr=sr)
        
        ax2.plot(times_envelope, onset_envelope, color='darkblue', linewidth=1)
        ax2.set_title('Onset Strength Function')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Onset Strength')
        ax2.grid(True, alpha=0.3)
        
        # Add peak markers to onset plot too
        for peak_time in peak_times:
            ax2.axvline(x=peak_time, color='red', alpha=0.6, linewidth=1)
            
        # Show detection threshold if available
        if 'detection_method' in metadata:
            method = metadata['detection_method']
            ax2.set_title(f'Onset Strength Function (Method: {method})')
            
    except Exception as e:
        print(f"Could not plot onset strength: {e}")
        ax2.text(0.5, 0.5, 'Onset strength plot unavailable', 
                transform=ax2.transAxes, ha='center', va='center')
        ax2.set_title('Onset Strength Function (unavailable)')
    
    # Add metadata text
    info_text = f"Peaks detected: {len(peak_times)}\n"
    if 'detection_method' in metadata:
        info_text += f"Method: {metadata['detection_method']}\n"
    if 'sensitivity' in metadata:
        info_text += f"Sensitivity: {metadata['sensitivity']}\n"
    if 'ticks_per_second' in metadata:
        info_text += f"Resolution: {metadata['ticks_per_second']} ticks/sec"
    
    fig.text(0.02, 0.02, info_text, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save or show plot
    if output_plot:
        plt.savefig(output_plot, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_plot}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize audio with detected peaks')
    parser.add_argument('audio_file', help='Input audio file')
    parser.add_argument('yaml_file', help='YAML file with detected peaks')
    parser.add_argument('-o', '--output', help='Output plot file (PNG/PDF)')
    
    args = parser.parse_args()
    
    # Validate files exist
    if not Path(args.audio_file).exists():
        print(f"Error: Audio file '{args.audio_file}' not found")
        return
    
    if not Path(args.yaml_file).exists():
        print(f"Error: YAML file '{args.yaml_file}' not found")
        return
    
    try:
        plot_audio_with_peaks(args.audio_file, args.yaml_file, args.output)
    except Exception as e:
        print(f"Error creating plot: {e}")

if __name__ == "__main__":
    main()
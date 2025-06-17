
```
conda activate unified-env    
python ../bwv-zeug/audio/fermata_chopper.py \
  -i exports/bwv245.wav \
  --energy-percentile 10 \
  --stability-percentile 90 \
  --min-duration 0.1 \
  -o segments \
  --plot

python ../bwv-zeug/audio/add_clicks.py segments/ --clean
python ../bwv-zeug/python/sync_with_audio.py bwv245_note_heads.csv exports/bwv245.yaml detected_beats.yaml -o exports/bwv245_audio_sync_final.yaml
python ../bwv-zeug/audio/visualize_beats.py --audio-dir segments --beats-yaml detected_beats.yaml --yaml-timing exports/bwv245_audio_sync_final.yaml
conda deactivate
```
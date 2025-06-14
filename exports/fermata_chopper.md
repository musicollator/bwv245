Here's the updated script with both detection methods! It will automatically choose the right method based on your parameters.

## Option 2: Simple Energy Method (what you wanted)
```bash
# Find regions below 10th percentile energy
python fermata_chopper.py -i chorale.wav \
  --energy-percentile 10 \
  --stability-percentile 90 \
  --min-duration 0.1 \
  --plot \
  -o segments
```

## More examples for the simple energy method:
```bash
# Very low energy threshold (should catch all valleys)
python fermata_chopper.py -i chorale.wav \
  --energy-percentile 5 \
  --stability-percentile 95 \
  --min-duration 0.05 \
  --min-gap 1.0 \
  --plot \
  -o segments

# Medium sensitivity
python fermata_chopper.py -i chorale.wav \
  --energy-percentile 15 \
  --stability-percentile 80 \
  --min-duration 0.2 \
  --plot \
  -o segments

```

## Key Parameters for Simple Energy Method:

**For detection sensitivity:**
- `--energy-percentile`: Lower = detects higher energy as "low" (try 5-20 to catch obvious valleys)
- `--stability-percentile`: Higher = ignores harmony requirements (try 80-95 to focus only on energy)
- `--min-duration`: Minimum time for a low energy region (try 0.05-0.5 seconds)

**The script automatically chooses:**
- **Simple Energy Method** when you specify `--energy-percentile` 
- **Energy Drop Method** when you specify `--drop-threshold`

Try the first example - it should catch those obvious energy valleys at 5s, 12s, 29s, 42s, and 57s!#!/usr/bin/env python3

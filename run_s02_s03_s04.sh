#!/bin/bash
# Chain S02 → S03 → S04 automatically

set -e

# cd ~/Mimic-Synth (Commented out to use the tool's workdir)

echo "=== S02: Starting capture (will resume if prior data exists) ==="
source /home/sanss/miniconda/bin/activate mimic-synth
cd s02_capture
python capture_v1_2.py

if [ $? -ne 0 ]; then
    echo "❌ S02 capture failed"
    exit 1
fi

echo ""
echo "=== S02 complete. Data saved to s02_capture/data/ ==="
echo ""

echo "=== S03: Building dataset ==="
cd ../
python -m s03_dataset.build_dataset --profile s01_profiles/obxf.yaml --m 10 --out s03_dataset/data/

if [ $? -ne 0 ]; then
    echo "❌ S03 dataset build failed"
    exit 1
fi

echo ""
echo "=== S03 complete. Dataset manifest written to s03_dataset/data/manifest.yaml ==="
echo ""

echo "=== S04: Computing EnCodec embeddings on 4090 ==="
python -m s04_embed.index_dataset --dataset s02_capture/data/ --out s04_embed/data/ --pool mean --batch-size 64

if [ $? -ne 0 ]; then
    echo "❌ S04 embedding failed"
    exit 1
fi

echo ""
echo "=== S04 complete. Embeddings written to s04_embed/data/encodec_embeddings.npy ==="
echo ""

echo "✅ Pipeline S02 → S03 → S04 complete!"

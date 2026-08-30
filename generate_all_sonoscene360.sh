SONOSCENE360_ROOT=/path/to/SonoScene360

METADATA_FILE=$SONOSCENE360_ROOT/data/metadata.json

ALLSCENES=$(jq -r '.scene_names | join(" ")' $METADATA_FILE)

for SCENE in $ALLSCENES; do
  echo "Generating scene: $SCENE"
  python generate.py \
    --scene_root outputs/sonoscene360/$SCENE \
    --config configs/default.yaml \
    --input_panorama $SONOSCENE360_ROOT/data/$SCENE/images/panorama_outpainted.jpg \
    --known_sources $SONOSCENE360_ROOT/data/$SCENE/metadata/known_sources.json \
    --resume
done

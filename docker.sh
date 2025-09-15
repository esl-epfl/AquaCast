docker run --rm -it --gpus all \
  --shm-size=8g   \
  -v "$(pwd)/logs:/workspace/logs"   \
  -v "$(pwd)/checkpoints:/workspace/checkpoints"   \
  -v "$(pwd)/test_results:/workspace/test_results"   \
  -v "$(pwd)/Data:/workspace/Data"   \
  synth:latest
.PHONY: install lint test integration api bench clean

install:
	python -m pip install -e ".[onnx,dev]"

lint:
	ruff check src tests scripts

test:
	pytest

# Requires VOXORA_INTEGRATION_MODELS pointing at a populated models directory
integration:
	pytest -m integration -v

api:
	voxora-api --models-dir $(VOXORA_MODELS_DIR)

bench:
	voxora run --engine sensevoice --audio-dir data/fixtures -o /tmp/sensevoice.json

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .coverage htmlcov
	find . -name __pycache__ -type d -exec rm -rf {} +

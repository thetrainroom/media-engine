.PHONY: install-mlx install-cuda install-cpu install-diarization install-dev install-qwen test lint

# Platform-specific installation (pick one)
install-mlx:
	pip install -e ".[mlx]"

install-cuda:
	pip install -e ".[cuda]"

install-cpu:
	pip install -e ".[cpu]"

# Speaker diarization (optional, run AFTER your platform target)
# pyannote-audio pins an older torch version, so we install it first
# then restore torch to the latest version to match the rest of the stack
install-diarization:
	pip install pyannote-audio>=4.0.0
	pip install --upgrade torch torchaudio torchvision

# Visual language model descriptions
install-qwen:
	pip install -e ".[qwen]"

# Development tools
install-dev:
	pip install -e ".[dev]"

# Run tests
test:
	pytest tests/

# Lint and type check
lint:
	ruff check media_engine/
	pyright media_engine/

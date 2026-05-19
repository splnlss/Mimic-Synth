CONDA = conda run -n mimic-synth --no-capture-output --cwd $(shell pwd)

.PHONY: install capture build embed train invert \
        verify-dataset verify-embeddings verify-surrogate \
        test test-unit test-integration

install:
	pip install -e ".[dev]"

capture:
	$(CONDA) mimic-capture

build:
	$(CONDA) mimic-build

embed:
	$(CONDA) mimic-embed --pool mean --batch-size 64

train:
	$(CONDA) mimic-train

invert:
	$(CONDA) mimic-invert --target $(TARGET)

verify-dataset:
	$(CONDA) mimic-verify-dataset

verify-embeddings:
	$(CONDA) mimic-verify-embeddings

verify-surrogate:
	$(CONDA) mimic-verify-surrogate

test:
	.venv/bin/pytest tests/unit -m "not integration" -v

test-integration:
	$(CONDA) python -m pytest tests/integration -m integration -v

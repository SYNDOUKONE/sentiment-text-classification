.PHONY: install finetune train test serve
install:
	pip install -r requirements.txt
finetune:   ## ~1 h on 2 CPU cores; optional
	python -m stc.finetune --epochs 3
train:      ## EDA + all models + exports (~5 min)
	python -m stc.train
test:
	python -m pytest -q tests
serve:
	cd docs && python -m http.server 8000

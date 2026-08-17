.PHONY: install collect data features train index eval serve demo test stage-artifacts docker-build docker-run clean

install:
	pip install -r requirements.txt

collect:
	python -m src.collect --seeds $(SEEDS) --max-users $(or $(MAX_USERS),3000)

data:
	python -m src.data.build_matrix

features:
	python -m src.features.artist_features

train:
	python -m src.model.train

index:
	python -m src.serve.build_index

eval:
	python -m src.eval.evaluate

serve:
	uvicorn src.serve.api:app --host 0.0.0.0 --port 8000

demo:
	streamlit run app/streamlit_app.py

test:
	pytest -q

stage-artifacts:
	python -m scripts.stage_artifacts

docker-build:
	docker build -t two-tower-lastfm .

# Local test of the combined-container image, reusing local data/models via
# volume mounts (skips the HF_MODEL_REPO download -- see start.sh) rather than
# the real deploy path, which fetches from a Hugging Face Hub model repo.
docker-run:
	docker run --rm -p 8000:8000 -p 7860:7860 \
		-v $(CURDIR)/data:/app/data -v $(CURDIR)/models:/app/models \
		two-tower-lastfm

clean:
	rm -rf data/processed/* models/*

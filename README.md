# ChaosRAG: Chaos Theory Assistant
A RAG application that allows you to interact with a knowledge base based on Edward Lorenz's book "The Essence of Chaos"

## Project SetUp

1. Clone the repo
```bash
https://github.com/yanakravrts/ChaosRAG.git
```

2. Environment setup
```bash
1. cd path_to/ChaosRAG

2. poetry config virtualenvs.in-project true

3. poetry install
```

3. Create `.env` file with your api key `GOOGLE_API_KEY`:
```bash
GOOGLE_API_KEY=YourApiKey
```

3. Run application 
```bash
# run in first terminal:
docker-compose up --build

# after you see: URL: http://0.0.0.0:8501, move to db cration
# run in second terminal:
poetry run python src/vector_to_db.py
```

## Pipeline
1. Data Ingestion
![Data Ingestion](assets/data_ingestion.png)

2. Inference
![Inference](assets/inference.png)

## Demo
1. Image as an input 
![Image as an input demo](assets/image.gif)

2. Text queries
a)
![Text 1 demo](assets/text1.gif)

b)
![Text 2 demo](assets/text2.gif)

3. Equation
![Equatin demo](assets/equation.gif)

4. Image as output
![Image as an output demo](assets/image_output.gif)
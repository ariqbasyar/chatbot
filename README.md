# install locally
```
conda env create -f environment.yml
```
or if not working
```
conda create -n chatbot-gemini python pytorch torchvision numpy sentence-transformers faiss openai
```
then run the main.py
```
OPENROUTER_API_KEY=your_api_key_here python main.py
```

# build docker image
```
docker buildx build --platform linux/amd64,linux/arm64 -t ariqbasyar/chatbot-gemini:gemini-2.0-flash-exp --push .
```

# run with docker
```
docker run -it -e OPENROUTER_API_KEY=your_api_key_here ariqbasyar/chatbot-gemini:gemini-2.0-flash-exp
```

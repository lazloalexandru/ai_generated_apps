docker stop crossword-container
docker rm crossword-container
docker run -d --name crossword-container -p 3000:3000 crossword-app

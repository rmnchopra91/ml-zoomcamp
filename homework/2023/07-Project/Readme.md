Seps:
1: FInal Data\
2: Load Data
3: Perform EDA
4: Choose Model
5: Train Model
6: Save model
7: Create virtual enviorment and install [flask, gunicorn, scikit-learn]
8:



Docker related commands
How to create Docker IMage using Dockerfile
    Run: docker build -t term-predict-app:latest .
How to check list of docker images
    Run: Docker images
How to check running Docker containers
    docker ps -a
How to run docker container using docker image
    Run: docker run -it -p 9696:9696 term-predict-app:latest
How to remove running Docker container
    Run: docker rm container-id
How to delete unwanted Docker images
    Run: docker rmi -f inage-id

Steps to push docker image to docker registry:
    1: docker login
    2: docker tag term-predict-app:latest rmnchopra91/term-predict-app:latest
    3: docker push rmnchopra91/term-predict-app:latest

Steps to run the complete project.
prerequisites:
    start docker deamon in your system
Step 1: docker pull rmnchopra91/term-predict-app:latest
Step 2: docker run -it -p 9696:9696 rmnchopra91/term-predict-app:latest
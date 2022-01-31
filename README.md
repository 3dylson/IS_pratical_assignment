## Using Docker

### Building the Docker Container

Open the cmd at the /docker directory and run the following commands:
```
docker build -t is_model .
```

### Running the Docker Container

```
docker run -v $your_path_to_the_docker_container_directory:/app --rm is_modal
```
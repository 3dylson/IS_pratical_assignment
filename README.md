## Using Docker

### Building the Docker Container

```
docker container build -t is_modal $your_path_to_the_docker_container_directory
```

### Running the Docker Container

```
docker run -v $your_path_to_the_docker_container_directory:/app --rm is_modal
```
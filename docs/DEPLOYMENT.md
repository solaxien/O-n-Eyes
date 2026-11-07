# Deployment and availability

The working application is a FastAPI service with one resident diffusion worker. Phone browsers use the server's GPU. Native phone inference is a separate benchmark executable.

The earlier `trycloudflare.com` URL was a temporary session. It depends on the laptop and tunnel and is not an always-on submission deployment. Its prior browser timings remain evidence of those requests, not a current uptime guarantee.

## Reproducible GPU service

On a Linux machine with Docker Compose supporting `gpus`, an NVIDIA driver, and NVIDIA Container Toolkit:

```sh
docker compose -f deploy/compose.yaml up --build -d
docker compose -f deploy/compose.yaml logs -f image-lab
curl --fail http://127.0.0.1:7860/api/ready
```

The image runs as a non-root user. A named volume retains pinned model downloads across restarts. Compose restarts a crashed service and rotates logs. The container readiness check returns non-success during loading or failure. `/api/health` remains available for the UI to display those states. A failed readiness probe marks the container unhealthy; Docker does not automatically restart a live unhealthy container. Inspect the logs and restart it explicitly after correcting the cause.

Keep one Uvicorn worker. Multiple web workers each load a separate model. A process-manager restart must allow the model to load and warm before advertising readiness. The initial download can exceed the healthcheck's 15-minute startup allowance on a slow connection; inspect progress rather than counting that as warm inference latency.

For a persistent public URL, place a managed HTTPS reverse proxy in front of `127.0.0.1:7860` on an available, continuously running GPU machine. Supply DNS and hosting access separately. Do not expose a development tunnel as guaranteed permanent hosting. No new external host was provisioned during this review because the user excluded external GPU use.

The Dockerfile and Compose service are supplied for reproduction. Docker is absent on the current Windows laptop, so container build and runtime verification remain pending. The Python service, readiness behavior, and actual local model are checked independently.

Hugging Face Docker Spaces are another compatible hosting route if later authorized. Such a deployment needs its own hardware and readiness check and does not establish native phone performance. See the [official Docker Spaces guide](https://huggingface.co/docs/hub/spaces-sdks-docker) and [GPU hardware settings](https://huggingface.co/docs/hub/spaces-gpus). No hosted GPU performance is inferred from those specifications.

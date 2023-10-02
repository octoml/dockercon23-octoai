"""HTTP Inference serving interface using sanic."""
import os

import model
from sanic import Request, Sanic, response
from sanic.worker.manager import WorkerManager

WorkerManager.THRESHOLD = 6000

_DEFAULT_PORT = 8000
"""Default port to serve inference on."""

# Load and initialize the model on startup globally, so it can be reused.
model_instance = model.Model()
"""Global instance of the model to serve."""

server = Sanic("server")
"""Global instance of the web server."""


@server.route("/healthcheck", methods=["GET"])
def healthcheck(_: Request) -> response.JSONResponse:
    """Responds to healthcheck requests.

    :param request: the incoming healthcheck request.
    :return: json responding to the healthcheck.
    """
    return response.json({"healthy": "yes"})


@server.route("/predict", methods=["POST"])
def predict(request: Request) -> response.JSONResponse:
    """Responds to inference/prediction requests.

    :param request: the incoming request containing inputs for the model.
    :return: json containing the inference results.
    """
    inputs = request.json
    output = model_instance.predict(inputs)
    return response.json(output)


def main():
    """Entry point for the server."""
    port = int(os.environ.get("SERVING_PORT", _DEFAULT_PORT))
    server.run(host="0.0.0.0", port=port, workers=1)


if __name__ == "__main__":
    main()

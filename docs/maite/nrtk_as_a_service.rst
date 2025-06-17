===========================
NRTK-as-a-Service (NRTKaaS)
===========================

Dependencies
============

In addition to `NRTK installation <../installation.html>`_, the ``uvicorn`` package is required.

Usage
=====

To run the app, open a command prompt and navigate to ``src/nrtk/interop/maite/api/``, then run the command::

    uvicorn app:app --host 127.0.0.1 --port 8000 --reload

This command starts the server with the API accessible at ``https://127.0.0.1:8000``.

if the user needs to run at a different port or address, the NRTK_IP variable in api/AUKUS_app.env must be modified to
match. For the rest of the example we will use the ip address and port specified above.

To invoke the service with ``curl``, use the following command::

    curl -X POST http://127.0.0.1:8000/ -H "Content-Type: application/json" -d '{"key": "value"}'

This command sends a POST request with JSON data ``{"key": "value"}`` to the REST server. Replace
``http://127.0.0.1:8000`` with the appropriate URL if you have specified a different host or port. If successful, you
should receive a response containing a message indicating the success of the operation and the JSON stub.

The expected `JSON schema is detailed here <schema.html>`_.

AUKUS Requirements
==================

In addition to the standard AUKUS Dataset schema, we require that any POST request to the AUKUS endpoint
include a parameter ``nrtk_config`` which will point to a ``.yaml`` file containing the following NRTK specific
parameters

- **gsds** (List[float])
  A list of gsds (pixel/m) where the length of the list is the same as the number of images in the dataset.

- **theta_keys** (List[str])
  The PyBSM parameters to be perturbed

- **thetas** (List[List[float]])
  The values for each of the parameters specified in "thetas"

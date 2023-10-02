# OctoML @ DockerCon23

Documentation live at [https://dockercon23-mu.vercel.app/](https://dockercon23-mu.vercel.app/)

* TODO: change documentation url with the right one.


## Building the mkdocs documentation page

Install the pre-requisites to build mkdocs page:
```bash
python3 -m pip install wheel
python3 -m pip install mkdocs
python3 -m pip install mkdocs-material-extensions
python3 -m pip install pymdown-extensions
python3 -m pip install mkdocs-material
python3 -m pip install mkdocs-macros-plugin
```

Then go in the `octoml` directory and serve the webpage:
```bash
cd octoml
mkdocs serve
```

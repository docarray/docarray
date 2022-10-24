import yaml
import json

with open("./docs/_versions.yml") as versions_yml, open(
    "./docs/_versions.json", "w"
) as versions_json:
    json.dump(yaml.safe_load(versions_yml), versions_json)

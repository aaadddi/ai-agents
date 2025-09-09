import yaml

def get_config():
    try:
        with open("./config.yml", mode="r") as file:
            data = yaml.safe_load(file)
            return data

    except Exception as e:
        raise e
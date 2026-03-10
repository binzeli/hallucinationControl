import yaml

def load_yaml(file_path):
    """
    Load a YAML file and return its content as a Python dictionary.

    :param file_path: Path to the YAML file.
    :return: Parsed content of the YAML file.
    """
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def save_yaml(data, file_path):
    """
    Save a Python dictionary to a YAML file.

    :param data: Python dictionary to save.
    :param file_path: Path to the YAML file.
    """
    with open(file_path, 'w') as file:
        yaml.safe_dump(data, file)
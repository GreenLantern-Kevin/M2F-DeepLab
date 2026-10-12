"""Create a fresh run directory without overwriting an earlier experiment."""
from pathlib import Path


def create_exp_folder():
    root = Path('run/train')
    root.mkdir(parents=True, exist_ok=True)
    number = 0
    while True:
        path = root / ('exp' + (str(number) if number else ''))
        try:
            path.mkdir()
            (path / 'weights').mkdir()
            return str(path), str(path / 'weights')
        except FileExistsError:
            number += 1

import configparser
import logging
import os.path
import platform
import sys

import yaml

settings = {}

logger = logging.getLogger("configuration")


# Source - https://stackoverflow.com/a/7205107
# Posted by andrew cooke, modified by community. See post 'Timeline' for change history
# Retrieved 2026-02-02, License - CC BY-SA 4.0

def merge(a: dict, b: dict, path=None, allow_override: bool = True):
    if path is None:
        path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge(a[key], b[key], path + [str(key)], allow_override)
            else:
                if allow_override:
                    a[key] = b[key]
                elif a[key] != b[key]:
                    raise Exception('Conflict at ' + '.'.join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a


def configure():
    global settings

    uname = platform.uname()
    machine = uname[4] + ('-rpi' if '-rpi-' in uname[2].lower() else '')
    # platform (Linux, etc), machine(aarch, x86_64), node)
    for filename_name in ('platform.' + uname[0], 'machine.' + machine, uname[1]):
        full_filename = f'config/{filename_name.lower()}.yaml'
        if os.path.isfile(full_filename):
            with open(full_filename, 'r') as yaml_file:
                s1 = yaml.safe_load(yaml_file)
                logger.debug("%s yielded %s", full_filename, s1)
                settings = merge(settings, s1)
                logger.debug("settings are now %s", settings)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    print(platform.uname())
    configure()
    print(settings)

import configparser
import logging
import os.path
import platform
import sys

settings = configparser.ConfigParser()

logger = logging.getLogger("configuration")


def configure():
    global settings

    uname = platform.uname()
    machine = uname[4] + ('-rpi' if '-rpi-' in uname[2].lower() else '')
    filenames = []
    # platform (Linux, etc), machine(aarch, x86_64), node)
    for filename_name in ('platform.' + uname[0], 'machine.' + machine, uname[1]):
        full_filename = f'config/{filename_name.lower()}.cfg'
        if os.path.isfile(full_filename):
            filenames.append(full_filename)
    logging.info("reading config from %s", filenames)
    settings.read(filenames)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    print(platform.uname())
    configure()
    settings.write(sys.stdout)

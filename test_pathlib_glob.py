import pathlib

path = pathlib.Path('testing/collected/glob_test')
for g in ('*.jp*', '*[0-9].jp*'):
    print(g)
    for p in path.glob(g):
        print('  ', p)


def a(log = None, **kwargs):
    print(log)


a(log='foo', ** {'log': 'bar'})

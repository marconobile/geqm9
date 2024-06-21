def exists(val):
    return val is not None

def print_tensor(t):
    print(f'mean: {t.mean()}, std: {t.std()} min: {t.min()}, max {t.max()}')


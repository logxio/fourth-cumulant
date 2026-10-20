"""One role palette for the report's figures. Four low-saturation families and no red or green:
blue is ARC's full K=4 chain, gray ARC's augmented K=3 reference and every baseline, iris our scored K=3 chain B,
gold the single accent (the reading a panel points at, and what was removed, compressed or fitted).
"""
from matplotlib.colors import to_hex, to_rgb


def mix(a, b, t):
    a, b = to_rgb(a), to_rgb(b)
    return to_hex([x + (y - x) * t for x, y in zip(a, b)])


# family: (middle-tone stroke/mark, pale fill)
SCHEME = {
    'blue': ('#7896B8', '#E0E9F3'),
    'iris': ('#9A92C0', '#E7E4F2'),
    'gold': ('#C2A060', '#F5EBD7'),
    'gray': ('#9AA0A8', '#EEF0F2'),
}

# per family: (stroke, mid, fill, wash)
HUES = {name: (stroke, mix(stroke, fill, .45), fill, mix(fill, '#FFFFFF', .48)) for name, (stroke, fill) in SCHEME.items()}

ROLE = {
    'k4': 'blue', 'k3': 'gray', 'chainb': 'iris', 'focus': 'gold', 'cut': 'gold', 'fit': 'gold', 'base': 'gray',
}
INK = '#2F3340'
MUTED = '#6F7581'
RULE = GRID = '#E6E8EC'


def hue(role):
    return ROLE.get(role, role)


def dark(role):
    return HUES[hue(role)][0]


def mid(role):
    return HUES[hue(role)][1]


def light(role):
    return HUES[hue(role)][2]

import sys
import signal

from math import log2
from os import popen, get_terminal_size
from time import sleep
from functools import lru_cache


output = ""
class Node:
    def __init__(self, level, nw, ne, se, sw, num_alive_cells, hash):
        self.level = level
        self.nw = nw
        self.ne = ne
        self.se = se
        self.sw = sw
        self.num_alive_cells = num_alive_cells
        self.hash = hash
        return

    def __hash__(self):
        return self.hash

    def __repr__(self):
            return f"Node level={self.level}, {1<<self.level} x {1<<self.level}, percent alive (max 50%): {self.num_alive_cells*100/(1<<self.level)**2}, population: {self.num_alive_cells}"

on = Node(0, None, None, None, None, 1, 1)
off = Node(0, None, None, None, None, 0, 0)

@lru_cache(maxsize=2**24)
def join(nw: Node, ne: Node, se: Node, sw: Node):

    num_alive_cells = nw.num_alive_cells + ne.num_alive_cells + se.num_alive_cells + sw.num_alive_cells
    hash = (
        nw.level + 2 +
            + 5131830419411 * nw.hash + 3758991985019 * ne.hash
            + 8973110871315 * se.hash + 4318490180473 * sw.hash
        ) & ((1 << 63) - 1)
    return Node(nw.level + 1, nw, ne, se, sw, num_alive_cells, hash)


@lru_cache(maxsize=1024)
def get_zero(level):
    return off if level==0 else join(get_zero(level - 1), get_zero(level - 1),
                                 get_zero(level - 1), get_zero(level - 1))


def centre(node: Node):
    zero = get_zero(node.level - 1)
    return join(
        join(zero, zero, zero, node.nw), join(zero, zero, node.ne, zero),
        join(zero, node.se, zero, zero), join(node.sw, zero, zero, zero)
    )

def life(nw: Node, n: Node, ne: Node, e: Node, center: Node, se: Node, s: Node, sw: Node, w: Node):
    outer = sum([subquad.num_alive_cells for subquad in [nw, n, ne, e, se, s, sw, w]])
    return on if (center.num_alive_cells and outer == 2) or outer == 3 else off

def life_4x4(node: Node):
    nwsw = life(node.nw.nw, node.nw.ne, node.ne.nw, node.nw.se, node.nw.sw, node.ne.se, node.se.nw, node.se.ne, node.sw.nw)
    nese = life(node.nw.ne, node.ne.nw, node.ne.ne, node.nw.sw, node.ne.se, node.ne.sw, node.se.ne, node.sw.nw, node.sw.ne)
    sene = life(node.nw.se, node.nw.sw, node.ne.se, node.se.nw, node.se.ne, node.sw.nw, node.se.se, node.se.sw, node.sw.se)
    swnw = life(node.nw.sw, node.ne.se, node.ne.sw, node.se.ne, node.sw.nw, node.sw.ne, node.se.sw, node.sw.se, node.sw.sw)
    return join(nwsw, nese, sene, swnw)


@lru_cache(maxsize=2 ** 24)
def successor(node: Node, j=None):
    """Return the 2**k-1 x 2**k-1 successor, 2**j generations in the future"""
    if all([node.nw is None, node.ne is None, node.se is None, node.sw is None]):  # empty
        return node.nw
    elif node.level == 2:  # base case
        s = life_4x4(node)
    else:
        j = node.level - 2 if j is None else min(j, node.level - 2)
        c1 = successor(join(node.nw.nw, node.nw.ne, node.nw.se, node.nw.sw), j)
        c2 = successor(join(node.nw.ne, node.ne.nw, node.nw.sw, node.ne.se), j)
        c3 = successor(join(node.ne.nw, node.ne.ne, node.ne.se, node.ne.sw), j)
        c4 = successor(join(node.nw.se, node.nw.sw, node.se.nw, node.se.ne), j)
        c5 = successor(join(node.nw.sw, node.ne.se, node.se.ne, node.sw.nw), j)
        c6 = successor(join(node.ne.se, node.ne.sw, node.sw.nw, node.sw.ne), j)
        c7 = successor(join(node.se.nw, node.se.ne, node.se.se, node.se.sw), j)
        c8 = successor(join(node.se.ne, node.sw.nw, node.se.sw, node.sw.se), j)
        c9 = successor(join(node.sw.nw, node.sw.ne, node.sw.se, node.sw.sw), j)

        if j < node.level - 2:
            s = join(
                (join(c1.sw, c2.se, c4.ne, c5.nw)),
                (join(c2.sw, c3.se, c5.ne, c6.nw)),
                (join(c4.sw, c5.se, c7.ne, c8.nw)),
                (join(c5.sw, c6.se, c8.ne, c9.nw)),
            )
        else:
            s = join(
                successor(join(c1, c2, c4, c5), j),
                successor(join(c2, c3, c5, c6), j),
                successor(join(c4, c5, c7, c8), j),
                successor(join(c5, c6, c8, c9), j),
            )
    return s
@lru_cache(maxsize=2**20)

def next_gen(node: Node):
    """Return the 2**k-1 x 2**k-1 successor, 1 generation forward"""
    if node.num_alive_cells ==0: # empty
        return node.nw
    elif node.level == 2:  # base case
        s = life_4x4(node)
    else:
        c1 = next_gen(join(node.nw.nw, node.nw.ne, node.nw.se, node.nw.sw))
        c2 = next_gen(join(node.nw.ne, node.ne.nw, node.nw.sw, node.ne.se))
        c3 = next_gen(join(node.ne.nw, node.ne.ne, node.ne.se, node.ne.sw))
        c4 = next_gen(join(node.nw.se, node.nw.sw, node.se.nw, node.se.ne))
        c5 = next_gen(join(node.nw.sw, node.ne.se, node.se.ne, node.sw.nw))
        c6 = next_gen(join(node.ne.se, node.ne.sw, node.sw.nw, node.sw.ne))
        c7 = next_gen(join(node.se.nw, node.se.ne, node.se.se, node.se.sw))
        c8 = next_gen(join(node.se.ne, node.sw.nw, node.se.sw, node.sw.se))
        c9 = next_gen(join(node.sw.nw, node.sw.ne, node.sw.se, node.sw.sw))

        s = join(
            (join(c1.sw, c2.se, c4.ne, c5.nw)),
            (join(c2.sw, c3.se, c5.ne, c6.nw)),
            (join(c4.sw, c5.se, c7.ne, c8.nw)),
            (join(c5.sw, c6.se, c8.ne, c9.nw)),
        )
    return s

def advance(node: Node, n: int):
    if n==0:
        return node

    # get binary expansion and make sure we've padded enough
    bits = []
    while n > 0:
        bits.append(n & 1)
        n = n >> 1
        node = centre(node) # nest

    for k, bit in enumerate(reversed(bits)):
        j = len(bits) - k  - 1
        if bit:

            node = successor(node, j)
    return node


def ffwd(node: Node, n: int):
    for i in range(n):
        while (node.level < 3 or node.nw.num_alive_cells != node.nw.sw.sw.num_alive_cells or
                node.ne.num_alive_cells != node.ne.se.se.num_alive_cells or
                node.se.num_alive_cells != node.se.ne.ne.num_alive_cells or
                node.sw.num_alive_cells != node.sw.nw.nw.num_alive_cells):
                node = centre(node)
        print(node)
        node = successor(node)
    return node


def expand(node: Node, x=0, y=0, clip=None, level=0):
    if node.num_alive_cells ==0: # quick zero check
        return []

    size = 2 ** node.level
    # bounds check
    if clip is not None:
        if x + size < clip[0] or x > clip[1] or\
           y + size < clip[2] or y > clip[3]:
            return []

    if node.level == level:
        return [((x >> level) - clip[0] + 1, (y >> level) - clip[2] + 1)]

    else:
        # return all points contained inside this node
        offset = size >> 1
        return (
            expand(node.nw, x, y, clip, level)
            + expand(node.ne, x + offset, y, clip, level)
            + expand(node.se, x, y + offset, clip, level)
            + expand(node.sw, x + offset, y + offset, clip, level)
        )

def construct(pts):
    """Turn a list of (x,y) coordinates into a quadtree"""
    # Force start at (0,0)
    min_x = min(*[x for x, y in pts])
    min_y = min(*[y for x, y in pts])
    pattern = {(x - min_x, y - min_y): on for x, y in pts}
    k = 0

    while len(pattern) != 1:
        # bottom-up construction
        next_level = {}
        z = get_zero(k)

        while len(pattern) > 0:
            x, y = next(iter(pattern))
            x, y = x - (x & 1), y - (y & 1)
            # read all 2x2 neighbours, removing from those to work through
            # at least one of these must exist by definition
            a = pattern.pop((x, y), z)
            b = pattern.pop((x + 1, y), z)
            c = pattern.pop((x, y + 1), z)
            d = pattern.pop((x + 1, y + 1), z)
            next_level[x >> 1, y >> 1] = join(a, b, c, d)

        # merge at the next level
        pattern = next_level
        k += 1
    return pattern.popitem()[1]

def million_constructor(points):
    node = construct(points)
    while node.level <= 19:
        node = centre(node)
    return node

def points_center(node: Node, level:int, x=0, y=0):
    print(node, level, x, y)
    if node is None:
        return []
    elif node.num_alive_cells == 0:
        return []
    elif level == 0 and node.num_alive_cells:
        return [(x, y)]
    else:
        print(node.num_alive_cells)
        print(node.nw.num_alive_cells)
        print(node.ne.num_alive_cells)
        print(node.se.num_alive_cells)
        print(node.sw.num_alive_cells)
        return (
            points_center(node.nw, level-1, 0, 0)+
            points_center(node.ne, level-1, 2**(level-1), 0)+
            points_center(node.se, level-1, 2**(level-1), 2**(level-1))+
            points_center(node.sw, level-1, 0, 2**(level-1))
        )



def view_center(node: Node):
    level = 12
    while node.level > level and node is not None:

        node = Node(node.level-1, node.nw.se, node.ne.sw, node.se.nw, node.sw.ne, sum([node.nw.se.num_alive_cells, node.ne.sw.num_alive_cells, node.se.nw.num_alive_cells, node.sw.ne.num_alive_cells]), 1)
        print(node)
    return points_center(node, level)

def handle_sigint(sig, frame):
    print(output)
    exit(0)

def text_life_points(path: str):
    points = []
    with open(path, 'r') as f:
        for y, line in enumerate(f):
            for x, c in enumerate(line):
                if c == "O":
                    points.append((x, y))
    return points

if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_sigint)
    points = text_life_points("glider_gun.txt")
    node = million_constructor(points)

    i=1
    while True:
        output = f"\rGeneration: {i}, {node.__repr__()}\n"
        node = centre(node)
        node = next_gen(node)
        terminal_size = get_terminal_size()
        size = int(log2(min(terminal_size.columns//3, terminal_size.lines-1)))
        alive_cell_points = expand(node=node, clip=[2**19-2**(size-1), 2**19+2**(size-1), 2**19-2**(size-1), 2**19+2**(size-1)])
        for y in range(2**size):
            for x in range(2**size):
                if (x, y) in alive_cell_points:
                    output += " \u2B24 "
                else:
                    output += "   "

            output += "\n"
        output = output.rstrip('\n')
        print("\033[H\033[J", end="")
        print(output, end='')
        sleep(0.1)
        i +=    1

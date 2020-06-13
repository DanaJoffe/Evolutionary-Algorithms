import os
from graphviz import Digraph
import matplotlib.image as mpimg


def create_dot_graph(head):
    tree = head.get_nodes()
    repr = {comp: str(i) for i, comp in enumerate(tree)}

    dot = Digraph(comment='GP')
    # add nodes
    for node, rep in repr.items():
        dot.node(rep, node.symbol)
    for node in tree:
        for child in node.children:
            dot.edge(repr[node], repr[child])
    return str(dot.source)


def plot_tree(head=None, name='result', dot=None):
    if head:
        dot = create_dot_graph(head)
    assert dot is not None
    with open("graph_string.dot", 'w') as f:
        f.write(dot)
    os.system(f"dot -Tpng graph_string.dot -o {name}.png")
    os.remove("graph_string.dot")
    img = mpimg.imread(f'{name}.png')
    # plt.imshow(img)
    # plt.show()
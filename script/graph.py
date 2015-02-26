import pygraphviz
import networkx as nx
from networkx.readwrite import json_graph
import json

with open("node-link.json") as fd:
  nodelink=json.load(fd)
  G=json_graph.node_link_graph(nodelink)
  A = nx.to_agraph(G)
  A.layout('dot', args='-Nfontsize=10 -Nwidth=".2" -Nheight=".2" -Nmargin=0 -Gfontsize=8')
  A.draw('test.png')


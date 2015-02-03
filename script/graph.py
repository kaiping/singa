
'''
import networkx as nx
import matplotlib.pyplot as plt
G = nx.DiGraph()

G.add_node("ROOT")

for i in xrange(5):
  G.add_node("Child_%i" % i)
  G.add_node("Grandchild_%i" % i)
  G.add_node("Greatgrandchild_%i" % i)

  G.add_edge("ROOT", "Child_%i" % i)
  G.add_edge("Child_%i" % i, "Grandchild_%i" % i)
  G.add_edge("Grandchild_%i" % i, "Greatgrandchild_%i" % i)

# write dot file to use with graphviz
# run "dot -Tpng test.dot >test.png"
#nx.write_dot(G,'test.dot')

# same layout using matplotlib with no labels
plt.title("draw_networkx")
pos=nx.graphviz_layout(G,prog='dot')
nx.draw(G,pos,with_labels=True,arrows=True)
plt.savefig('nx_test.png')

import pygraphviz
import networkx
import networkx as nx
G = nx.Graph()
G.add_node("ROOT")
for i in xrange(5):
  G.add_node("Child_%i" % i)
  G.add_node("Grandchild_%i" % i)
  G.add_node("Greatgrandchild_%i" % i)
  G.add_edge("ROOT", "Child_%i" % i)
  G.add_edge("Child_%i" % i, "Grandchild_%i" % i)
  G.add_edge("Grandchild_%i" % i, "Greatgrandchild_%i" % i)

A = nx.to_agraph(G)
A.layout('dot', args='-Nfontsize=10 -Nwidth=".2" -Nheight=".2" -Nmargin=0 -Gfontsize=8')
A.draw('test.png')
'''
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


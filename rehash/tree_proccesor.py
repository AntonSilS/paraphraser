import nltk
from nltk.tree import *
from itertools import permutations
from typing import List

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')


class SearchRule:
    def __init__(self): 
        self.subtrees = []

    def find_subtrees(self, mtree: MultiParentedTree) -> List[MultiParentedTree]:

        for subtree in mtree.subtrees(filter=lambda x: x.label() == "NP" and len(x) >= 3):
            if subtree[0].label() == "NP" and subtree[-1].label() == "NP":
                if all(el.label() == "NP" for el in subtree[::2]) and all(el.label() in [",", "CC"] for el in subtree[1::2]):
                    self.subtrees.append(subtree)

        return self.subtrees


class SubtreeParaphraser:

    def get_np_nodes(self, subtree: MultiParentedTree) -> List[MultiParentedTree]:
        return [node for node in subtree if node.label() == "NP"]

    def get_sep_nodes(self, subtree: MultiParentedTree) -> List[tuple]:
        return [(i, node) for i, node in enumerate(subtree) if node.label() in [",", "CC"]]
    
    def get_all_combination(self, subtree: MultiParentedTree) -> List[MultiParentedTree]:
        nodes_np = self.get_np_nodes(subtree)
        nodes_sep = self.get_sep_nodes(subtree)
        paraph_subtrees = []
        for paraph_node_tuples in [nodes_tuple for nodes_tuple in permutations(nodes_np)]:
            paraph_node_lst = list(paraph_node_tuples)

            for i, node_sep in nodes_sep:
                paraph_node_lst.insert(i, node_sep)
            
            paraph_subtree = MultiParentedTree.convert(Tree("NP", paraph_node_lst))

            paraph_subtrees.append(paraph_subtree)
        
        return paraph_subtrees[1:]

class Paraphraser:
    def __init__ (self, sentence_tree: str, rule: SearchRule, subtree_paraphraser: SubtreeParaphraser):
        self.rule = rule
        self.sentence_tree = sentence_tree 
        self.paraph_trees = []
        self.subtree_paraphraser = subtree_paraphraser
    
    def get_mtree (self) -> MultiParentedTree:
        return MultiParentedTree.fromstring(self.sentence_tree)
    
    def get_subtree_index(self, mtree: MultiParentedTree, subtree: MultiParentedTree) -> List[tuple]:
        return [i for i in subtree.treepositions(mtree)][0]

    def get_all_trees(self) -> List[str]:
        mtree = self.get_mtree()

        for subtree in self.rule.find_subtrees(mtree):
            
            pos_subtree = self.get_subtree_index(mtree, subtree)
            paraphrase_subtrees = self.subtree_paraphraser.get_all_combination(subtree)

            for subtree in paraphrase_subtrees:
                mtree_new = self.get_mtree()
                pos_remove_insert = pos_subtree if len(pos_subtree) == 1 else pos_subtree[:-1]

                if len(pos_remove_insert) == 1:
                     mtree_new.remove(mtree_new[pos_subtree])
                else:
                    mtree_new[pos_subtree[:-1]].remove(mtree_new[pos_subtree])
                mtree_new[pos_remove_insert] = subtree
                self.paraph_trees.append(mtree_new)

        return [' '.join(str(tree).split()) for tree in self.paraph_trees]
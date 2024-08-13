import logging
import re
import random
import hashlib
import base64
import numpy as np
import copy
import pandas as pd
from multiproc import *


def group_labels(df, label_col, name_col):
    grouped = df.groupby(label_col)[name_col].apply(list).to_dict()
    return grouped

class DepthManager: #TRY TO MAKE DEPTHMANAGER SINGLETON 
    _max_depth = 5  # Default maximum depth
    _min_depth = 3
    _instance = None

    @classmethod
    def getInstance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if DepthManager._instance is not None:
            raise Exception("This is a singleton class. Use 'getInstance()'.")
        self.depth = 0  # Starting depth
        self.depth_record = {}

    @classmethod
    def set_max_depth(cls, depth):
        if depth > cls._min_depth:
            cls._max_depth = depth
        else:
            print("Maximum depth cannot be smaller than the min depth! \n The default max_depth is", cls._max_depth)
    
    def reset_depth(self):
        self.depth = 0
    
    # def reset_depth_record(self):
    #     self.depth_record = {}

def depth_control(func):
    def wrapper(*args, **kwargs):
        dm = DepthManager.getInstance()
        if dm.depth == dm._max_depth:
            print("Max depth reached")
            return None
        result = func(*args, **kwargs)
        dm.depth += 1  # Increment depth after function call
        return result
    return wrapper

def class_depth_control(cls):
    class WrappedClass(cls):  # Create a new class that wraps the original class
        def __init__(self, *args, **kwargs):
            dm = DepthManager.getInstance()
            if dm.depth == dm._max_depth:
                print("Max depth reached")
                return None
            super().__init__(*args, **kwargs)
            dm.depth += 1
    return WrappedClass 

class TreeNode:
    depth = 0
    score = 1 #defaultly assume tree as garbage query
   
    def __init__(self, value):
        self.value = value
        self.children = []
        # self.level = 0

    def add_child(self, node):
        """Add a TreeNode or value as a child."""
        # if not isinstance(node, TreeNode):
            # node = TreeNode(node)  # Ensure all children are TreeNode instances
        self.children.append(node)
        # self.level += 1

    def __str__(self):
        # Use the helper method for generating the string with indentation
        return self._str_recursive(level=0)

    def _str_recursive(self,level):
        # Create the string representation with indentation for current node
        ret = "\t" *level + str(self.value) + "\n"  # Indent based on the current level
        for child in self.children:
            ret += child._str_recursive(level+1)
        return ret

    def __repr__(self):
        return f'<TreeNode {self.value}>'
    
    def get_depth(self):
        pass #because it needs later-defined class type

    def to_querystr(self):
        """
        convert the generate query tree into query string with ; separation to get ready for querying the Memgraph client
        """
        child_compose = ''
        final_query_str = 'MATCH'
        for child in self.children:
            if child.children:
                for grandchild in child.children:
                    child_compose = ' '+ str(grandchild.value)
            final_query_str += ' ' + str(child.value) + child_compose
        final_query_str += ';'
        return final_query_str
        
class Clause(TreeNode):
    def __init__(self, value, children=None):
        super().__init__(value)
        self.children = children if children is not None else []

    def __str__(self):
        if not self.children:
            return str(self.value)
        if self.value == "RETURN":
            return f"{self.value} {', '.join(str(child) for child in self.children)}"
        return f"{self.value} {' '.join(str(child) for child in self.children)}"
    
class Node(TreeNode):
    """
    When called, will add connector to nodes and make nodes Node type
    """ 
    def __init__(self, value, children=None):
        super().__init__(value)
        self.children = children if children is not None else []
 
    def __str__(self):
        if not self.children:
            return str(self.value)
        # if self.value == '-':  
        #     return ' '.join(str(child) for child in self.children)
        return f"{self.value}({', '.join(str(child) for child in self.children)})"

class Relationship(TreeNode):
    """
    When called, will add connector to relationships and make relationships Relationship type
    """
    def __init__(self, value, hop_only=False):
        super().__init__(value)
        # self.children = children if children is not None else []
        self.hop_only = True if hop_only else False
        
    @depth_control
    def __str__(self):
        # if not self.children:
            # return str(self.value)
        return f"{self.value}"
        # return f"{self.value} {' '.join(str(child) for child in self.children)}"

    def calculate_individual_depth(self):
        # This method checks for both the presence of a relationship and additional depth from hops
        base_depth = 1 if self.hop_only == False else 0 # Start with a depth of 1 for the relationship itself
        # Look for hop patterns, each '*' adds one to the depth
        hop_matches = re.findall(r'\*', self.value.value)
        return base_depth + len(hop_matches)  # Add one additional depth for each hop pattern

class Condition(TreeNode):
    def __init__(self, value, children=None):
        super().__init__(value)
        self.children = children if children is not None else []

    def __str__(self):
        if not self.children:
            return str(self.value)
        return f"{self.value} {' '.join(str(child) for child in self.children)}"

def get_depth(self):
    # This now maps through children, checking if they are Relationship instances, and sums their depths
    updated_depth = 0
    for child in self.children:
        if isinstance(child, Relationship):
            updated_depth += child.calculate_individual_depth()
        elif isinstance(child,Condition):
            updated_depth += 1
        else:
            updated_depth = updated_depth
    # updated_depth = sum(child.calculate_individual_depth() if isinstance(child, Relationship) else 0 for child in self.children)
    return updated_depth
TreeNode.get_depth = get_depth


class QueryManager:
    def __init__(self, dm):
        self.root = TreeNode("ROOT")  # All parts will be children of root
        self.current_node = self.root  # Current node context for adding parts
        self.node_labels = []
        self.relationships = []
        self.grouped_info = {}
        self.usable_labels = set()
        # self.usable_props = set()

        
        self.parts = []
        self.selected_label_alias = {}

        self.query_str = ''
        
        self.final_depth = 0
        # self.depth_manager = DepthManager()
        # self.id = self.generate_id()
        self.observed_depths = set()
        self.dm = dm
       
    
    @staticmethod
    def generate_id(input_object):
        if isinstance(input_object, TreeNode):
            content = str(input_object)  
        elif isinstance(input_object, str):
            content = input_object
        else:
            raise ValueError("Unsupported object type for ID generation")
        hash_digest = hashlib.sha256(content.encode('utf-8')).digest()
        return base64.urlsafe_b64encode(hash_digest).decode('utf-8').rstrip('=')

        

    def reset_per_gen(self):
        self.root = TreeNode("ROOT")
        self.current_node = self.root  # Reset the tree for new generation
        self.parts = []
        self.selected_label_alias = {}
        self.usable_labels.clear()
        # self.usable_props.clear()
        self.final_depth = 0
        self.query_str = ''

    def import_grouped_info(self, input_group):
        if input_group:
            if type(input_group) is dict:
                self.grouped_info = input_group
                self.node_labels = list(self.grouped_info.keys())
                # print("loaded node_labels:",self.node_labels)
            else:
                print("input grouped info need to be dictionary type")
        else:
            print("input grouped info cannot be empty")
    
    def import_relationships(self, input_relationships):
        if input_relationships:
            self.relationships = input_relationships
        else:
            print("relationships cannot be empty")
    
    def create_unique_alias(self, label):
        """Creates a unique alias for a node to prevent label overlap in queries."""
        base_alias = label.lower()
        alias = base_alias
        counter = 1
        while alias in self.usable_labels:
            alias = f"{base_alias}{counter}"
            counter += 1
        return alias
    
    def extract_alias_label(self, node_value_str):
        # Pattern to capture both the alias and the node label
        pattern = r"^\(([^:]+):([^ {]+)"
        match = re.search(pattern, node_value_str)

        if match:
            alias = match.group(1).strip()  # Get the alias, strip any whitespace
            label = match.group(2).strip()  # Get the label, strip any whitespace
            return alias, label
        else:
            # No match found, return None for both
            return None, None

    def add_node(self):
        """Adds a node with random properties selected from grouped_info."""
        # node_label = ''
        if self.node_labels:
            node_label = random.choice(self.node_labels) 
            possible_props = self.grouped_info[node_label]
            property_label, properties_list = random.choice(list(possible_props.items()))
            alias = self.create_unique_alias(node_label)
            self.selected_label_alias[alias]=node_label

            property_value = random.choice(properties_list)
            properties_str = f'''{property_label}: "{property_value}"''' if possible_props else ''
            node_value = f"{node_label} {{{properties_str}}}"


            node = Node(f"({alias}:{node_value})")
            # self.current_node.add_child(node)

            # self.nodes.append(node)
            self.usable_labels.add(alias)  # Store label for possible RETURN clause usage
            return node 
        print("No node labels available. Please import grouped info first.")
        return None

    def add_hop(self):
        """
        Randomly generate hops as condition to relationship based on a customizable possibility;
        the default possibility is 0.2
        """
        current_depth = self.dm.depth
        hop = random.randint(1,3) #TODO: see if this is reasonable
        upper_hop = hop + random.randint(1,5)
        exact_hop = f"*{hop}"
        ceiling_hop = f"*..{upper_hop}"
        floor_hop = f"*{hop}.."
        interval_hop = f"*{hop}..{upper_hop}"
        hop_choices = [exact_hop, ceiling_hop, floor_hop, interval_hop]
        if current_depth < self.dm._max_depth: #random.random() > hop_p and 
            hop_choice = random.choice(hop_choices)
            return hop_choice
        else:
            return ''



    def add_relationship(self, bi_dir_p=0.3, rev_dir_p=0.5, hop_only_p=0.2, hop_p=0.2):
        """ 
        Randomly generate a relationship between two nodes 
        bi_dir: probability of getting a bidirectional direction
        rev_dir_p: probability of getting a reversed direction
        hop_only_p: probability of getting only hops without specific relationships
        hop_p: probability of getting hops in addition to a specific relationship
        """
        current_depth = self.dm.depth
        rel_type = random.choice(self.relationships)
        if random.random() < bi_dir_p:
            direction1 = "-"
            direction2 = "-"
        if current_depth>=3 and random.random() > rev_dir_p: 
            direction1 = "<-"
            direction2 = "-"
        else:
            direction1 = "-" 
            direction2 = "->"
        # if random.random() > hop_p:
        
        hop_result = Relationship(self.add_hop()) if random.random() > hop_p else ''
        if random.random() > hop_only_p and hop_result:
            relationship = Node(f"{direction1} [{hop_result}] {direction2}")
            return Relationship(relationship, hop_only=True)
        else:
            relationship = Node(f"{direction1} [:{rel_type}{hop_result}] {direction2}")
            return Relationship(relationship)
        # self.current_node.add_child(relationship)
        # return relationship
        # return Relationship(relationship, hop_only)
        
    @depth_control
    def add_condition(self, children_source, where_p=0.5, for_ea=False):
        """
        Randomly generate WHERE clause based on a customizable possibility;
        Will add to a random node as its child (no)
        the default possibility where_p is 0.5;
        currently only accepts and will only generate values that are str type properties
        children_source: list of children. If from tree_population, should be tree.children, or other list type.
        """
        if for_ea == True:
            current_depth = 0 #to make sure as long as WHERE is found in previous query, it will be replaced only by chance and not depth
        else:
            current_depth = self.dm.depth
        if random.random() > where_p and current_depth < self.dm._max_depth:
            
            np_children = np.array(children_source,dtype=object)
            is_node = np.vectorize(lambda x: isinstance(x, Node))
            # Apply the function to the numpy array
            node_checks = is_node(np_children)
            node_idx = np.where(node_checks)
            node_children = np_children[node_idx]
            random_node = random.choice(node_children)

            alias, node_label = self.extract_alias_label(random_node.value)
            # alias, node_label = random.choice(list(self.selected_label_alias.items()))
            # print(alias, node_label)

            # selected_node_label = random.choice(selected_node_labels)
            possible_properties = self.grouped_info[node_label.strip()]
            if possible_properties:
                property_label, properties_list = random.choice(list(possible_properties.items()))
                sample_prop_type = properties_list[0]
                # value = random.randint(20, 50) if isinstance(sample_prop_type, int) else random.choice(properties_list) 
                value = random.choice(properties_list) #TODO: generalize to other data type
            #TODO: customize the int part

                operator = random.choice([">", "<", "=", "<=", ">="]) if isinstance(sample_prop_type, int) else '='
                # grandchild = Condition("WHERE", [Condition(f"{alias}.{property_label} {operator} '{value}'")])
                # random_node.add_child(grandchild)
                # print("Added WHERE clause to node", random_node)
                return Condition("WHERE", [Clause(f'''{alias}.{property_label} {operator} "{value}"''', [])])
            else:
                raise ValueError("No available properties for the label selected:", {node_label})
        return 
    
    @staticmethod
    def is_relationship(part):
        """
        Determine if the given part of a query is a relationship based on containing "[]"
        Ensures that part is a string before checking.
        """
        # pattern = re.compile(r'\[(.*?)\]')
        trying = r"-\s*\[:?([A-Za-z0-9_]+)?(\*\d*(\.\.\d*)?)?\]\s*[-<>]?"
        # Ensure part is a string or bytes-like object
        if isinstance(part,str):
            # if pattern.search(part):
            if re.search(trying, part):
                return True
            else:
                return False
        else:   
            print("input has to be str!")
            return None
        

    
    def get_usable_labels(self):
        return list(self.usable_labels)
    
    def add_return(self, return_num=None):

        # print("selected_label_alias:", self.selected_label_alias)

        selected_alias = list(self.selected_label_alias.keys())
        selected_node_labels = list(self.selected_label_alias.values())
        
        if return_num:
            random_k = random.randint(1,return_num)
        else:
            random_k = random.randint(1,len(selected_alias))
           
        # print("selected_node_labels:",selected_node_labels)
        # choices = random.sample(self.usable_labels, random_k)
        random_indices = random.sample(range(len(selected_node_labels)), random_k)
        return_list = []
        for i in random_indices:
            current_alias = selected_alias[i]
            current_label = selected_node_labels[i]
            # print("type of current_label:", repr(current_label), type(current_label))
            # print("check if current_label is in self.node_labels", repr(self.node_labels), current_label in self.node_labels)
            current_possible_properties = self.grouped_info[str(current_label).strip()]
            if current_possible_properties:
                property_label = random.choice(list(current_possible_properties.keys()))
                current_return = Clause(f"{current_alias}.{property_label}")
                return_list.append(current_return)

        return Clause("RETURN", return_list)
        # return None
    
    def parts_to_str(self):
        """
        convert the generate query tree into query string with ; separation to get ready for querying the Memgraph client
        """
        final_query_str = 'MATCH'
        for part in self.parts:
            final_query_str = final_query_str + ' ' + str(part)
        final_query_str += ';'
        return final_query_str


    
    ### FOR CROSSOVER RETURN ADJUSTMENT

    def collect_alias_labels(self, tree):
        """ Recursively collect labels from the tree that are usable in the RETURN clause. """
        if isinstance(tree, TreeNode) and isinstance(tree.children, list):
            for child in tree.children:
                # Extract label from the current node's value and add it to usable labels
                child_value = str(child.value)
                # print(child_value, type(child_value))
                # label = self.extract_node_alias(child_value)
                alias, label = self.extract_alias_label(child_value)
                if alias and label:
                    self.selected_label_alias[alias] = label
                    # self.usable_labels.add(label)
            # Recursively process each child
                # self.collect_labels(child)
                
    
    def adjust_return(self, tree):
        """ Adjust the RETURN clause based on the labels collected from the tree. """
        if not isinstance(tree, TreeNode):
            raise TypeError("Expected a tree that is TreeNode instance")
        # Clear existing labels and recollect from the new tree structure
        # self.usable_labels.clear() 
        self.selected_label_alias = {}
        self.collect_alias_labels(tree)
        
        if self.selected_label_alias:
            # random_k = random.randint(1, len(self.usable_labels))
            # choices = random.sample(self.usable_labels, random_k)
            new_return = self.add_return()
            
            if tree.children and isinstance(tree.children[-1], TreeNode) and "RETURN" in str(tree.children[-1].value):
                tree.children[-1] = new_return  # Replace the last child with the new RETURN clause
            else:
                tree.add_child(new_return)  # Add new if no RETURN exists
            # print("updated return:",tree.children[-1])
            return tree
        else:
            return None
        
    
    def generate_query(self, flag=True, return_num=None, part_num=None, hop_p=0.5, where_p=0.5):
        self.reset_per_gen()
        self.dm.reset_depth()
        def alternate_functions(flag):
            if flag:
                return self.add_node(), not flag
            else:
                return self.add_relationship(hop_p), not flag
        if part_num is None:
            part_num = random.randint(1, self.dm._max_depth-2)
        # Keep adding nodes and relationships while depth is within limit
        for _ in range(part_num+1):
            part, flag = alternate_functions(flag)
            if part is None:
                break
            self.parts.append(part)
            # self.current_node.add_child(TreeNode(part))
            self.current_node.add_child(part)
        if self.parts and self.is_relationship(str(self.parts[-1]))==True: #ensure the input part is in string format
            final_node = self.add_node()  # Generate a final node
            if final_node:
                self.parts.append(final_node)
                # print("final_node added:", final_node)
                # self.current_node.add_child(TreeNode(final_node))
                self.current_node.add_child(final_node)
        # Optionally add a WHERE clause to a random node if depth is still under max_depth
        condition = self.add_condition(self.current_node.children, where_p) 
        if condition:
            self.parts.append(condition)
            self.current_node.add_child(condition)
           
        # Add RETURN clause 
        ret = self.add_return(return_num)
        if ret:
            self.parts.append(ret)
            self.current_node.add_child(ret)

        self.query_str = self.parts_to_str()
        self.current_node.depth = self.dm.depth
        return self.current_node, self.query_str #return the treenode type and string type of query

################
#Evolutionary Algorithm#
################

def get_score(individual):
    return individual.score

def deep_copy_tree(node):
    if isinstance(node, TreeNode):
        new_node = TreeNode(copy.deepcopy(node.value))
        new_node.depth = node.depth
        new_node.score = node.score
        new_node.children = [deep_copy_tree(child) for child in node.children]
        return new_node
    elif hasattr(node, '__dict__'):
        return copy.deepcopy(node)
    else:
        return node
def clear_folder(directory_path):
    # Remove the entire directory
    shutil.rmtree(directory_path)
    # Recreate the empty directory
    os.makedirs(directory_path)

class EvolutionaryAlgorithm:
    def __init__(self, qm, depth_manager, population_size, max_depth, max_generation, mut_rate=0.05):
        self.population_size = population_size
        self.max_depth = max_depth
        self.tree_population = []
        self.str_population = []
        self.fitness_scores = {}
        self.query_ids = {}
        self.observed_depths = set() #TODO: clear this set after each initializED population
        self.qm = qm
        self.depth_manager = depth_manager
        self.depth_manager.set_max_depth(self.max_depth)
        # self.database_conn = database_conn
        self.generation = 0
        self.max_generation = max_generation
        self.mut_rate = mut_rate

        # self.host = host
        # self.port = port

        self.valid_queries = [] #store queries with score = 2 across all generations

            
    def population_to_query_list(self, population):
        return [tree.to_querystr() for tree in population]

    def evaluate_population(self, max_workers=4):
        """ Evaluates the entire population and updates fitness scores. """
        self.generation += 1 
        query_list = self.population_to_query_list(self.tree_population)
        x_multi_batch_processing(query_list, batch_size=10, max_workers=max_workers)
        merge_csv_files(directory='./aggregates', output_file='merged_results.csv')
        successful_indices, failed_indices, time_failed_indices = get_indices('merged_results.csv')
        self.update_score(successful_indices, failed_indices, time_failed_indices)


    def initialize_population(self): #TODO: change to generate till size is satisfied
        """ Initializes the population with random depth queries. """
        for _ in range(self.population_size):
            # self.depth_manager.set_max_depth(self.max_depth)
            tree, query = self.qm.generate_query()
            self.tree_population.append(tree)
            self.str_population.append(query)
            self.qm.reset_per_gen()

    def is_valid_query(self, cursor, query):
        """ Execute the query and check if it returns any results. """
        try:
            cursor.execute(query)
            results = cursor.fetchall()
            return len(results) > 0  # Return True if there are results
        except Exception as e:
            print(f"Query failed: {e}")
            return False  # Query failed or returned no results

    def tournament_parent_selection(self, k: int = None):
        """
        Selects the fittest individual from a random sample of the population using a tournament selection approach.

        Parameters:
        - k (int, optional): The number of individuals to sample for the tournament. Defaults to half the population size.

        Returns:
        - The fittest individual from the sampled tournament.
        """
        if k > len(self.tree_population):
            raise ValueError("Sample size k cannot be larger than the population size.")
        tournament = random.sample(self.tree_population, k)
        fittest = max(tournament, key=get_score)
        return fittest
    
    def Selection(self):
        parents = []
        for _, tree in enumerate(self.tree_population):
            if tree.score > 0:
                parents.append(tree)
        return parents
    
    def Reproduction(self,parents: list):
        offspring_pool = []
        for parent in parents:
            offspring = self.mutate_query(parent)
            offspring_pool.append(offspring)
        return offspring_pool

    # def select_parents(self, num_pairs, k):
    #     """
    #     Selects pairs of parents with highest scores for reproduction, ensuring parents within a pair would not repeat.

    #     Parameters:
    #     - num_pairs (int): The number of parent pairs to select.

    #     Returns:
    #     - List of tuples, where each tuple contains two parent individuals.
    #     """
    #     if num_pairs <= 0:
    #         raise ValueError("num_pairs must be a positive integer")
    #     elif num_pairs * 2 > len(self.tree_population):
    #         raise ValueError("Insufficient population to select the requested number of unique pairs")

    #     parents = []
    #     parent_pairs=[]
    #     # selected_individuals = set()  # Keep track of selected individuals

    #     while len(parents) < num_pairs * 2: #and len(selected_individuals) < len(self.tree_population):
    #         parent1 = self.tournament_parent_selection(k)
    #         parent2 = self.tournament_parent_selection(k)
    #         while parent1 == parent2:
    #             parent2 = self.tournament_parent_selection(k)
    #         parent_pair = (parent1,parent2)
    #         parents.append(parent1)
    #         parents.append(parent2)
    #         parent_pairs.append(parent_pair)
    #     return parent_pairs

    # JUL25: implementing mutation
        
    def mutate_query(self, old_tree):
        """
        Randomly mutate either choice based on mut_rate probability below:
        - node label of the query
        - WHERE clause
        And the returned tree query will also have an adjusted RETURN clause
        Now seems like it will automatically update old_tree in self.tree_population
        """
        tree = deep_copy_tree(old_tree)
        self.depth_manager.reset_depth()
        mutations = ['node_label','condition']
        mut_type = random.choice(mutations)
        if type(tree)==TreeNode:
            if mut_type == 'condition':
                # mutated_condition = self.qm.add_condition(for_ea=True)
                for index, element in enumerate(tree.children):
                    if 'WHERE' in str(element.value):
                        children_source = tree.children[:index] + tree.children[index+1:]
                        tree.children[index] = self.qm.add_condition(children_source, for_ea=True)
                    else:
                        mut_type = 'node_label' #ensure mutation takes place for selected queries without WHERE clause
            if mut_type == 'node_label':
                # mutated_node = self.qm.add_node()
                indices = []
                for index, element in enumerate(tree.children):
                    if type(element)==Node:
                        indices.append(index)
                ind = random.choice(indices)
                tree.children[ind] = self.qm.add_node()
            tree = self.qm.adjust_return(tree)
            return tree
        else:
            raise ValueError("The input tree query has to be TreeNode type!")

    # def mutation(self):
    #     num_to_mutate = int(len(self.tree_population) * self.mut_rate)
    #     for _ in range(num_to_mutate):
    #         # Randomly pick an individual to mutate
    #         individual_index = random.randint(0, len(self.tree_population) - 1)
    #         print("mutated index:", individual_index)
    #         # Perform mutation on this individual
    #         self.tree_population[individual_index] = self.mutate_query(self.tree_population[individual_index])
    #     return self.tree_population  

    def swap(self, tree1, tree2):
        if not tree1.children or not tree2.children:
            print("One of the trees does not have children to perform swapping.")
            return
        # Select random subtree indices from both trees
        index1 = random.randint(0, len(tree1.children) - 1)
        index2 = random.randint(0, len(tree2.children) - 1)

        tree1_swap = copy.deepcopy(tree1)
        tree2_swap = copy.deepcopy(tree2)

        # Swap the subtrees
        tree1_swap.children[index1], tree2_swap.children[index2] = \
            tree2_swap.children[index2], tree1_swap.children[index1]
        print("Swapping completed.")
        return tree1_swap, tree2_swap

    def one_point_crossover(self, tree1, tree2):
        if not tree1.children or not tree2.children:
            print("One of the trees does not have children to perform crossover.")
            return
        #get indices that are not relationships as possible crossover point
        node_indices1 = [index for index, child in enumerate(tree1.children) if type(child)!= Relationship]
        node_indices2 = [index for index, child in enumerate(tree2.children) if type(child)!= Relationship]

        #check node existence
        if not node_indices1 or not node_indices2:
            print("No nodes available for crossover in one or both trees.")
            return

        #select random node indices from the filtered lists
        index1 = random.choice(node_indices1[:-1])
        index2 = random.choice(node_indices2[:-1])

        #exchange the subtrees at these indices
        tree1_crossover = copy.deepcopy(tree1)
        tree2_crossover = copy.deepcopy(tree2)

        tree1_crossover.children[index1:], tree2_crossover.children[index2:] = \
            tree2_crossover.children[index2:], tree1_crossover.children[index1:]
        
        #adjust RETURN clause based on exchanged trees
        tree1_crossover = self.qm.adjust_return(tree1_crossover)
        tree2_crossover = self.qm.adjust_return(tree2_crossover)

        print("Crossover and Return clause adjustment completed.")
        return tree1_crossover, tree2_crossover
    
    def update_score(self, successful_indices, failed_indices, time_failed_indices):
        for i, query in enumerate(self.tree_population):
            if i in successful_indices:
                query.score += 2
                self.valid_queries.append(query)
            elif i in failed_indices:
                query.score -= 2
            elif i in time_failed_indices:
                query.score -= 5 #punish timeout


    def output_top_queries(self, top_n):
        """
        Outputs the top N queries from the current population based on fitness scores,
        considering depth diversity and query diversity.
        
        Parameters:
        - top_n (int): Number of top queries to return.
        
        Returns:
        - list: Top N queries as per the defined criteria.
        """
        # Sort the population based on fitness scores

        sorted_population = sorted(self.tree_population, key=lambda x: self.fitness_scores[self.query_ids[x]], reverse=True)
        # Implement logic to ensure diversity if needed, example placeholder:
        # diverse_population = self.ensure_diversity(sorted_population, top_n)
        top_queries_with_scores = [(query, self.fitness_scores[self.query_ids[query]]) for query in sorted_population[:top_n]]
        output_file_path = 'output_queries.txt'
        with open(output_file_path, 'w') as file:
            for tuple in top_queries_with_scores:
                tree = tuple[0]
                querystr = tree.to_querystr()
                file.write(querystr + '\n')
        print("All queries have been written to", output_file_path)
        return top_queries_with_scores
        # return diverse_population[:top_n]


    def reset_ea(self):
        self.depth_manager.reset_depth()
        self.observed_depths = set()
        self.tree_population = []
        self.str_population = []
        self.fitness_scores = {}
        self.query_ids = {}
        self.generation=0
        clear_folder('./aggregates')
        clear_folder('./outputs')
        clear_folder('./input_batch')

    def Evolve(self):
        while self.generation <= self.max_generation:
            self.evaluate_population()
            parents = self.Selection()
            offsprings = self.Reproduction(parents)
            for tree in self.tree_population:
                del tree
            self.tree_population = []
            self.tree_population = offsprings
            self.generation += 1
